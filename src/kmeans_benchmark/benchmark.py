"""Benchmark orchestrator: RDD vs MLlib K-Means, wrapped in MLflow runs.

Run layout in MLflow::

    kmeans_benchmark (parent)          dataset params, comparison metrics
     ├── rdd_kmeans    (nested)        params + metrics + centroids + convergence curve
     └── mllib_kmeans  (nested)        params + metrics + centroids + logged Spark model

Timed section (identical scope for both implementations, starting from the
same cached TF-IDF DataFrame): ``fit`` + ``transform`` + materialising the
cached predictions. Evaluation (silhouette, ARI) and MLflow I/O are excluded.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import mlflow
import numpy as np
from pyspark import StorageLevel
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import DataFrame, SparkSession

from kmeans_benchmark.config import (
    BenchmarkConfig,
    DataConfig,
    FeatureConfig,
    KMeansParams,
    TrackingConfig,
)
from kmeans_benchmark.evaluation import (
    adjusted_rand_index,
    agreement_between,
    cluster_sizes,
    silhouette_score,
    top_terms_per_cluster,
)
from kmeans_benchmark.ingestion import load_corpus
from kmeans_benchmark.metrics import StageMetrics, StageMetricsCollector
from kmeans_benchmark.mllib_kmeans import fit_mllib_kmeans
from kmeans_benchmark.preprocessing import TfidfResult, build_tfidf, to_vector_rdd
from kmeans_benchmark.rdd_kmeans import CustomRDDKMeans, IterationStats, RDDKMeansModel
from kmeans_benchmark.spark_session import get_spark, ship_package_to_executors
from kmeans_benchmark.tracking import (
    configure_mlflow,
    log_benchmark_context,
    log_implementation_run,
    log_mllib_model,
)
from kmeans_benchmark.vector_ops import FloatArray

logger = logging.getLogger(__name__)

ImplementationType = Literal["RDD", "MLlib"]


@dataclass(slots=True)
class ImplementationResult:
    """Everything measured for one implementation."""

    implementation_type: ImplementationType
    model: RDDKMeansModel | KMeansModel
    predictions: DataFrame  # cached: doc_id, label, features, prediction
    centers: FloatArray  # float64[K, D]
    num_iterations: int
    converged: bool | None
    training_cost: float
    total_execution_time_seconds: float
    stage_metrics: StageMetrics | None
    history: list[IterationStats] = field(default_factory=list)
    silhouette_score: float = math.nan
    ari_vs_ground_truth: float | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    top_terms: dict[int, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkReport:
    """Final comparison returned by :func:`run_benchmark`."""

    rdd: ImplementationResult
    mllib: ImplementationResult
    ari_rdd_vs_mllib: float
    speedup_mllib_over_rdd: float
    parent_run_id: str | None = None

    def summary_rows(self) -> list[dict[str, object]]:
        """Tabular summary (one dict per implementation) for display()."""
        rows: list[dict[str, object]] = []
        for r in (self.rdd, self.mllib):
            sm = r.stage_metrics
            rows.append(
                {
                    "implementation": r.implementation_type,
                    "time_s": round(r.total_execution_time_seconds, 3),
                    "iterations": r.num_iterations,
                    "silhouette": round(r.silhouette_score, 5),
                    "training_cost": round(r.training_cost, 4),
                    "ari_vs_truth": None if r.ari_vs_ground_truth is None else round(r.ari_vs_ground_truth, 4),
                    "shuffle_read_bytes": sm.shuffleReadBytes if sm else None,
                    "shuffle_write_bytes": sm.shuffleWriteBytes if sm else None,
                }
            )
        return rows


# --------------------------------------------------------------------------- #
# Individual runs
# --------------------------------------------------------------------------- #
def run_rdd_kmeans(spark: SparkSession, tfidf: TfidfResult, params: KMeansParams) -> ImplementationResult:
    """Time + measure the from-scratch RDD implementation."""
    vectors = to_vector_rdd(tfidf.features_df)
    with StageMetricsCollector(spark) as measure:
        model = CustomRDDKMeans(params).fit(spark, vectors)  # persists `vectors`
        predictions = model.transform(spark, tfidf.features_df).persist(StorageLevel.MEMORY_AND_DISK)
        predictions.count()
    vectors.unpersist()
    return ImplementationResult(
        implementation_type="RDD",
        model=model,
        predictions=predictions,
        centers=model.centers,
        num_iterations=model.num_iterations,
        converged=model.converged,
        training_cost=model.training_cost,
        total_execution_time_seconds=measure.elapsed_seconds,
        stage_metrics=measure.metrics,
        history=model.history,
    )


def run_mllib_kmeans(spark: SparkSession, tfidf: TfidfResult, params: KMeansParams) -> ImplementationResult:
    """Time + measure the MLlib DataFrame implementation."""
    with StageMetricsCollector(spark) as measure:
        res = fit_mllib_kmeans(tfidf.features_df, params)
    return ImplementationResult(
        implementation_type="MLlib",
        model=res.model,
        predictions=res.predictions,
        centers=np.vstack(res.model.clusterCenters()),
        num_iterations=res.num_iterations,
        # MLlib does not expose a convergence flag; stopping early implies convergence.
        converged=True if res.num_iterations < params.max_iter else None,
        training_cost=res.training_cost,
        total_execution_time_seconds=measure.elapsed_seconds,
        stage_metrics=measure.metrics,
    )


def evaluate(result: ImplementationResult, tfidf: TfidfResult, cfg: BenchmarkConfig) -> None:
    """Fill quality metrics in place (outside the timed section)."""
    preds = result.predictions
    result.silhouette_score = silhouette_score(preds, cfg.kmeans)
    result.cluster_sizes = cluster_sizes(preds)
    result.top_terms = top_terms_per_cluster(
        result.centers, tfidf.vocabulary, cfg.tracking.top_terms_per_cluster
    )
    if preds.where("label IS NOT NULL").limit(1).count() > 0:
        result.ari_vs_ground_truth = adjusted_rand_index(preds, "label", "prediction")
    logger.info(
        "[%s] time=%.2fs silhouette=%.4f iters=%d",
        result.implementation_type, result.total_execution_time_seconds,
        result.silhouette_score, result.num_iterations,
    )


# --------------------------------------------------------------------------- #
# End-to-end benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(
    spark: SparkSession,
    cfg: BenchmarkConfig,
    corpus_df: DataFrame | None = None,
    tfidf: TfidfResult | None = None,
    track: bool = True,
) -> BenchmarkReport:
    """Execute Phases 1-4 end to end.

    Args:
        spark: Active SparkSession.
        cfg: Benchmark configuration.
        corpus_df: Pre-loaded corpus (skips ingestion when given).
        tfidf: Pre-computed TF-IDF result (skips Phase 1 when given).
        track: Log everything to MLflow.
    """
    ship_package_to_executors(spark)
    if tfidf is None:
        corpus_df = corpus_df if corpus_df is not None else load_corpus(spark, cfg.data)
        tfidf = build_tfidf(corpus_df, cfg.features)

    if not track:
        rdd_res, mllib_res = _run_both(spark, tfidf, cfg)
        return _build_report(rdd_res, mllib_res, None)

    configure_mlflow(cfg.tracking)
    with mlflow.start_run(run_name="kmeans_benchmark") as parent:
        log_benchmark_context(spark, cfg, tfidf)

        with mlflow.start_run(run_name="rdd_kmeans", nested=True):
            rdd_res = run_rdd_kmeans(spark, tfidf, cfg.kmeans)
            evaluate(rdd_res, tfidf, cfg)
            log_implementation_run(rdd_res, cfg, tfidf)

        with mlflow.start_run(run_name="mllib_kmeans", nested=True):
            mllib_res = run_mllib_kmeans(spark, tfidf, cfg.kmeans)
            evaluate(mllib_res, tfidf, cfg)
            log_implementation_run(mllib_res, cfg, tfidf)
            assert isinstance(mllib_res.model, KMeansModel)
            log_mllib_model(mllib_res.model, tfidf, corpus_df, cfg.tracking)

        report = _build_report(rdd_res, mllib_res, parent.info.run_id)
        mlflow.log_metrics(
            {
                "speedup_mllib_over_rdd": report.speedup_mllib_over_rdd,
                "ari_rdd_vs_mllib": report.ari_rdd_vs_mllib,
                "rdd_total_execution_time_seconds": rdd_res.total_execution_time_seconds,
                "mllib_total_execution_time_seconds": mllib_res.total_execution_time_seconds,
                "rdd_silhouette_score": rdd_res.silhouette_score,
                "mllib_silhouette_score": mllib_res.silhouette_score,
            }
        )
        mlflow.log_dict({"rows": report.summary_rows()}, "benchmark_summary.json")
    return report


def _run_both(
    spark: SparkSession, tfidf: TfidfResult, cfg: BenchmarkConfig
) -> tuple[ImplementationResult, ImplementationResult]:
    rdd_res = run_rdd_kmeans(spark, tfidf, cfg.kmeans)
    evaluate(rdd_res, tfidf, cfg)
    mllib_res = run_mllib_kmeans(spark, tfidf, cfg.kmeans)
    evaluate(mllib_res, tfidf, cfg)
    return rdd_res, mllib_res


def _build_report(
    rdd_res: ImplementationResult, mllib_res: ImplementationResult, run_id: str | None
) -> BenchmarkReport:
    return BenchmarkReport(
        rdd=rdd_res,
        mllib=mllib_res,
        ari_rdd_vs_mllib=agreement_between(rdd_res.predictions, mllib_res.predictions),
        speedup_mllib_over_rdd=rdd_res.total_execution_time_seconds
        / max(mllib_res.total_execution_time_seconds, 1e-9),
        parent_run_id=run_id,
    )


def cleanup(report: BenchmarkReport, tfidf: TfidfResult | None = None) -> None:
    """Release cached DataFrames and broadcasts created by the benchmark."""
    report.rdd.predictions.unpersist()
    report.mllib.predictions.unpersist()
    if isinstance(report.rdd.model, RDDKMeansModel):
        report.rdd.model.release()
    if tfidf is not None:
        tfidf.features_df.unpersist()


# --------------------------------------------------------------------------- #
# CLI (local / spark-submit / Databricks job "Python script" task)
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> tuple[BenchmarkConfig, bool]:
    d_cfg, f_cfg, k_cfg, t_cfg = DataConfig(), FeatureConfig(), KMeansParams(), TrackingConfig()
    ap = argparse.ArgumentParser(description="RDD vs MLlib K-Means benchmark")
    ap.add_argument("--input-path", default=d_cfg.input_path)
    ap.add_argument("--num-docs", type=int, default=d_cfg.synthetic_num_docs)
    ap.add_argument("--num-partitions", type=int, default=d_cfg.synthetic_num_partitions)
    ap.add_argument("--in-memory", action="store_true", help="Do not write the synthetic corpus")
    ap.add_argument("--vocab-size", type=int, default=f_cfg.vocab_size)
    ap.add_argument("--k", type=int, default=k_cfg.k)
    ap.add_argument("--max-iter", type=int, default=k_cfg.max_iter)
    ap.add_argument("--tol", type=float, default=k_cfg.tol)
    ap.add_argument("--seed", type=int, default=k_cfg.seed)
    ap.add_argument("--distance", choices=["euclidean", "cosine"], default=k_cfg.distance_measure)
    ap.add_argument("--experiment", default=t_cfg.experiment_name)
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    cfg = BenchmarkConfig(
        data=DataConfig(
            input_path=args.input_path,
            synthetic_num_docs=args.num_docs,
            synthetic_num_partitions=args.num_partitions,
            materialize_synthetic=not args.in_memory,
            seed=args.seed,
        ),
        features=FeatureConfig(vocab_size=args.vocab_size),
        kmeans=KMeansParams(
            k=args.k, max_iter=args.max_iter, tol=args.tol, seed=args.seed,
            distance_measure=args.distance,
        ),
        tracking=TrackingConfig(experiment_name=args.experiment, tracking_uri=args.tracking_uri),
    )
    return cfg, not args.no_mlflow


def main(argv: list[str] | None = None) -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    cfg, track = _parse_args(argv)
    spark = get_spark()
    report = run_benchmark(spark, cfg, track=track)
    for row in report.summary_rows():
        print(row)
    print(f"ARI(RDD, MLlib) = {report.ari_rdd_vs_mllib:.4f} | "
          f"speed-up MLlib/RDD = x{report.speedup_mllib_over_rdd:.2f}")


if __name__ == "__main__":
    main()
