"""Phase 4 — MLflow tracking helpers (params, metrics, artifacts, models)."""

from __future__ import annotations

import inspect
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
from pyspark.ml import PipelineModel
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import DataFrame, SparkSession

from kmeans_benchmark.config import BenchmarkConfig, TrackingConfig
from kmeans_benchmark.spark_session import get_spark_context, has_rdd_support

if TYPE_CHECKING:  # avoid a circular import at runtime
    from kmeans_benchmark.benchmark import ImplementationResult
    from kmeans_benchmark.preprocessing import TfidfResult

logger = logging.getLogger(__name__)


def configure_mlflow(cfg: TrackingConfig) -> str:
    """Point MLflow at the tracking server and select the experiment.

    On Databricks the tracking URI defaults to the workspace (``databricks``)
    and ``experiment_name`` must be a workspace path, e.g.
    ``/Users/<you>/kmeans_benchmark``. Locally it defaults to ``./mlruns``.

    Returns:
        Experiment id.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    experiment = mlflow.set_experiment(cfg.experiment_name)
    return experiment.experiment_id


def log_benchmark_context(spark: SparkSession, cfg: BenchmarkConfig, tfidf: TfidfResult) -> None:
    """Parent-run params: dataset shape, feature config, Spark environment."""
    sc = get_spark_context(spark)
    num_parts = (
        tfidf.features_df.rdd.getNumPartitions()
        if has_rdd_support(spark)
        else (cfg.features.num_partitions or cfg.data.synthetic_num_partitions)
    )
    mlflow.log_params(
        {
            "num_docs": tfidf.num_docs,
            "num_features": tfidf.num_features,
            "num_partitions": num_parts,
            "spark_version": spark.version,
            "default_parallelism": sc.defaultParallelism,
            "input_path": cfg.data.input_path,
            **{k: v for k, v in cfg.to_flat_dict().items() if k.startswith("features.")},
        }
    )


def log_implementation_run(
    result: ImplementationResult, cfg: BenchmarkConfig, tfidf: TfidfResult
) -> None:
    """Log one implementation's params / metrics / artifacts into the *active* run.

    Params (required): ``K_clusters``, ``maxIter``, ``tolerance``,
    ``implementation_type`` (+ seed, distance, init mode).
    Metrics (required): ``silhouette_score``, ``total_execution_time_seconds``,
    ``shuffle_read_bytes``, ``shuffle_write_bytes`` (+ cost, iterations, ...).
    """
    p = cfg.kmeans
    mlflow.log_params(
        {
            "K_clusters": p.k,
            "maxIter": p.max_iter,
            "tolerance": p.tol,
            "implementation_type": result.implementation_type,
            "seed": p.seed,
            "distance_measure": p.distance_measure,
            "init_mode": "random",
        }
    )
    mlflow.set_tag("shuffle_metrics_source", "spark_rest_api" if result.stage_metrics else "unavailable")

    metrics: dict[str, float] = {
        "silhouette_score": result.silhouette_score,
        "total_execution_time_seconds": result.total_execution_time_seconds,
        "num_iterations": float(result.num_iterations),
        "training_cost": result.training_cost,
        "converged": float(result.converged) if result.converged is not None else float("nan"),
        "empty_clusters": float(p.k - len(result.cluster_sizes)),
    }
    if result.ari_vs_ground_truth is not None:
        metrics["ari_vs_ground_truth"] = result.ari_vs_ground_truth
    if result.stage_metrics is not None:
        metrics.update(result.stage_metrics.as_mlflow_metrics())
    else:  # explicit sentinel so dashboards don't silently drop the column
        metrics.update({"shuffle_read_bytes": -1.0, "shuffle_write_bytes": -1.0})
    mlflow.log_metrics({k: v for k, v in metrics.items() if not np.isnan(v)})

    # Per-iteration convergence curve (RDD implementation only).
    for it in result.history:
        mlflow.log_metrics(
            {
                "iter_max_center_shift": it.max_center_shift,
                "iter_cost": it.cost,
                "iter_seconds": it.duration_seconds,
            },
            step=it.iteration,
        )

    mlflow.log_dict({str(k): v for k, v in result.cluster_sizes.items()}, "cluster_sizes.json")
    mlflow.log_dict(
        {str(k): v for k, v in result.top_terms.items()}, "top_terms_per_cluster.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"centroids_{result.implementation_type.lower()}.npy"
        np.save(path, result.centers)
        mlflow.log_artifact(str(path), artifact_path="centroids")


def _log_spark_model(model: Any, name: str, **kwargs: Any) -> None:
    """``mlflow.spark.log_model`` compatible with MLflow 2.x (``artifact_path``)
    and MLflow 3.x (``name``)."""
    params = inspect.signature(mlflow.spark.log_model).parameters
    call_kwargs = {"name": name, **kwargs} if "name" in params else {"artifact_path": name, **kwargs}
    try:
        mlflow.spark.log_model(model, **call_kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if (
            "winutils" in msg.lower()
            or "UnsatisfiedLinkError" in msg
            or "JVM_ATTRIBUTE_NOT_SUPPORTED" in msg
            or "UC_VOLUME" in msg
            or "dfs_tmpdir" in msg
        ):
            logger.warning("Skipping mlflow.spark.log_model in current environment: %s", exc)
            mlflow.set_tag("spark_model_log_status", "skipped_environment_restriction")
            return
        raise


def log_mllib_model(
    kmeans_model: KMeansModel,
    tfidf: TfidfResult,
    corpus_sample: DataFrame | None,
    cfg: TrackingConfig,
) -> None:
    """Log the trained MLlib model with ``mlflow.spark.log_model``.

    With ``cfg.log_full_pipeline`` the logged artifact is
    ``PipelineModel([tokenizer, stop-words, CountVectorizer, IDF, Normalizer,
    KMeansModel])`` so the served model scores **raw text** directly; a
    signature ``text -> prediction`` is inferred from ``corpus_sample``.
    """
    if not cfg.log_full_pipeline:
        _log_spark_model(kmeans_model, "mllib_kmeans_model")
        return

    full = PipelineModel(stages=[*tfidf.pipeline_model.stages, kmeans_model])
    extra: dict[str, Any] = {}
    if corpus_sample is not None:
        try:
            from mlflow.models import infer_signature

            sample_in = corpus_sample.select("text").limit(5)
            sample_out = full.transform(sample_in).select("prediction")
            extra["signature"] = infer_signature(sample_in.toPandas(), sample_out.toPandas())
        except Exception as exc:  # noqa: BLE001 - signature is optional
            logger.warning("Could not infer model signature: %s", exc)
    _log_spark_model(full, "mllib_kmeans_pipeline", **extra)
