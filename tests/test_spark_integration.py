"""Spark integration tests for the RDD K-Means and the end-to-end benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyspark.ml.linalg import SparseVector, Vectors
from pyspark.sql import SparkSession

from kmeans_benchmark.config import (
    BenchmarkConfig,
    DataConfig,
    FeatureConfig,
    KMeansParams,
    TrackingConfig,
)
from kmeans_benchmark.rdd_kmeans import CustomRDDKMeans


def _blobs(n_per: int, seed: int = 0) -> tuple[list[tuple[int, SparseVector]], list[int]]:
    """Three well separated sparse clusters in R^6."""
    rng = np.random.default_rng(seed)
    supports = [[0, 1], [2, 3], [4, 5]]
    data, labels = [], []
    for label, support in enumerate(supports):
        for _ in range(n_per):
            vals = 1.0 + 0.05 * rng.standard_normal(2)
            data.append((len(data), Vectors.sparse(6, support, vals)))
            labels.append(label)
    return data, labels


def _numpy_lloyd(x: np.ndarray, centers: np.ndarray, max_iter: int, tol: float) -> np.ndarray:
    """Reference single-machine Lloyd's algorithm (same update/empty/convergence rules)."""
    for _ in range(max_iter):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        new = centers.copy()
        for k in range(centers.shape[0]):
            if np.any(assign == k):
                new[k] = x[assign == k].mean(0)
        shift = np.linalg.norm(new - centers, axis=1).max()
        centers = new
        if shift <= tol:
            break
    return centers


def test_rdd_kmeans_matches_numpy_reference(spark: SparkSession) -> None:
    data, labels = _blobs(50)
    rdd = spark.sparkContext.parallelize(data, 4)
    params = KMeansParams(k=3, max_iter=25, tol=1e-6, seed=7)
    est = CustomRDDKMeans(params)

    init = est._init_centers(rdd)
    dense = np.vstack([v.toArray() for _, v in data])
    expected = _numpy_lloyd(dense, init.copy(), params.max_iter, params.tol)

    model, assignments = est.fit_predict(spark, rdd)
    np.testing.assert_allclose(model.centers, expected, atol=1e-10)
    assert model.converged

    assigned = dict(assignments.collect())
    assert len(assigned) == len(data)
    # Perfect recovery up to label permutation.
    mapping = {assigned[i]: labels[i] for i in range(len(data))}
    assert len(mapping) == 3
    assert all(mapping[assigned[i]] == labels[i] for i in range(len(data)))
    assert model.history[-1].cost == pytest.approx(model.training_cost)


def test_rdd_kmeans_cosine(spark: SparkSession) -> None:
    data, _ = _blobs(30, seed=1)
    rdd = spark.sparkContext.parallelize(data, 3)
    model = CustomRDDKMeans(KMeansParams(k=3, max_iter=20, seed=3, distance_measure="cosine")).fit(spark, rdd)
    np.testing.assert_allclose(np.linalg.norm(model.centers, axis=1), 1.0)


def test_end_to_end_benchmark(spark: SparkSession, tmp_path: Path) -> None:
    from kmeans_benchmark.benchmark import cleanup, run_benchmark

    cfg = BenchmarkConfig(
        data=DataConfig(
            input_path=str(tmp_path / "missing_corpus"),
            materialize_synthetic=False,
            synthetic_num_docs=400,
            synthetic_num_partitions=2,
        ),
        features=FeatureConfig(vocab_size=512, min_df=2.0),
        kmeans=KMeansParams(k=4, max_iter=6, tol=1e-3, seed=11),
        tracking=TrackingConfig(
            experiment_name="pytest_kmeans",
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            log_full_pipeline=True,
        ),
    )
    report = run_benchmark(spark, cfg, track=True)

    for res in (report.rdd, report.mllib):
        assert -1.0 <= res.silhouette_score <= 1.0
        assert sum(res.cluster_sizes.values()) == res.predictions.count()
        assert res.centers.shape[0] == 4
        assert res.ari_vs_ground_truth is not None
        print(res.implementation_type, res.total_execution_time_seconds, res.silhouette_score,
              res.ari_vs_ground_truth, res.stage_metrics)
    assert report.parent_run_id is not None
    assert -1.0 <= report.ari_rdd_vs_mllib <= 1.0
    print("ARI rdd/mllib", report.ari_rdd_vs_mllib, "speedup", report.speedup_mllib_over_rdd)
    cleanup(report)
