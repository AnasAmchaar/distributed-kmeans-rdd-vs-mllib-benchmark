"""Clustering quality metrics: Silhouette, cluster sizes, ARI, top terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from kmeans_benchmark.config import KMeansParams
from kmeans_benchmark.preprocessing import FEATURES_COL
from kmeans_benchmark.vector_ops import FloatArray


def silhouette_score(
    predictions: DataFrame,
    params: KMeansParams,
    features_col: str = FEATURES_COL,
    prediction_col: str = "prediction",
) -> float:
    """Silhouette via ``ClusteringEvaluator`` (same metric for both implementations).

    The distance measure matches the K-Means objective (``squaredEuclidean``
    or ``cosine``). Spark computes the *simplified* silhouette with
    per-cluster aggregates: ``O(N)`` with one aggregation (**shuffle**) job,
    not the ``O(N²)`` textbook definition.
    """
    evaluator = ClusteringEvaluator(
        featuresCol=features_col,
        predictionCol=prediction_col,
        metricName="silhouette",
        distanceMeasure=params.evaluator_distance,
    )
    return float(evaluator.evaluate(predictions))


def cluster_sizes(predictions: DataFrame, prediction_col: str = "prediction") -> dict[int, int]:
    """``{cluster_id: n_docs}`` (small ``groupBy`` -> shuffle of K rows/partition)."""
    rows = predictions.groupBy(prediction_col).count().collect()
    return {int(r[prediction_col]): int(r["count"]) for r in sorted(rows, key=lambda r: r[0])}


def _comb2(x: FloatArray) -> FloatArray:
    return x * (x - 1.0) / 2.0


def adjusted_rand_index_from_contingency(table: FloatArray) -> float:
    """ARI from a contingency matrix ``n_ij`` (rows: partition A, cols: partition B)."""
    n = table.sum()
    if n < 2:
        return 1.0
    sum_ij = _comb2(table).sum()
    sum_a = _comb2(table.sum(axis=1)).sum()
    sum_b = _comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / _comb2(np.asarray(n))
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    return 1.0 if denom == 0 else float((sum_ij - expected) / denom)


def adjusted_rand_index(df: DataFrame, col_a: str, col_b: str) -> float:
    """Adjusted Rand Index between two labelings stored in the *same* DataFrame.

    Only the ``|A| x |B|`` contingency table (``groupBy(a, b).count()``) is
    collected to the driver, so this scales to any ``N``. Rows with a null
    label are ignored.
    """
    rows = df.where(F.col(col_a).isNotNull() & F.col(col_b).isNotNull()) \
        .groupBy(col_a, col_b).count().collect()
    if not rows:
        return float("nan")
    a_values = sorted({r[0] for r in rows}, key=str)
    b_values = sorted({r[1] for r in rows}, key=str)
    a_idx = {v: i for i, v in enumerate(a_values)}
    b_idx = {v: i for i, v in enumerate(b_values)}
    table = np.zeros((len(a_values), len(b_values)), dtype=np.float64)
    for a, b, count in rows:
        table[a_idx[a], b_idx[b]] = count
    return adjusted_rand_index_from_contingency(table)


def agreement_between(
    pred_a: DataFrame, pred_b: DataFrame, key: str = "doc_id", prediction_col: str = "prediction"
) -> float:
    """ARI between two implementations' assignments (join on ``doc_id`` -> shuffle)."""
    joined = pred_a.select(key, F.col(prediction_col).alias("pred_a")).join(
        pred_b.select(key, F.col(prediction_col).alias("pred_b")), on=key
    )
    return adjusted_rand_index(joined, "pred_a", "pred_b")


def top_terms_per_cluster(
    centers: FloatArray | Sequence[FloatArray], vocabulary: Sequence[str], n_terms: int
) -> dict[int, list[str]]:
    """Highest-weight vocabulary terms of each centroid (cluster interpretation)."""
    matrix = np.asarray(centers, dtype=np.float64)
    top = np.argsort(-matrix, axis=1)[:, :n_terms]
    return {k: [vocabulary[j] for j in row] for k, row in enumerate(top)}

