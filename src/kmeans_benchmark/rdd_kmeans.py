"""Phase 2 — K-Means implemented from scratch with PySpark RDD primitives only.

Algorithm (Lloyd's iterations, MapReduce formulation)::

          driver                                   executors
    ┌──────────────────┐   broadcast(C, ||C||²)   ┌───────────────────────────────┐
    │ C_t  (K x D)     │ ───────────────────────▶ │ MAP:  x ↦ (argmin_k d(x,c_k), │
    │                  │                          │               (x, 1))         │
    │                  │                          │ COMBINE (map-side, per part.):│
    │                  │                          │   dense sum_k += x ; n_k += 1 │
    │                  │      shuffle (K keys)    │ REDUCE (aggregateByKey):      │
    │ C_{t+1} = sum/n  │ ◀─────────────────────── │   merge partial (sum_k, n_k)  │
    └──────────────────┘   collectAsMap (K rows)  └───────────────────────────────┘

Spark DAG per iteration: **one job, two stages** separated by a single shuffle
boundary (``aggregateByKey``). Thanks to map-side combining the shuffle volume
is ``O(P * K * D * 8 bytes)`` (P = input partitions) — independent of the
number of documents ``N``. The lineage does *not* grow across iterations
because centroids live on the driver and every iteration restarts from the
same cached RDD, so no checkpointing is required.

Important:
    All functions shipped to executors are module-level and receive only
    picklable arguments (broadcast handles, accumulators, primitives). Never
    capture ``self`` or a ``SparkContext`` in a closure — it is not serialisable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np
from pyspark import RDD, Broadcast, StorageLevel
from pyspark.accumulators import Accumulator
from pyspark.ml.linalg import Vector, VectorUDT
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType

from kmeans_benchmark import vector_ops as vo
from kmeans_benchmark.config import KMeansParams
from kmeans_benchmark.vector_ops import FloatArray

logger = logging.getLogger(__name__)

VectorRDD = RDD[tuple[int, Vector]]
"""Input: ``RDD[(doc_id, vector[D])]``."""
AssignmentRDD = RDD[tuple[int, int]]
"""Output: ``RDD[(doc_id, cluster_index)]``."""

# Broadcast payload: (centers float64[K, D], squared norms float64[K])
CentersPayload = tuple[FloatArray, FloatArray]
# Combiner value: (dense running sum float64[D], running count)
ClusterAcc = tuple[FloatArray, int]


# =========================================================================== #
# Executor-side functions (module level => cheaply picklable)
# =========================================================================== #
def _map_to_closest(
    vector: Vector,
    bc_centers: Broadcast[CentersPayload],
    distance_measure: str,
    cost_acc: Accumulator[float],
) -> tuple[int, tuple[Vector, int]]:
    """MAP phase: ``x -> (closest_centroid_index, (x, 1))``.

    ``bc_centers.value`` is deserialised **once per Python worker** and then
    served from the worker-local cache, so the per-record overhead is a single
    ``O(K * nnz)`` sparse gather-dot. The distance to the chosen centroid is
    added to ``cost_acc`` (within-cluster cost, same definition as MLlib).

    Warning:
        Accumulator updates inside a *transformation* may be applied more than
        once if a task is retried / speculatively executed. The cost is
        therefore a diagnostic, never used for control flow.
    """
    centers, sq_norms = bc_centers.value
    best, cost = vo.closest_center(vector, centers, sq_norms, distance_measure)
    cost_acc.add(cost)
    return best, (vector, 1)


def _seq_op(acc: ClusterAcc, value: tuple[Vector, int]) -> ClusterAcc:
    """aggregateByKey ``seqOp`` — fold one ``(x, 1)`` into the partition-local
    dense accumulator **in place** (Spark explicitly allows mutating ``acc``).

    Densifying here (instead of summing ``SparseVector`` objects pairwise)
    avoids re-allocating ever-growing sparse vectors: ``O(nnz)`` per record.
    """
    vec_sum, count = acc
    vector, one = value
    vo.add_into(vec_sum, vector)
    return vec_sum, count + one


def _comb_op(left: ClusterAcc, right: ClusterAcc) -> ClusterAcc:
    """aggregateByKey ``combOp`` — merge two partial cluster sums (reduce side)."""
    left_sum, left_count = left
    right_sum, right_count = right
    left_sum += right_sum
    return left_sum, left_count + right_count


def _predict_one(
    vector: Vector, bc_centers: Broadcast[CentersPayload], distance_measure: str
) -> int:
    """Return the index of the nearest centroid for a single vector."""
    centers, sq_norms = bc_centers.value
    return vo.closest_center(vector, centers, sq_norms, distance_measure)[0]


def _assign_record(
    record: tuple[int, Vector], bc_centers: Broadcast[CentersPayload], distance_measure: str
) -> tuple[int, int]:
    """``(doc_id, x) -> (doc_id, cluster_index)``."""
    doc_id, vector = record
    return doc_id, _predict_one(vector, bc_centers, distance_measure)


def _append_prediction(
    rows: Iterable[Row],
    bc_centers: Broadcast[CentersPayload],
    distance_measure: str,
    features_col: str,
) -> Iterator[tuple[Any, ...]]:
    """``mapPartitions`` kernel: append the predicted cluster to every Row."""
    for row in rows:
        yield (*row, _predict_one(row[features_col], bc_centers, distance_measure))


# =========================================================================== #
# Model & training statistics
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class IterationStats:
    """Diagnostics of one Lloyd iteration (logged to MLflow with ``step``)."""

    iteration: int
    max_center_shift: float
    cost: float
    empty_clusters: int
    duration_seconds: float


@dataclass(slots=True)
class RDDKMeansModel:
    """Fitted from-scratch K-Means model.

    Attributes:
        centers: ``float64[K, D]`` final centroids.
        distance_measure: ``"euclidean"`` | ``"cosine"``.
        num_iterations: Lloyd iterations actually executed.
        converged: ``True`` if every centroid shift <= ``tol`` before ``max_iter``.
        training_cost: Within-cluster cost of the last iteration (sum of squared
            distances, or of cosine distances) — same semantics as MLlib's
            ``KMeansModel.summary.trainingCost``.
        history: Per-iteration :class:`IterationStats`.
    """

    centers: FloatArray
    distance_measure: str
    num_iterations: int
    converged: bool
    training_cost: float
    history: list[IterationStats] = field(default_factory=list)
    _bc: Broadcast[CentersPayload] | None = field(default=None, repr=False)

    @property
    def k(self) -> int:
        return int(self.centers.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.centers.shape[1])

    def predict(self, vector: Vector) -> int:
        """Driver-local prediction for a single vector."""
        sq = np.einsum("ij,ij->i", self.centers, self.centers)
        return vo.closest_center(vector, self.centers, sq, self.distance_measure)[0]

    def broadcast(self, spark: SparkSession) -> Broadcast[CentersPayload]:
        """Lazily broadcast ``(centers, ||centers||²)`` once for all scoring jobs."""
        if self._bc is None:
            sq = np.einsum("ij,ij->i", self.centers, self.centers)
            self._bc = spark.sparkContext.broadcast((self.centers, sq))
        return self._bc

    def assign(self, spark: SparkSession, data: VectorRDD) -> AssignmentRDD:
        """Cluster assignments ``RDD[(doc_id, cluster_index)]``.

        DAG: a single narrow ``map`` — **no shuffle**. Lazy until an action.
        """
        bc, dm = self.broadcast(spark), self.distance_measure
        return data.map(partial(_assign_record, bc_centers=bc, distance_measure=dm))

    def transform(
        self,
        spark: SparkSession,
        df: DataFrame,
        features_col: str = "features",
        prediction_col: str = "prediction",
    ) -> DataFrame:
        """MLlib-like ``transform``: append an ``int`` prediction column to ``df``.

        Every input column is preserved (e.g. ``doc_id``, ``label``), so the
        result can be fed straight into ``ClusteringEvaluator``.

        DAG: ``df.rdd.mapPartitions`` (narrow, **no shuffle**) +
        ``createDataFrame`` — rows round-trip JVM -> Python -> JVM, which is the
        intrinsic serialisation cost of a Python-side model.
        """
        bc, dm = self.broadcast(spark), self.distance_measure
        schema = StructType(
            [*df.schema.fields, StructField(prediction_col, IntegerType(), nullable=False)]
        )
        rdd = df.rdd.mapPartitions(
            partial(_append_prediction, bc_centers=bc, distance_measure=dm, features_col=features_col)
        )
        return spark.createDataFrame(rdd, schema)

    def release(self) -> None:
        """Destroy the scoring broadcast (frees executor + driver memory)."""
        if self._bc is not None:
            self._bc.destroy()
            self._bc = None


# =========================================================================== #
# Estimator
# =========================================================================== #
class CustomRDDKMeans:
    """Distributed K-Means using only RDD transformations / actions.

    Input shape:
        ``RDD[(doc_id: int, x: Vector[D])]`` — typically sparse, L2-normalised
        TF-IDF rows produced by :func:`kmeans_benchmark.preprocessing.to_vector_rdd`.

    Output shape:
        :class:`RDDKMeansModel` (centroids ``float64[K, D]``) and, via
        :meth:`fit_predict`, ``RDD[(doc_id, cluster_index)]``.

    Spark DAG implications:
        * ``fit`` persists the input once (``storage_level``) and materialises
          it with ``count()`` -> the TF-IDF lineage is never recomputed.
        * Initialisation: ``takeSample`` = 1-2 jobs, no shuffle.
        * Each iteration: 1 job = ShuffleMapStage (map + map-side combine) +
          ResultStage (``aggregateByKey`` merge, ``collectAsMap``).
          Shuffle bytes ≈ ``P * K * D * 8`` per iteration.
        * Broadcast size ≈ ``K * D * 8`` bytes per iteration; the previous
          iteration's broadcast is destroyed eagerly to avoid executor
          memory build-up.

    Args:
        params: Hyper-parameters shared with the MLlib run.
        storage_level: Persistence level for the input vectors.
        reduce_partitions: Number of reduce-side partitions for
            ``aggregateByKey`` (defaults to ``min(K, defaultParallelism)`` —
            there are only ``K`` keys, more partitions would be empty).
        iteration_callback: Optional hook ``f(stats)`` called after every
            iteration (e.g. live MLflow logging).
    """

    def __init__(
        self,
        params: KMeansParams,
        storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
        reduce_partitions: int | None = None,
        iteration_callback: Callable[[IterationStats], None] | None = None,
    ) -> None:
        self.params = params
        self.storage_level = storage_level
        self.reduce_partitions = reduce_partitions
        self.iteration_callback = iteration_callback

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, spark: SparkSession, data: VectorRDD) -> RDDKMeansModel:
        """Run Lloyd's algorithm until convergence (``tol``) or ``max_iter``.

        Args:
            spark: Active SparkSession.
            data: ``RDD[(doc_id, Vector[D])]``.

        Returns:
            Fitted :class:`RDDKMeansModel`.
        """
        sc = spark.sparkContext
        p = self.params

        vectors: RDD[Vector] = data.values()
        if not data.is_cached:
            data.persist(self.storage_level)
        n_docs = data.count()  # action: materialise the cache before iterating
        if n_docs < p.k:
            raise ValueError(f"Need at least k={p.k} documents, got {n_docs}")

        centers = self._init_centers(data)
        n_partitions = self.reduce_partitions or max(1, min(p.k, sc.defaultParallelism))
        logger.info(
            "RDD K-Means: N=%d, D=%d, K=%d, input partitions=%d",
            n_docs, centers.shape[1], p.k, data.getNumPartitions(),
        )

        history: list[IterationStats] = []
        converged = False
        cost = float("nan")
        bc: Broadcast[CentersPayload] | None = None

        for iteration in range(1, p.max_iter + 1):
            t0 = time.perf_counter()

            # (2) Broadcast current centroids + their squared norms.
            sq_norms = np.einsum("ij,ij->i", centers, centers)
            bc = sc.broadcast((centers, sq_norms))
            cost_acc = sc.accumulator(0.0)

            # (3) MAP + (4) REDUCE  ->  {k: (sum_vector[D], count)}
            zero: ClusterAcc = (np.zeros(centers.shape[1], dtype=np.float64), 0)
            cluster_stats: dict[int, ClusterAcc] = (
                vectors.map(
                    partial(
                        _map_to_closest,
                        bc_centers=bc,
                        distance_measure=p.distance_measure,
                        cost_acc=cost_acc,
                    )
                )
                .aggregateByKey(zero, _seq_op, _comb_op, numPartitions=n_partitions)  # SHUFFLE
                .collectAsMap()  # action -> K small records back to the driver
            )

            # (5) UPDATE on the driver.
            new_centers, max_shift = self._update_centers(centers, cluster_stats)
            bc.destroy()  # previous centroids are no longer referenced by any RDD
            bc = None

            cost = float(cost_acc.value)
            stats = IterationStats(
                iteration=iteration,
                max_center_shift=max_shift,
                cost=cost,
                empty_clusters=p.k - len(cluster_stats),
                duration_seconds=time.perf_counter() - t0,
            )
            history.append(stats)
            logger.info(
                "iter=%02d shift=%.3e cost=%.6f empty=%d (%.2fs)",
                iteration, max_shift, cost, stats.empty_clusters, stats.duration_seconds,
            )
            if self.iteration_callback is not None:
                self.iteration_callback(stats)

            centers = new_centers
            # (6) Convergence: every centroid moved by <= tol (MLlib semantics).
            if max_shift <= p.tol:
                converged = True
                break

        return RDDKMeansModel(
            centers=centers,
            distance_measure=p.distance_measure,
            num_iterations=len(history),
            converged=converged,
            training_cost=cost,
            history=history,
        )

    def fit_predict(
        self, spark: SparkSession, data: VectorRDD
    ) -> tuple[RDDKMeansModel, AssignmentRDD]:
        """(7) Fit and return ``(model, RDD[(doc_id, cluster_index)])`` (lazy RDD)."""
        model = self.fit(spark, data)
        return model, model.assign(spark, data)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _init_centers(self, data: VectorRDD) -> FloatArray:
        """(1) Pick ``K`` random documents as initial centroids (``takeSample``).

        Same strategy as MLlib's ``initMode="random"``. ``takeSample`` triggers
        a ``count`` + sampling job; only ``K`` vectors are collected, then
        densified to ``float64[K, D]``.
        """
        p = self.params
        sample = data.takeSample(withReplacement=False, num=p.k, seed=p.seed)
        dim = int(sample[0][1].size)
        centers = np.vstack([vo.to_dense(vec, dim) for _, vec in sample])
        if p.distance_measure == "cosine":
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
            centers = centers / np.where(norms == 0.0, 1.0, norms)
        n_unique = np.unique(centers, axis=0).shape[0]
        if n_unique < p.k:
            logger.warning("Only %d distinct initial centroids out of k=%d", n_unique, p.k)
        return np.ascontiguousarray(centers)

    def _update_centers(
        self, centers: FloatArray, cluster_stats: dict[int, ClusterAcc]
    ) -> tuple[FloatArray, float]:
        """(5) ``c_k = sum_k / n_k``; empty clusters keep their previous centroid
        (identical to MLlib). Returns ``(new_centers, max_shift)``."""
        dm = self.params.distance_measure
        new_centers = centers.copy()
        max_shift = 0.0
        for k, (vec_sum, count) in cluster_stats.items():
            new_centers[k] = vo.centroid(vec_sum, count, dm)
            max_shift = max(max_shift, vo.center_shift(centers[k], new_centers[k], dm))
        return new_centers, max_shift
