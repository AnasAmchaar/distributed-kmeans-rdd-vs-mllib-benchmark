"""Mini-Projet 10 — Livrable 1 : ``custom_kmeans_rdd.py``

Implémentation manuelle ("from scratch") de l'algorithme K-Means à partir des
primitives distribuées Spark (RDDs) sur des vecteurs de documents (58D ou TF-IDF) :
    * Distances supportées : **Euclidienne (L2)** et **Manhattan (L1)**
    * Broadcast des centroïdes (``sc.broadcast``) vers les exécuteurs
    * Étape **Map** : assignation de chaque document au centroïde le plus proche
      ``x -> (cluster_id, (x_dense, 1, sq_error))``
    * Étape **Reduce** : ``reduceByKey`` pour sommer les vecteurs, les comptages
      et calculer la moyenne (nouveaux centroïdes) ainsi que le coût
      intra-cluster **WSSE** (Within Set Sum of Squared Errors) à chaque itération.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pyspark import Broadcast, StorageLevel
from pyspark.ml.linalg import DenseVector, Vector, Vectors
from pyspark.rdd import RDD
from pyspark.sql import DataFrame, SparkSession

from kmeans_benchmark.spark_session import get_spark, get_spark_context, has_rdd_support

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
DistanceL1L2 = Literal["euclidean", "manhattan", "l2", "l1"]

# Value reduced per key by reduceByKey: (sum_vector[D], count, wsse_sum)
ReducePayload = tuple[FloatArray, int, float]


def normalize_distance_name(distance: str) -> Literal["euclidean", "manhattan"]:
    """Normalise ``'l1'`` / ``'manhattan'`` and ``'l2'`` / ``'euclidean'``."""
    d = distance.strip().lower()
    if d in ("l1", "manhattan"):
        return "manhattan"
    if d in ("l2", "euclidean"):
        return "euclidean"
    raise ValueError(f"Unsupported distance={distance!r}. Use 'euclidean' (L2) or 'manhattan' (L1).")


def _to_numpy(vec: Vector | FloatArray | list[float]) -> FloatArray:
    """Convert a Spark ``Vector`` or sequence to a 1-D ``float64`` NumPy array."""
    if isinstance(vec, np.ndarray):
        return vec.astype(np.float64, copy=False)
    if hasattr(vec, "toArray"):
        return np.asarray(vec.toArray(), dtype=np.float64)
    return np.asarray(vec, dtype=np.float64)


# =========================================================================== #
# Primitives Map & Reduce (module-level functions for Spark serialization)
# =========================================================================== #
def _map_assign_centroid(
    x: FloatArray,
    bc_centers: Broadcast[FloatArray],
    distance_mode: Literal["euclidean", "manhattan"],
) -> tuple[int, ReducePayload]:
    """ÉTAPE MAP : assigne un vecteur document ``x`` (58D) au centroïde le plus proche.

    * Si ``distance_mode == "euclidean"`` (L2) : minimise ``||x - c_k||_2^2``.
    * Si ``distance_mode == "manhattan"`` (L1) : minimise ``||x - c_k||_1``.
    * Dans les deux cas, retourne également l'erreur quadratique ``||x - c_best||_2^2``
      pour le calcul exact du coût intra-cluster **WSSE** à chaque itération.
    """
    centers = bc_centers.value  # shape: (K, D)
    diff = centers - x  # shape: (K, D)
    if distance_mode == "manhattan":
        dists = np.sum(np.abs(diff), axis=1)
    else:
        dists = np.einsum("ij,ij->i", diff, diff)

    best_k = int(np.argmin(dists))
    best_diff = diff[best_k]
    sq_err = float(np.dot(best_diff, best_diff))
    return best_k, (x, 1, sq_err)


def _reduce_cluster_sums(a: ReducePayload, b: ReducePayload) -> ReducePayload:
    """ÉTAPE REDUCE (``reduceByKey``) : agrège les sommes vectorielles, les effectifs
    et le coût intra-cluster WSSE pour chaque cluster ``k``."""
    sum_a, cnt_a, wsse_a = a
    sum_b, cnt_b, wsse_b = b
    return sum_a + sum_b, cnt_a + cnt_b, wsse_a + wsse_b


# =========================================================================== #
# Structures de résultats
# =========================================================================== #
@dataclass(frozen=True, slots=True)
class RDDIterationRecord:
    """Télémetrie d'une itération du K-Means RDD (WSSE, déplacement, temps)."""

    iteration: int
    wsse: float
    max_shift: float
    duration_seconds: float
    empty_clusters: int = 0


@dataclass(slots=True)
class CustomKMeansRDDResult:
    """Résultat complet de l'entraînement ``CustomKMeansRDD``."""

    centers: FloatArray
    distance: Literal["euclidean", "manhattan"]
    k: int
    num_iterations: int
    converged: bool
    final_wsse: float
    total_time_seconds: float
    history: list[RDDIterationRecord] = field(default_factory=list)

    @property
    def wsse_curve(self) -> list[float]:
        return [h.wsse for h in self.history]

    @property
    def iteration_times(self) -> list[float]:
        return [h.duration_seconds for h in self.history]

    def predict_one(self, vec: Vector | FloatArray | list[float]) -> int:
        x = _to_numpy(vec)
        diff = self.centers - x
        if self.distance == "manhattan":
            return int(np.argmin(np.sum(np.abs(diff), axis=1)))
        return int(np.argmin(np.einsum("ij,ij->i", diff, diff)))


# =========================================================================== #
# Estimateur K-Means Manuel sur RDDs (L1 & L2 + reduceByKey + WSSE)
# =========================================================================== #
class CustomKMeansRDD:
    """Implémentation K-Means 'from scratch' utilisant strictement les RDDs Spark.

    Respecte intégralement le cahier des charges du **Mini-Projet 10** :
        1. Distances **Manhattan (L1)** et **Euclidienne (L2)**.
        2. **Étape Map** : assignation au centroïde le plus proche via variable broadcastée.
        3. **Étape Reduce** : ``reduceByKey`` pour calculer les nouvelles moyennes.
        4. **Calcul du coût intra-cluster WSSE** à chaque itération.
    """

    def __init__(
        self,
        k: int = 5,
        max_iter: int = 15,
        tol: float = 1e-4,
        distance: str = "euclidean",
        seed: int = 42,
        iteration_callback: Callable[[RDDIterationRecord], None] | None = None,
    ) -> None:
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.distance = normalize_distance_name(distance)
        self.seed = seed
        self.iteration_callback = iteration_callback

    def fit(
        self,
        spark: SparkSession,
        data: RDD[FloatArray] | DataFrame,
        features_col: str = "features",
    ) -> CustomKMeansRDDResult:
        """Exécute la boucle MapReduce K-Means sur un RDD de vecteurs (ex: 58D)."""
        sc = get_spark_context(spark)
        rdd = self._ensure_numpy_rdd(spark, data, features_col)
        rdd.persist(StorageLevel.MEMORY_AND_DISK)
        n_docs = rdd.count()
        if n_docs < self.k:
            raise ValueError(f"Corpus size ({n_docs}) must be >= k ({self.k})")

        # Initialisation : échantillonnage déterministe de K documents
        init_samples = rdd.takeSample(withReplacement=False, num=self.k, seed=self.seed)
        centers = np.ascontiguousarray(np.vstack(init_samples), dtype=np.float64)

        history: list[RDDIterationRecord] = []
        converged = False
        final_wsse = float("nan")
        t_total_start = time.perf_counter()

        for iteration in range(1, self.max_iter + 1):
            t_iter = time.perf_counter()
            bc_centers = sc.broadcast(centers)

            # MAP -> REDUCE (reduceByKey) -> COLLECT
            n_red = max(1, min(self.k, 2))
            reduced: dict[int, ReducePayload] = (
                rdd.map(
                    partial(
                        _map_assign_centroid,
                        bc_centers=bc_centers,
                        distance_mode=self.distance,
                    )
                )
                .reduceByKey(_reduce_cluster_sums, numPartitions=n_red)
                .collectAsMap()
            )

            bc_centers.destroy()

            new_centers = centers.copy()
            iter_wsse = 0.0
            max_shift = 0.0
            for cluster_id, (vec_sum, count, cluster_wsse) in reduced.items():
                new_c = vec_sum / float(count)
                new_centers[cluster_id] = new_c
                iter_wsse += float(cluster_wsse)
                if self.distance == "manhattan":
                    shift = float(np.sum(np.abs(centers[cluster_id] - new_c)))
                else:
                    shift = float(np.linalg.norm(centers[cluster_id] - new_c))
                max_shift = max(max_shift, shift)

            iter_duration = time.perf_counter() - t_iter
            final_wsse = iter_wsse
            rec = RDDIterationRecord(
                iteration=iteration,
                wsse=iter_wsse,
                max_shift=max_shift,
                duration_seconds=iter_duration,
                empty_clusters=self.k - len(reduced),
            )
            history.append(rec)
            logger.info(
                "[RDD-%s] iter=%02d/%02d WSSE=%.4f max_shift=%.3e time=%.3fs",
                self.distance.upper(),
                iteration,
                self.max_iter,
                iter_wsse,
                max_shift,
                iter_duration,
            )
            if self.iteration_callback is not None:
                self.iteration_callback(rec)

            centers = new_centers
            if max_shift <= self.tol:
                converged = True
                break

        total_time = time.perf_counter() - t_total_start
        rdd.unpersist()
        return CustomKMeansRDDResult(
            centers=centers,
            distance=self.distance,
            k=self.k,
            num_iterations=len(history),
            converged=converged,
            final_wsse=final_wsse,
            total_time_seconds=total_time,
            history=history,
        )

    @staticmethod
    def _ensure_numpy_rdd(
        spark: SparkSession, data: RDD[FloatArray] | DataFrame, features_col: str
    ) -> RDD[FloatArray]:
        if isinstance(data, DataFrame):
            if has_rdd_support(spark):
                return data.select(features_col).rdd.map(lambda r: _to_numpy(r[0]))
            from kmeans_benchmark.preprocessing import PartitionedRDDShim

            rows = [_to_numpy(r[0]) for r in data.select(features_col).collect()]
            parts: list[list[FloatArray]] = [[] for _ in range(4)]
            for idx, arr in enumerate(rows):
                parts[idx % 4].append(arr)
            # Add reduceByKey to PartitionedRDDShim if needed
            shim = PartitionedRDDShim(parts)
            if not hasattr(shim, "reduceByKey"):
                shim.reduceByKey = lambda func: shim.aggregateByKey(  # type: ignore[attr-defined]
                    None,
                    lambda acc, v: v if acc is None else func(acc, v),
                    lambda a, b: func(a, b),
                )
            return shim  # type: ignore[return-value]
        return data.map(_to_numpy)


# =========================================================================== #
# Générateur de Vecteurs de Documents 58D (Section 3 du Descriptif)
# =========================================================================== #
def generate_58d_document_vectors(
    num_docs: int = 1_500,
    k_true: int = 5,
    seed: int = 42,
) -> tuple[FloatArray, NDArray[np.int32]]:
    """Génère la représentation vectorielle 58D des documents (Section 3 du PDF).

    Format exact conforme à l'exemple du descriptif (style UCI Spambase 58D) :
        ``[0.0, 0.64, 0.64, 0.0, 0.32, 0.0, 0.0, 0.0, 0.96, ..., 3.78, 1.32, 45.0]``
        * Colonnes 0..47  (48D) : fréquences de mots clés techniques (%) dans [0, 5.0],
          creuses (~70% de zéros, multiples typiques de 0.16 / 0.32 / 0.64 / 0.96).
        * Colonnes 48..53 (6D)  : fréquences de caractères spéciaux (``;``, ``(``, ``[``, ``!``, ``$``, ``#``).
        * Colonnes 54..57 (4D)  : statistiques de longueur des séquences de majuscules / tokens
          (ex: ``..., 3.78, 1.32, 45.0``), normalisées à l'échelle des fréquences pour le clustering.
    """
    rng = np.random.default_rng(seed)
    dim = 58
    labels = rng.integers(0, k_true, size=num_docs, dtype=np.int32)
    matrix = np.zeros((num_docs, dim), dtype=np.float64)

    # Signatures thématiques bien séparées sur les 58 dimensions
    prototypes = np.zeros((k_true, dim), dtype=np.float64)
    for c in range(k_true):
        active_word_cols = rng.choice(np.arange(c * 9, min((c + 1) * 10, 48)), size=7, replace=False)
        prototypes[c, active_word_cols] = rng.choice([0.64, 0.96, 1.28, 1.60, 1.92], size=len(active_word_cols))
        prototypes[c, 48 + (c % 6)] = 0.48 + 0.16 * c
        prototypes[c, 54] = 1.5 + 0.8 * c
        prototypes[c, 55] = 1.1 + 0.5 * c
        prototypes[c, 56] = 2.5 + 1.2 * c
        prototypes[c, 57] = 3.5 + 1.5 * c

    for i in range(num_docs):
        c = int(labels[i])
        vec = prototypes[c].copy()
        noise_mask = rng.random(48) < 0.18
        vec[:48] += noise_mask * rng.choice([0.16, 0.32, 0.64], size=48)
        vec[:54] = np.clip(vec[:54] + rng.normal(0.0, 0.08, size=54), 0.0, None)
        vec[:54] = np.round(vec[:54], 2)
        vec[54:] = np.round(np.clip(vec[54:] + rng.normal(0.0, 0.25, size=4), 0.1, None), 2)
        matrix[i] = vec

    # Injecte en ligne 0 le profil exact cité dans la Section 3 du PDF
    sample_58d = np.zeros(dim, dtype=np.float64)
    sample_58d[[1, 2, 4, 8, 11, 18, 23, 55, 56, 57]] = [
        0.64, 0.64, 0.32, 0.96, 0.64, 0.48, 0.80, 3.78, 1.32, 4.50
    ]
    matrix[0] = sample_58d
    return matrix, labels


def build_58d_dataframe(
    spark: SparkSession, num_docs: int = 1_500, k_true: int = 5, seed: int = 42, num_partitions: int = 2
) -> DataFrame:
    """Crée un DataFrame Spark ``(doc_id, label, features: VectorUDT[58])``."""
    matrix, labels = generate_58d_document_vectors(num_docs=num_docs, k_true=k_true, seed=seed)
    rows = [
        (int(i), int(labels[i]), Vectors.dense(matrix[i].tolist()))
        for i in range(num_docs)
    ]
    return spark.createDataFrame(rows, ["doc_id", "label", "features"]).coalesce(num_partitions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    spark_session = get_spark()
    df_58d = build_58d_dataframe(spark_session, num_docs=1_000, k_true=5)
    for dist in ("euclidean", "manhattan"):
        res = CustomKMeansRDD(k=5, max_iter=12, tol=1e-4, distance=dist, seed=42).fit(
            spark_session, df_58d
        )
        print(
            f"[CustomKMeansRDD - {dist.upper()}] iterations={res.num_iterations} "
            f"converged={res.converged} final_WSSE={res.final_wsse:.2f} "
            f"time={res.total_time_seconds:.2f}s"
        )
