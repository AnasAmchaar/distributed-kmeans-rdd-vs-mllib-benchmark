"""Phase 3 — The equivalent pipeline with the high-level PySpark MLlib DataFrame API."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.sql import DataFrame

from kmeans_benchmark.config import KMeansParams
from kmeans_benchmark.preprocessing import FEATURES_COL

logger = logging.getLogger(__name__)

PREDICTION_COL = "prediction"


@dataclass(slots=True)
class MLlibKMeansResult:
    """Output of :func:`fit_mllib_kmeans`.

    Attributes:
        model: Fitted ``KMeansModel`` (``clusterCenters()`` -> ``K`` arrays of ``D``).
        predictions: **Cached** ``features_df`` + ``prediction: int`` column.
        num_iterations: Iterations actually run (``model.summary.numIter``).
        training_cost: ``model.summary.trainingCost``.
    """

    model: KMeansModel
    predictions: DataFrame
    num_iterations: int
    training_cost: float


def build_mllib_kmeans(params: KMeansParams, features_col: str = FEATURES_COL) -> KMeans:
    """Instantiate ``pyspark.ml.clustering.KMeans`` with the shared hyper-parameters.

    ``initMode="random"`` is forced (MLlib's default is ``"k-means||"``) so both
    implementations use the *same* initialisation strategy — K random
    documents. Note that the two engines use different RNGs, so the exact
    seed documents differ; compare results over several seeds for rigour.
    """
    return KMeans(
        featuresCol=features_col,
        predictionCol=PREDICTION_COL,
        k=params.k,
        maxIter=params.max_iter,
        tol=params.tol,
        seed=params.seed,
        initMode="random",
        distanceMeasure=params.distance_measure,
    )


def fit_mllib_kmeans(
    features_df: DataFrame,
    params: KMeansParams,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
) -> MLlibKMeansResult:
    """Fit MLlib K-Means and assign a cluster id to every document.

    Args:
        features_df: Cached ``(doc_id, label, features: VectorUDT[D])`` — ``N x D``.
        params: Hyper-parameters shared with the RDD run.
        storage_level: Persistence level for the prediction DataFrame.

    Returns:
        :class:`MLlibKMeansResult`; ``predictions`` has shape ``N x (cols + 1)``.

    DAG implications:
        * ``fit`` converts the DataFrame to an internal JVM ``RDD[VectorWithNorm]``
          (persisted by MLlib if the input is not cached) and runs the same
          broadcast -> map-side-combine -> ``reduceByKey`` loop as our RDD code,
          but entirely in the JVM with BLAS kernels and no Python serialisation.
        * ``transform`` is a narrow projection (UDF), **no shuffle**. It is
          persisted and materialised here so prediction cost is included in the
          timed section and the evaluator re-uses the cached result.
    """
    model: KMeansModel = build_mllib_kmeans(params).fit(features_df)
    predictions = model.transform(features_df).persist(storage_level)
    predictions.count()  # cached DF => full plan (incl. prediction UDF) is computed

    summary = model.summary
    logger.info(
        "MLlib K-Means: iterations=%d cost=%.6f sizes=%s",
        summary.numIter, summary.trainingCost, summary.clusterSizes,
    )
    return MLlibKMeansResult(
        model=model,
        predictions=predictions,
        num_iterations=int(summary.numIter),
        training_cost=float(summary.trainingCost),
    )

