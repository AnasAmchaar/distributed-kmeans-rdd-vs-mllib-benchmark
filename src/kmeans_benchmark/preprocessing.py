"""Phase 1b — Distributed text preprocessing and TF-IDF featurisation.

Pipeline (all stages are Spark ML ``Transformer``/``Estimator`` running on the JVM)::

    text ──RegexTokenizer──▶ tokens ──StopWordsRemover──▶ filtered
         ──CountVectorizer──▶ tf ──IDF──▶ tfidf ──Normalizer(L2)──▶ features

The resulting ``features`` column is a Spark ML ``VectorUDT`` (``SparseVector``)
— consumed natively by MLlib — and :func:`to_vector_rdd` exposes the very same
vectors as ``RDD[(doc_id, SparseVector)]`` whose ``.indices`` / ``.values`` are
raw NumPy arrays, i.e. exactly what the from-scratch RDD K-Means needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import IDF, CountVectorizer, CountVectorizerModel, Normalizer
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.ml.linalg import Vector
from pyspark.rdd import RDD
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from kmeans_benchmark.config import FeatureConfig

logger = logging.getLogger(__name__)

FEATURES_COL = "features"

VectorRDD = RDD[tuple[int, Vector]]
"""``RDD[(doc_id, vector)]`` — input type of :class:`CustomRDDKMeans`."""


@dataclass(slots=True)
class TfidfResult:
    """Output of :func:`build_tfidf`.

    Attributes:
        features_df: **Cached** DataFrame ``(doc_id: long, label: string,
            features: VectorUDT[D])`` — shape ``N x D`` sparse.
        pipeline_model: Fitted preprocessing ``PipelineModel`` (re-usable for
            scoring new raw text and for packaging with the MLlib model).
        vocabulary: Term for each feature index (``len == D``).
        num_docs: ``N`` — number of rows in ``features_df``.
    """

    features_df: DataFrame
    pipeline_model: PipelineModel
    vocabulary: list[str]
    num_docs: int

    @property
    def num_features(self) -> int:
        """``D`` — dimensionality of the feature space."""
        return len(self.vocabulary)


def build_preprocessing_pipeline(cfg: FeatureConfig) -> Pipeline:
    """Build (unfitted) tokenisation -> stop-words -> TF -> IDF -> L2 pipeline.

    Design notes:
        * ``CountVectorizer`` (rather than ``HashingTF``) gives a collision-free,
          *interpretable* vocabulary, enabling "top terms per cluster" reports.
          It costs one extra aggregation job at ``fit`` time.
        * ``Normalizer(p=2)`` makes every row a unit vector, so Euclidean
          K-Means behaves like spherical (cosine) K-Means, the standard choice
          for document clustering; it also bounds distances for stable ``tol``.
    """
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern=r"[^a-z0-9]+",  # split on anything that is not alnum (after lower-casing)
        gaps=True,
        toLowercase=True,
        minTokenLength=cfg.min_token_length,
    )
    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english"),
    )
    tf = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=cfg.vocab_size,
        minDF=cfg.min_df,
    )
    idf = IDF(inputCol="tf", outputCol="tfidf" if cfg.l2_normalize else FEATURES_COL)
    stages = [tokenizer, remover, tf, idf]
    if cfg.l2_normalize:
        stages.append(Normalizer(inputCol="tfidf", outputCol=FEATURES_COL, p=2.0))
    return Pipeline(stages=stages)


@F.udf(returnType=BooleanType())
def _has_nonzeros(v: Vector | None) -> bool:
    """``True`` when the vector has at least one non-zero entry."""
    return v is not None and int(v.numNonzeros()) > 0


def build_tfidf(
    corpus_df: DataFrame,
    cfg: FeatureConfig,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
) -> TfidfResult:
    """Fit the TF-IDF pipeline and return a **cached, materialised** feature matrix.

    Args:
        corpus_df: ``(doc_id: long, text: string, label: string|null)``.
        cfg: Feature configuration.
        storage_level: Persistence level of the feature matrix.

    Returns:
        :class:`TfidfResult` whose ``features_df`` has shape ``N x D``.

    DAG implications:
        * ``CountVectorizer.fit`` -> one job with a **shuffle** (term counting
          via ``reduceByKey``) + ``top(vocabSize)``.
        * ``IDF.fit`` -> one ``treeAggregate`` job (document frequencies).
        * The final ``count()`` materialises the cache so neither K-Means
          implementation pays for tokenisation / TF-IDF inside its timed loop.
          Without this ``persist`` every Lloyd iteration would re-read and
          re-featurise the whole corpus.
    """
    pipeline_model = build_preprocessing_pipeline(cfg).fit(corpus_df)
    features = pipeline_model.transform(corpus_df).select("doc_id", "label", FEATURES_COL)

    if cfg.drop_empty_vectors:
        features = features.where(_has_nonzeros(F.col(FEATURES_COL)))
    if cfg.num_partitions:
        features = features.repartition(cfg.num_partitions)  # explicit shuffle, once

    features = features.persist(storage_level)
    num_docs = features.count()  # action -> materialise cache

    cv_model = next(s for s in pipeline_model.stages if isinstance(s, CountVectorizerModel))
    vocabulary = list(cv_model.vocabulary)
    logger.info("TF-IDF matrix: N=%d docs x D=%d features (cached)", num_docs, len(vocabulary))
    return TfidfResult(features, pipeline_model, vocabulary, num_docs)


def to_vector_rdd(features_df: DataFrame, features_col: str = FEATURES_COL) -> VectorRDD:
    """Expose the TF-IDF matrix as ``RDD[(doc_id, Vector)]`` for the RDD K-Means.

    Each element is ``(doc_id: int, SparseVector)`` where ``vector.indices``
    (``int32[nnz]``) and ``vector.values`` (``float64[nnz]``) are NumPy arrays,
    so the map phase can do vectorised NumPy math without densifying.

    DAG implications:
        ``DataFrame.rdd`` adds a JVM -> Python deserialisation step (rows are
        pickled to the Python workers). The returned RDD is *not* cached here;
        :class:`CustomRDDKMeans` persists it before iterating.
    """
    return features_df.select("doc_id", features_col).rdd.map(lambda row: (int(row[0]), row[1]))

