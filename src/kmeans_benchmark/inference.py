"""Interactive document clustering and inference using both RDD and MLlib K-Means.

Allows fitting on a corpus (or reusing a completed benchmark run) and clustering
arbitrary raw text documents with **both** approaches side-by-side, returning the
assigned cluster ID and the human-readable top topic keywords for each approach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pyspark.ml.clustering import KMeansModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from kmeans_benchmark.benchmark import BenchmarkReport
from kmeans_benchmark.config import BenchmarkConfig, FeatureConfig, KMeansParams
from kmeans_benchmark.evaluation import top_terms_per_cluster
from kmeans_benchmark.ingestion import load_corpus
from kmeans_benchmark.mllib_kmeans import build_mllib_kmeans
from kmeans_benchmark.preprocessing import FEATURES_COL, TfidfResult, build_tfidf, to_vector_rdd
from kmeans_benchmark.rdd_kmeans import CustomRDDKMeans, RDDKMeansModel


@dataclass(frozen=True, slots=True)
class DocumentClusterAssignment:
    """Cluster assignment for a single document from both K-Means engines."""

    doc_id: int
    text: str
    rdd_cluster: int
    rdd_cluster_terms: list[str]
    mllib_cluster: int
    mllib_cluster_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "rdd_cluster": self.rdd_cluster,
            "rdd_cluster_terms": ", ".join(self.rdd_cluster_terms),
            "mllib_cluster": self.mllib_cluster,
            "mllib_cluster_terms": ", ".join(self.mllib_cluster_terms),
        }


@dataclass(slots=True)
class DualDocumentClusterer:
    """End-to-end text clusterer backed by both ``CustomRDDKMeans`` and MLlib ``KMeansModel``."""

    tfidf: TfidfResult
    rdd_model: RDDKMeansModel
    mllib_model: KMeansModel
    rdd_top_terms: dict[int, list[str]]
    mllib_top_terms: dict[int, list[str]]

    @classmethod
    def fit(
        cls,
        spark: SparkSession,
        corpus_df: DataFrame | None = None,
        texts: Sequence[str] | None = None,
        cfg: BenchmarkConfig | None = None,
        top_n_terms: int = 6,
    ) -> DualDocumentClusterer:
        """Fit the TF-IDF pipeline and both K-Means models (RDD + MLlib) on a corpus.

        Args:
            spark: Active SparkSession.
            corpus_df: Optional DataFrame with ``(doc_id, text)`` or ``(text,)``.
            texts: Optional Python sequence of raw document strings to train on.
            cfg: Configuration (defaults to a fast interactive setup if omitted).
            top_n_terms: Number of representative keywords stored per cluster.
        """
        if cfg is None:
            cfg = BenchmarkConfig(
                features=FeatureConfig(vocab_size=4096, min_df=2.0),
                kmeans=KMeansParams(k=8, max_iter=15, tol=1e-4, seed=42),
            )

        if texts is not None:
            rows = [(i, str(t), None) for i, t in enumerate(texts)]
            corpus_df = spark.createDataFrame(rows, ["doc_id", "text", "label"])
        elif corpus_df is None:
            corpus_df = load_corpus(spark, cfg.data)
        else:
            if "doc_id" not in corpus_df.columns:
                corpus_df = corpus_df.withColumn("doc_id", F.monotonically_increasing_id())
            if "label" not in corpus_df.columns:
                corpus_df = corpus_df.withColumn("label", F.lit(None).cast("string"))

        tfidf = build_tfidf(corpus_df, cfg.features)
        vectors_rdd = to_vector_rdd(tfidf.features_df)
        rdd_model = CustomRDDKMeans(cfg.kmeans).fit(spark, vectors_rdd)
        mllib_model = build_mllib_kmeans(cfg.kmeans).fit(tfidf.features_df)

        return cls(
            tfidf=tfidf,
            rdd_model=rdd_model,
            mllib_model=mllib_model,
            rdd_top_terms=top_terms_per_cluster(rdd_model.centers, tfidf.vocabulary, top_n_terms),
            mllib_top_terms=top_terms_per_cluster(
                mllib_model.clusterCenters(), tfidf.vocabulary, top_n_terms
            ),
        )

    @classmethod
    def from_benchmark(
        cls, tfidf: TfidfResult, report: BenchmarkReport, top_n_terms: int = 6
    ) -> DualDocumentClusterer:
        """Build a clusterer directly from an already-executed :func:`run_benchmark`."""
        assert isinstance(report.rdd.model, RDDKMeansModel)
        assert isinstance(report.mllib.model, KMeansModel)
        return cls(
            tfidf=tfidf,
            rdd_model=report.rdd.model,
            mllib_model=report.mllib.model,
            rdd_top_terms=top_terms_per_cluster(
                report.rdd.centers, tfidf.vocabulary, top_n_terms
            ),
            mllib_top_terms=top_terms_per_cluster(
                report.mllib.centers, tfidf.vocabulary, top_n_terms
            ),
        )

    def cluster_texts(
        self, spark: SparkSession, texts: Sequence[str]
    ) -> list[DocumentClusterAssignment]:
        """Cluster one or more raw text strings with both RDD and MLlib models.

        Args:
            spark: Active SparkSession.
            texts: Sequence of raw text documents to assign to clusters.

        Returns:
            List of :class:`DocumentClusterAssignment` preserving input order.
        """
        input_rows = [(i, str(t)) for i, t in enumerate(texts)]
        raw_df = spark.createDataFrame(input_rows, ["doc_id", "text"])
        featurized = (
            self.tfidf.pipeline_model.transform(raw_df)
            .select("doc_id", "text", FEATURES_COL)
            .orderBy("doc_id")
            .collect()
        )

        results: list[DocumentClusterAssignment] = []
        for row in featurized:
            doc_id = int(row["doc_id"])
            text = str(row["text"])
            vec = row[FEATURES_COL]
            rdd_c = int(self.rdd_model.predict(vec))
            mllib_c = int(self.mllib_model.predict(vec))
            results.append(
                DocumentClusterAssignment(
                    doc_id=doc_id,
                    text=text,
                    rdd_cluster=rdd_c,
                    rdd_cluster_terms=self.rdd_top_terms.get(rdd_c, []),
                    mllib_cluster=mllib_c,
                    mllib_cluster_terms=self.mllib_top_terms.get(mllib_c, []),
                )
            )
        return results

    def cluster_dataframe(self, spark: SparkSession, docs_df: DataFrame) -> DataFrame:
        """Cluster a Spark DataFrame containing a ``text`` column using both models.

        Returns a DataFrame with ``doc_id``, ``text``, ``rdd_cluster``,
        ``rdd_cluster_terms``, ``mllib_cluster``, and ``mllib_cluster_terms``.
        """
        if "doc_id" not in docs_df.columns:
            docs_df = docs_df.withColumn("doc_id", F.monotonically_increasing_id())
        featurized = self.tfidf.pipeline_model.transform(docs_df).select(
            "doc_id", "text", FEATURES_COL
        )
        with_rdd = self.rdd_model.transform(
            spark, featurized, features_col=FEATURES_COL, prediction_col="rdd_cluster"
        )
        with_both = (
            self.mllib_model.setPredictionCol("mllib_cluster")
            .transform(with_rdd)
            .drop(FEATURES_COL)
        )

        rdd_map_expr = F.create_map(
            *[
                x
                for k, terms in self.rdd_top_terms.items()
                for x in (F.lit(k), F.lit(", ".join(terms)))
            ]
        )
        mllib_map_expr = F.create_map(
            *[
                x
                for k, terms in self.mllib_top_terms.items()
                for x in (F.lit(k), F.lit(", ".join(terms)))
            ]
        )
        return with_both.withColumn(
            "rdd_cluster_terms", rdd_map_expr[F.col("rdd_cluster")]
        ).withColumn("mllib_cluster_terms", mllib_map_expr[F.col("mllib_cluster")])
