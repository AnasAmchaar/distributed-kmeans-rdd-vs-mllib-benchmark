"""Centralised, immutable configuration for the K-Means benchmark.

Every hyper-parameter that influences *both* implementations lives in a single
:class:`KMeansParams` object so that the RDD and MLlib runs are guaranteed to be
configured identically (fair benchmark).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DistanceMeasure = Literal["euclidean", "manhattan", "cosine"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the corpus lives and how to (optionally) synthesise it.

    Attributes:
        input_path: DBFS / UC-Volume / local path of the text corpus. Examples:
            ``dbfs:/FileStore/kmeans_benchmark/corpus`` or
            ``/Volumes/main/default/raw/corpus``.
        one_doc_per_file: ``True`` -> ``spark.read.text(wholetext=True)`` (each
            file is a document, e.g. 20-Newsgroups layout). ``False`` -> one
            document per line (fastText-style ``__label__x <text>`` allowed).
        generate_if_missing: Generate a synthetic corpus when ``input_path``
            does not exist.
        materialize_synthetic: Write the generated corpus to ``input_path`` and
            read it back with ``spark.read.text`` (``True``, recommended on
            Databricks) or use it in memory (``False``, e.g. local Windows runs
            without ``winutils``).
        synthetic_num_docs: Number of synthetic documents to generate.
        synthetic_num_partitions: Parallelism of the synthetic generator (and
            the number of output files).
        synthetic_words_per_doc: (min, max) document length in tokens.
        synthetic_topic_purity: Probability that a content word is drawn from
            the document's own topic vocabulary (vs. a random other topic).
        seed: Seed for the synthetic generator.
    """

    input_path: str = "dbfs:/FileStore/kmeans_benchmark/synthetic_corpus"
    one_doc_per_file: bool = False
    generate_if_missing: bool = True
    materialize_synthetic: bool = True
    synthetic_num_docs: int = 200_000
    synthetic_num_partitions: int = 64
    synthetic_words_per_doc: tuple[int, int] = (40, 160)
    synthetic_topic_purity: float = 0.80
    seed: int = 42


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """TF-IDF featurisation settings.

    Attributes:
        vocab_size: Max vocabulary size of ``CountVectorizer`` == dimensionality
            ``D`` of every feature vector. Drives centroid size (``K x D``
            dense float64) and therefore broadcast / shuffle volume.
        min_df: Minimum number (or fraction if < 1.0) of documents a term must
            appear in.
        min_token_length: Tokens shorter than this are dropped by the tokenizer.
        l2_normalize: L2-normalise TF-IDF vectors. With unit vectors, squared
            Euclidean distance == ``2 * (1 - cosine)``, i.e. "spherical" K-Means.
        drop_empty_vectors: Remove documents whose TF-IDF vector has no
            non-zero entries (mandatory for cosine distance).
        num_partitions: Repartition the feature matrix to this many partitions
            before caching (``None`` = keep upstream partitioning).
    """

    vocab_size: int = 1 << 14
    min_df: float = 5.0
    min_token_length: int = 3
    l2_normalize: bool = True
    drop_empty_vectors: bool = True
    num_partitions: int | None = None


@dataclass(frozen=True, slots=True)
class KMeansParams:
    """Hyper-parameters shared *verbatim* by both K-Means implementations.

    Attributes:
        k: Number of clusters ``K``.
        max_iter: Maximum number of Lloyd iterations.
        tol: Convergence tolerance. Euclidean: every centroid must move less
            than ``tol`` (L2). Cosine: cosine distance between old/new centroid
            must be <= ``tol``. Mirrors MLlib's ``isCenterConverged`` exactly.
        seed: Random seed (centroid initialisation).
        distance_measure: ``"euclidean"`` or ``"cosine"``.
    """

    k: int = 8
    max_iter: int = 20
    tol: float = 1e-4
    seed: int = 42
    distance_measure: DistanceMeasure = "euclidean"

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError(f"k must be >= 2, got {self.k}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.tol < 0:
            raise ValueError(f"tol must be >= 0, got {self.tol}")
        if self.distance_measure not in ("euclidean", "manhattan", "cosine"):
            raise ValueError(f"Unsupported distance_measure={self.distance_measure!r}")

    @property
    def mllib_distance(self) -> str:
        """MLlib KMeans supports ``euclidean`` and ``cosine`` natively."""
        return "cosine" if self.distance_measure == "cosine" else "euclidean"

    @property
    def evaluator_distance(self) -> str:
        """Matching ``ClusteringEvaluator.distanceMeasure`` value."""
        return "cosine" if self.distance_measure == "cosine" else "squaredEuclidean"


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """MLflow settings.

    Attributes:
        experiment_name: Workspace path on Databricks
            (e.g. ``/Users/me@corp.com/kmeans_benchmark``) or a plain name locally.
        tracking_uri: ``None`` -> environment default (``databricks`` on a
            cluster, ``./mlruns`` locally).
        log_full_pipeline: Log ``PipelineModel(preprocessing + KMeans)`` so the
            served model accepts raw text; otherwise log only ``KMeansModel``.
        top_terms_per_cluster: Number of top centroid terms logged per cluster.
    """

    experiment_name: str = "kmeans_rdd_vs_mllib_benchmark"
    tracking_uri: str | None = None
    log_full_pipeline: bool = True
    top_terms_per_cluster: int = 15


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Root configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    kmeans: KMeansParams = field(default_factory=KMeansParams)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten to ``{"section.key": value}`` (used for MLflow params)."""
        flat: dict[str, Any] = {}
        for section, values in asdict(self).items():
            for key, value in values.items():
                flat[f"{section}.{key}"] = value
        return flat
