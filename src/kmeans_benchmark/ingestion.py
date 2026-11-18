"""Phase 1a — Data ingestion from DBFS (with a distributed synthetic fallback).

Output contract of this module (consumed by :mod:`preprocessing`)::

    root
     |-- doc_id: long      (unique, not necessarily contiguous)
     |-- text:   string    (raw document body)
     |-- label:  string    (nullable; ground-truth topic when known)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from kmeans_benchmark.config import DataConfig

logger = logging.getLogger(__name__)

# fastText-style label prefix: "__label__databases some text ..."
_LABEL_PREFIX_RE = r"^__label__(\S+)\s+"

# --------------------------------------------------------------------------- #
# Synthetic corpus vocabulary (technical documents, 8 latent topics)
# --------------------------------------------------------------------------- #
TOPIC_VOCABULARY: dict[str, tuple[str, ...]] = {
    "databases": (
        "sql", "index", "query", "transaction", "postgres", "mysql", "schema", "table",
        "join", "btree", "replication", "acid", "isolation", "lock", "deadlock", "optimizer",
        "partition", "shard", "normalization", "primary", "foreign", "key", "commit", "rollback",
        "wal", "vacuum", "materialized", "view", "cursor", "olap", "oltp", "columnar",
    ),
    "networking": (
        "tcp", "udp", "packet", "router", "switch", "latency", "bandwidth", "subnet", "dns",
        "bgp", "ospf", "firewall", "vlan", "ethernet", "socket", "handshake", "congestion",
        "throughput", "ipv6", "ipv4", "gateway", "nat", "mtu", "jitter", "topology", "mesh",
        "loadbalancer", "proxy", "http", "quic", "routing", "frame",
    ),
    "machine_learning": (
        "gradient", "neural", "network", "training", "inference", "tensor", "loss", "epoch",
        "regression", "classification", "overfitting", "regularization", "dropout", "backprop",
        "transformer", "attention", "embedding", "feature", "label", "dataset", "optimizer",
        "learning", "rate", "batch", "convolution", "recurrent", "accuracy", "precision",
        "recall", "kmeans", "cluster", "pca",
    ),
    "cloud": (
        "aws", "azure", "gcp", "kubernetes", "container", "docker", "serverless", "lambda",
        "autoscaling", "region", "zone", "bucket", "iam", "vpc", "terraform", "helm", "pod",
        "node", "cluster", "ingress", "service", "mesh", "istio", "elastic", "instance",
        "spot", "reserved", "billing", "tenant", "provisioning", "orchestration", "deployment",
    ),
    "security": (
        "encryption", "tls", "certificate", "vulnerability", "exploit", "malware", "phishing",
        "authentication", "authorization", "oauth", "token", "jwt", "hash", "salt", "cipher",
        "aes", "rsa", "firewall", "intrusion", "audit", "compliance", "zero", "trust", "cve",
        "patch", "sandbox", "privilege", "escalation", "ransomware", "siem", "pentest", "mfa",
    ),
    "frontend": (
        "react", "javascript", "typescript", "css", "html", "component", "render", "dom",
        "browser", "webpack", "vite", "state", "hook", "props", "redux", "angular", "vue",
        "responsive", "layout", "flexbox", "grid", "animation", "accessibility", "aria",
        "bundle", "hydration", "ssr", "spa", "router", "event", "listener", "tailwind",
    ),
    "devops": (
        "pipeline", "jenkins", "gitlab", "github", "actions", "build", "artifact", "release",
        "rollback", "canary", "bluegreen", "monitoring", "prometheus", "grafana", "alert",
        "incident", "oncall", "sre", "slo", "sli", "observability", "tracing", "logging",
        "ansible", "chef", "puppet", "configuration", "drift", "gitops", "argocd", "linting",
        "deployment",
    ),
    "hardware": (
        "cpu", "gpu", "cache", "memory", "dram", "ssd", "nvme", "bus", "pcie", "core",
        "thread", "pipeline", "branch", "predictor", "register", "instruction", "simd", "avx",
        "clock", "frequency", "voltage", "thermal", "motherboard", "chipset", "firmware",
        "bios", "fpga", "asic", "transistor", "silicon", "die", "bandwidth",
    ),
}

# Generic technical filler (shared by all topics -> realistic noise)
_FILLER_WORDS: tuple[str, ...] = (
    "system", "performance", "design", "implementation", "architecture", "issue", "problem",
    "solution", "approach", "version", "update", "configuration", "user", "team", "project",
    "production", "environment", "support", "documentation", "example", "result", "test",
)

# Stop words injected on purpose so StopWordsRemover has real work to do
_STOP_WORDS: tuple[str, ...] = (
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "was", "were", "this", "that", "it", "we", "our", "by", "from", "as", "at", "be",
)


def _generate_partition(
    partition_index: int,
    docs_per_partition: int,
    remainder: int,
    words_per_doc: tuple[int, int],
    topic_purity: float,
    seed: int,
) -> Iterator[str]:
    """Generate the documents of one partition (runs on an executor).

    Each partition is seeded with ``seed + partition_index`` so the corpus is
    fully reproducible regardless of scheduling. Lines are emitted in
    fastText format: ``__label__<topic> <text>``.
    """
    rng = random.Random(seed + partition_index)
    topics = list(TOPIC_VOCABULARY)
    n_docs = docs_per_partition + (1 if partition_index < remainder else 0)
    lo, hi = words_per_doc
    for _ in range(n_docs):
        topic = rng.choice(topics)
        own_vocab = TOPIC_VOCABULARY[topic]
        words: list[str] = []
        for _ in range(rng.randint(lo, hi)):
            r = rng.random()
            if r < 0.30:
                words.append(rng.choice(_STOP_WORDS))
            elif r < 0.42:
                words.append(rng.choice(_FILLER_WORDS))
            elif rng.random() < topic_purity:
                words.append(rng.choice(own_vocab))
            else:
                words.append(rng.choice(TOPIC_VOCABULARY[rng.choice(topics)]))
        # Light punctuation / casing noise to exercise the tokenizer.
        if words:
            words[0] = words[0].capitalize()
        yield f"__label__{topic} " + " ".join(words) + "."


def generate_synthetic_corpus(spark: SparkSession, cfg: DataConfig) -> DataFrame:
    """Generate a massive synthetic technical-text corpus **on the executors**.

    The driver only parallelises ``num_partitions`` integers; every executor
    task then produces ``~num_docs / num_partitions`` documents locally, so the
    corpus size is bounded by cluster capacity, not driver memory.

    Args:
        spark: Active SparkSession.
        cfg: Data configuration (size, partitions, seed, ...).

    Returns:
        DataFrame with a single column ``value: string`` — the exact schema
        produced by ``spark.read.text`` — so both paths share downstream code.

    DAG:
        ``parallelize -> flatMap`` — narrow only, **no shuffle**.
    """
    n_parts = max(1, cfg.synthetic_num_partitions)
    per_part, remainder = divmod(cfg.synthetic_num_docs, n_parts)
    words_per_doc, purity, seed = cfg.synthetic_words_per_doc, cfg.synthetic_topic_purity, cfg.seed

    rdd = spark.sparkContext.parallelize(range(n_parts), n_parts).flatMap(
        lambda pid: _generate_partition(pid, per_part, remainder, words_per_doc, purity, seed)
    )
    schema = StructType([StructField("value", StringType(), nullable=False)])
    return spark.createDataFrame(rdd.map(lambda line: (line,)), schema)


def path_exists(spark: SparkSession, path: str) -> bool:
    """Check existence of a DBFS / Volume / local path via the Hadoop FileSystem API."""
    try:
        jvm = spark.sparkContext._jvm  # type: ignore[attr-defined]
        hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
        fs = hadoop_path.getFileSystem(spark.sparkContext._jsc.hadoopConfiguration())  # type: ignore[attr-defined]
        return bool(fs.exists(hadoop_path))
    except Exception:  # noqa: BLE001 - py4j may be unavailable (e.g. Spark Connect)
        try:
            spark.read.text(path).limit(1).collect()
            return True
        except Exception:  # noqa: BLE001
            return False


def write_synthetic_corpus(spark: SparkSession, cfg: DataConfig) -> None:
    """Materialise the synthetic corpus as text files at ``cfg.input_path``."""
    logger.info("Generating %d synthetic docs -> %s", cfg.synthetic_num_docs, cfg.input_path)
    generate_synthetic_corpus(spark, cfg).write.mode("overwrite").text(cfg.input_path)


def _parse_lines(raw: DataFrame) -> DataFrame:
    """``value`` -> ``(doc_id, text, label)`` (strip optional fastText label prefix)."""
    label = F.regexp_extract("value", _LABEL_PREFIX_RE, 1)
    return (
        raw.select(
            F.monotonically_increasing_id().alias("doc_id"),
            F.trim(F.regexp_replace("value", _LABEL_PREFIX_RE, "")).alias("text"),
            F.when(label != "", label).alias("label"),
        )
        .where(F.length("text") > 0)
    )


def load_corpus(spark: SparkSession, cfg: DataConfig) -> DataFrame:
    """Load the text corpus from DBFS with ``spark.read.text`` (Phase 1, step 1).

    If ``cfg.input_path`` does not exist and ``cfg.generate_if_missing`` is set,
    a synthetic corpus is generated. With ``cfg.materialize_synthetic`` it is
    first written to ``input_path`` so the *same* ``spark.read.text`` code path
    is exercised (recommended on Databricks); otherwise the in-memory
    generator output is used directly (handy for local tests).

    Args:
        spark: Active SparkSession.
        cfg: Data configuration.

    Returns:
        DataFrame ``(doc_id: long, text: string, label: string|null)``.
        ``label`` is parsed from an optional fastText ``__label__<x>`` prefix
        (always present for synthetic data, typically null for real corpora).

    DAG:
        File scan + narrow projections only. ``monotonically_increasing_id`` is
        deterministic for a fixed input partitioning; the caller caches the
        downstream feature matrix so ids stay stable across actions.
    """
    if not path_exists(spark, cfg.input_path):
        if not cfg.generate_if_missing:
            raise FileNotFoundError(cfg.input_path)
        if not cfg.materialize_synthetic:
            logger.info("Using in-memory synthetic corpus (%d docs)", cfg.synthetic_num_docs)
            return _parse_lines(generate_synthetic_corpus(spark, cfg))
        write_synthetic_corpus(spark, cfg)

    raw = spark.read.text(cfg.input_path, wholetext=cfg.one_doc_per_file)
    return _parse_lines(raw)
