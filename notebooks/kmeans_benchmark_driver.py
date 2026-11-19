# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed Document Clustering — K-Means Benchmark (RDD vs PySpark MLlib)
# MAGIC
# MAGIC | Phase | Content | Module |
# MAGIC |---|---|---|
# MAGIC | 1 | Ingestion (`spark.read.text` on DBFS / synthetic fallback) + TF-IDF | `ingestion.py`, `preprocessing.py` |
# MAGIC | 2 | From-scratch MapReduce K-Means on RDDs (`CustomRDDKMeans`) | `rdd_kmeans.py`, `vector_ops.py` |
# MAGIC | 3 | `pyspark.ml.clustering.KMeans` with identical hyper-parameters | `mllib_kmeans.py` |
# MAGIC | 4 | MLflow tracking: params, silhouette, time, shuffle bytes, model | `tracking.py`, `metrics.py` |
# MAGIC
# MAGIC **Cluster requirements:** Databricks Runtime ML 14.3 LTS+ (Spark 3.5), **Dedicated / Single-user access mode**
# MAGIC (RDD APIs and `sparkContext` are not available on Serverless or Standard/shared access mode).

# COMMAND ----------

# MAGIC %md ## 0. Setup — import the project package from the Git folder

# COMMAND ----------

import logging
import os
import subprocess
import sys

# 1. If running inside a Databricks Git folder, load from ../src.
# 2. If running as a standalone notebook in Databricks, auto-install from the public GitHub repo.
SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if os.path.isdir(os.path.join(SRC_DIR, "kmeans_benchmark")):
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
else:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--upgrade",
            "git+https://github.com/AnasAmchaar/distributed-kmeans-rdd-vs-mllib-benchmark.git",
        ]
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

from kmeans_benchmark.spark_session import get_spark_context, ship_package_to_executors

# Executors must import the map/reduce functions -> ship the package as a zip (no-op on Serverless).
ship_package_to_executors(spark)

# COMMAND ----------

dbutils.widgets.text("input_path", "dbfs:/FileStore/kmeans_benchmark/synthetic_corpus", "Input path (DBFS / Volume)")
dbutils.widgets.text("num_docs", "200000", "Synthetic #docs (if path missing)")
dbutils.widgets.text("k", "8", "K clusters")
dbutils.widgets.text("max_iter", "20", "maxIter")
dbutils.widgets.text("tol", "1e-4", "tolerance")
dbutils.widgets.text("seed", "42", "seed")
dbutils.widgets.dropdown("distance", "euclidean", ["euclidean", "cosine"], "Distance")
dbutils.widgets.text("vocab_size", "16384", "Vocabulary size (D)")

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

from kmeans_benchmark.config import BenchmarkConfig, DataConfig, FeatureConfig, KMeansParams, TrackingConfig

cfg = BenchmarkConfig(
    data=DataConfig(
        input_path=dbutils.widgets.get("input_path"),
        synthetic_num_docs=int(dbutils.widgets.get("num_docs")),
        synthetic_num_partitions=max(16, get_spark_context(spark).defaultParallelism * 2),
        seed=int(dbutils.widgets.get("seed")),
    ),
    features=FeatureConfig(vocab_size=int(dbutils.widgets.get("vocab_size")), min_df=5.0),
    kmeans=KMeansParams(
        k=int(dbutils.widgets.get("k")),
        max_iter=int(dbutils.widgets.get("max_iter")),
        tol=float(dbutils.widgets.get("tol")),
        seed=int(dbutils.widgets.get("seed")),
        distance_measure=dbutils.widgets.get("distance"),
    ),
    tracking=TrackingConfig(experiment_name=f"/Users/{user}/kmeans_rdd_vs_mllib_benchmark"),
)
cfg

# COMMAND ----------

# MAGIC %md ## Phase 1 — Data ingestion & distributed TF-IDF

# COMMAND ----------

from kmeans_benchmark.ingestion import load_corpus

corpus_df = load_corpus(spark, cfg.data)  # spark.read.text(...) ; synthetic corpus generated on executors if missing
display(corpus_df.limit(5))

# COMMAND ----------

from kmeans_benchmark.preprocessing import build_tfidf, to_vector_rdd

tfidf = build_tfidf(corpus_df, cfg.features)  # cached + materialised N x D sparse matrix
print(f"N = {tfidf.num_docs:,} documents | D = {tfidf.num_features:,} features")
tfidf.features_df.printSchema()  # features: vector (VectorUDT)  -> MLlib
print(to_vector_rdd(tfidf.features_df).first())  # (doc_id, SparseVector) -> RDD K-Means

# COMMAND ----------

# MAGIC %md ## Phase 2 — From-scratch RDD K-Means (exploration run)
# MAGIC Broadcast centroids → `map` to `(closest_k, (x, 1))` → `aggregateByKey` (one shuffle) → driver update → repeat until `max shift ≤ tol`.

# COMMAND ----------

from kmeans_benchmark.rdd_kmeans import CustomRDDKMeans

vectors_rdd = to_vector_rdd(tfidf.features_df)
rdd_model, assignments_rdd = CustomRDDKMeans(cfg.kmeans).fit_predict(spark, vectors_rdd)
print(f"iterations={rdd_model.num_iterations} converged={rdd_model.converged} cost={rdd_model.training_cost:.4f}")
print("centroids shape:", rdd_model.centers.shape)
print("sample assignments (doc_id, cluster):", assignments_rdd.take(5))
display(spark.createDataFrame([h.__dict__ if hasattr(h, "__dict__") else
                               {f: getattr(h, f) for f in h.__dataclass_fields__} for h in rdd_model.history]))

# COMMAND ----------

rdd_model.release()
vectors_rdd.unpersist()

# COMMAND ----------

# MAGIC %md ## Phase 3 — PySpark MLlib K-Means (exploration run)

# COMMAND ----------

from kmeans_benchmark.mllib_kmeans import fit_mllib_kmeans

mllib_res = fit_mllib_kmeans(tfidf.features_df, cfg.kmeans)
print(f"iterations={mllib_res.num_iterations} cost={mllib_res.training_cost:.4f}")
display(mllib_res.predictions.groupBy("prediction").count().orderBy("prediction"))
mllib_res.predictions.unpersist()

# COMMAND ----------

# MAGIC %md ## Phase 4 — Tracked benchmark (MLflow parent run + nested RDD / MLlib runs)

# COMMAND ----------

from kmeans_benchmark.benchmark import cleanup, run_benchmark

report = run_benchmark(spark, cfg, corpus_df=corpus_df, tfidf=tfidf, track=True)
display(spark.createDataFrame(report.summary_rows()))
print(f"ARI(RDD vs MLlib) = {report.ari_rdd_vs_mllib:.4f}")
print(f"Speed-up MLlib over RDD = x{report.speedup_mllib_over_rdd:.2f}")
print(f"MLflow parent run: {report.parent_run_id}")

# COMMAND ----------

import json

print("Top terms per cluster (MLlib):")
print(json.dumps({k: v[:8] for k, v in report.mllib.top_terms.items()}, indent=2))

# COMMAND ----------

# MAGIC %md ## Phase 5 — Interactive Document Clustering (Cluster Any Custom Text with Both Approaches)

# COMMAND ----------

from kmeans_benchmark.inference import DualDocumentClusterer

clusterer = DualDocumentClusterer.from_benchmark(tfidf, report, top_n_terms=6)

sample_documents = [
    "PostgreSQL B-tree index deadlock during ACID transaction commit and SQL query optimization.",
    "Kubernetes pod autoscaling with Docker containers, Terraform, and AWS EKS ingress.",
    "Transformer neural network training with gradient descent, backpropagation, and attention layers.",
    "TLS encryption certificate vulnerability, OAuth JWT token authentication, and zero trust firewall.",
    "React TypeScript component rendering in the browser DOM with Tailwind CSS flexbox layout.",
]

assignments = clusterer.cluster_texts(spark, sample_documents)
display(spark.createDataFrame([a.to_dict() for a in assignments]))

# COMMAND ----------

# MAGIC %md ### Optional: sweep K (each K logs its own parent run)

# COMMAND ----------

from dataclasses import replace

SWEEP = False
if SWEEP:
    for k in (4, 8, 16, 32):
        r = run_benchmark(spark, replace(cfg, kmeans=replace(cfg.kmeans, k=k)), corpus_df=corpus_df, tfidf=tfidf)
        print(k, r.summary_rows())
        cleanup(r)

# COMMAND ----------

cleanup(report, tfidf)

