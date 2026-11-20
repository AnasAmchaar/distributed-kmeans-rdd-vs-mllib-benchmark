"""Real-time streaming HTTP + SSE server for the K-Means Benchmark Frontend.

Endpoints:
    * ``GET  /``                         -> serves ``frontend/index.html``
    * ``GET  /api/status``               -> server & ``mlflow.db`` status
    * ``GET  /api/benchmark/stream?...`` -> Server-Sent Events (SSE) stream that
      executes Phases 1-4 live on PySpark and pushes every phase transition and
      every single RDD Lloyd iteration (cost, shift, duration) in real time
    * ``POST /api/cluster``              -> clusters arbitrary raw text using
      the active ``DualDocumentClusterer`` (both RDD and MLlib models)
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from pyspark import StorageLevel

from kmeans_benchmark.config import (
    BenchmarkConfig,
    DataConfig,
    FeatureConfig,
    KMeansParams,
    TrackingConfig,
)
from kmeans_benchmark.evaluation import (
    adjusted_rand_index,
    agreement_between,
    cluster_sizes,
    silhouette_score,
    top_terms_per_cluster,
)
from kmeans_benchmark.inference import DualDocumentClusterer
from kmeans_benchmark.ingestion import load_corpus
from kmeans_benchmark.metrics import StageMetricsCollector
from kmeans_benchmark.mllib_kmeans import fit_mllib_kmeans
from kmeans_benchmark.preprocessing import build_tfidf, to_vector_rdd
from kmeans_benchmark.rdd_kmeans import CustomRDDKMeans, IterationStats
from kmeans_benchmark.spark_session import (
    get_spark,
    safe_persist,
    safe_unpersist,
    ship_package_to_executors,
)
from kmeans_benchmark.tracking import configure_mlflow, log_benchmark_context

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
MLFLOW_DB = ROOT_DIR / "mlflow.db"

_SPARK_LOCK = threading.Lock()
_ACTIVE_CLUSTERER: DualDocumentClusterer | None = None


def load_mlflow_sqlite_summary(db_path: Path = MLFLOW_DB) -> dict[str, Any]:
    """Read live runs summary from ``mlflow.db`` if present."""
    if not db_path.exists():
        return {"source": "embedded", "runs_count": 3, "has_live_clusterer": _ACTIVE_CLUSTERER is not None}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        runs = cur.execute(
            "SELECT run_uuid, name, status, start_time, end_time FROM runs WHERE lifecycle_stage='active'"
        ).fetchall()
        return {
            "source": "mlflow.db",
            "runs_count": len(runs),
            "has_live_clusterer": _ACTIVE_CLUSTERER is not None,
        }
    finally:
        conn.close()


class BenchmarkDashboardHandler(SimpleHTTPRequestHandler):
    """Serves ``frontend/index.html``, SSE benchmark streaming, and live inference."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    # ------------------------------------------------------------------ #
    # SSE helper
    # ------------------------------------------------------------------ #
    def _send_sse(self, event: str, data: dict[str, Any]) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ------------------------------------------------------------------ #
    # HTTP GET
    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self._send_json(200, load_mlflow_sqlite_summary())
            return
        if parsed.path == "/api/runs":
            self._send_json(200, {})
            return
        if parsed.path == "/api/benchmark/stream":
            self._handle_benchmark_stream(urllib.parse.parse_qs(parsed.query))
            return
        super().do_GET()

    # ------------------------------------------------------------------ #
    # HTTP POST (/api/cluster)
    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:  # noqa: N802
        global _ACTIVE_CLUSTERER
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/cluster":
            self._send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        body = json.loads(raw)
        text = str(body.get("text", "")).strip()
        if not text:
            self._send_json(400, {"error": "Empty text"})
            return

        with _SPARK_LOCK:
            spark = get_spark(local_cores="2")
            if _ACTIVE_CLUSTERER is None:
                cfg = BenchmarkConfig(
                    data=DataConfig(
                        materialize_synthetic=False,
                        synthetic_num_docs=600,
                        synthetic_num_partitions=4,
                    ),
                    features=FeatureConfig(vocab_size=1024, min_df=2.0),
                    kmeans=KMeansParams(k=8, max_iter=6, tol=1e-4, seed=42),
                )
                _ACTIVE_CLUSTERER = DualDocumentClusterer.fit(spark, cfg=cfg, top_n_terms=8)
            assignment = _ACTIVE_CLUSTERER.cluster_texts(spark, [text])[0]
            self._send_json(200, {"engine": "spark_live", "assignment": assignment.to_dict()})

    # ------------------------------------------------------------------ #
    # Live Streaming Benchmark Execution (Server-Sent Events)
    # ------------------------------------------------------------------ #
    def _handle_benchmark_stream(self, qs: dict[str, list[str]]) -> None:
        global _ACTIVE_CLUSTERER

        def _get_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(qs.get(key, [str(default)])[0])))
            except Exception:  # noqa: BLE001
                return default

        def _get_float(key: str, default: float) -> float:
            try:
                return float(qs.get(key, [str(default)])[0])
            except Exception:  # noqa: BLE001
                return default

        num_docs = _get_int("num_docs", 800, 100, 50_000)
        k = _get_int("k", 8, 2, 32)
        max_iter = _get_int("max_iter", 6, 1, 50)
        vocab_size = _get_int("vocab_size", 1024, 128, 32_768)
        num_partitions = _get_int("num_partitions", 4, 2, 64)
        seed = _get_int("seed", 42, 0, 999_999)
        tol = _get_float("tol", 1e-4)
        distance = qs.get("distance", ["euclidean"])[0]
        if distance not in ("euclidean", "cosine"):
            distance = "euclidean"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            with _SPARK_LOCK:
                t_start = time.perf_counter()
                self._send_sse(
                    "phase",
                    {
                        "phase": "spark_init",
                        "progress": 6,
                        "message": "Acquiring warm Apache Spark 3.5 session & executors...",
                    },
                )
                spark = get_spark(local_cores="2")
                ship_package_to_executors(spark)

                cfg = BenchmarkConfig(
                    data=DataConfig(
                        materialize_synthetic=False,
                        synthetic_num_docs=num_docs,
                        synthetic_num_partitions=num_partitions,
                        seed=seed,
                    ),
                    features=FeatureConfig(vocab_size=vocab_size, min_df=2.0),
                    kmeans=KMeansParams(
                        k=k,
                        max_iter=max_iter,
                        tol=tol,
                        seed=seed,
                        distance_measure=distance,  # type: ignore[arg-type]
                    ),
                    tracking=TrackingConfig(experiment_name="kmeans_rdd_vs_mllib_benchmark"),
                )

                # Phase 1a: Ingestion
                self._send_sse(
                    "phase",
                    {
                        "phase": "ingestion",
                        "progress": 14,
                        "message": f"Phase 1a: Generating {num_docs:,} synthetic technical docs across {num_partitions} Spark partitions...",
                    },
                )
                corpus_df = load_corpus(spark, cfg.data)

                # Phase 1b: TF-IDF
                self._send_sse(
                    "phase",
                    {
                        "phase": "tfidf",
                        "progress": 22,
                        "message": f"Phase 1b: Fitting distributed TF-IDF pipeline (vocab_size={vocab_size}, L2 Normalizer)...",
                    },
                )
                t_tfidf = time.perf_counter()
                tfidf = build_tfidf(corpus_df, cfg.features)
                tfidf_elapsed = time.perf_counter() - t_tfidf
                self._send_sse(
                    "tfidf_done",
                    {
                        "progress": 32,
                        "num_docs": tfidf.num_docs,
                        "num_features": tfidf.num_features,
                        "num_partitions": num_partitions,
                        "elapsed_s": round(tfidf_elapsed, 3),
                        "message": f"TF-IDF matrix cached: N={tfidf.num_docs:,} x D={tfidf.num_features} ({tfidf_elapsed:.2f}s)",
                    },
                )

                # Phase 2: Custom RDD K-Means with per-iteration live callback!
                self._send_sse(
                    "phase",
                    {
                        "phase": "rdd_start",
                        "progress": 36,
                        "message": f"Phase 2: Running CustomRDDKMeans (K={k}, maxIter={max_iter}, distance={distance})...",
                    },
                )
                rdd_history_payload: list[dict[str, Any]] = []
                rdd_t0 = time.perf_counter()

                def _on_iteration(stats: IterationStats) -> None:
                    pct = 36 + int((stats.iteration / max_iter) * 38)
                    item = {
                        "iter": stats.iteration,
                        "max_iter": max_iter,
                        "cost": round(stats.cost, 4),
                        "shift": stats.max_center_shift,
                        "empty_clusters": stats.empty_clusters,
                        "seconds": round(stats.duration_seconds, 3),
                        "cumulative_seconds": round(time.perf_counter() - rdd_t0, 3),
                        "progress": min(pct, 74),
                    }
                    rdd_history_payload.append(item)
                    self._send_sse("rdd_iteration", item)

                vectors = to_vector_rdd(tfidf.features_df)
                with StageMetricsCollector(spark, settle_polls=6, poll_s=0.15) as rdd_measure:
                    rdd_model = CustomRDDKMeans(
                        cfg.kmeans, iteration_callback=_on_iteration
                    ).fit(spark, vectors)
                    rdd_preds = safe_persist(
                        rdd_model.transform(spark, tfidf.features_df),
                        StorageLevel.MEMORY_AND_DISK,
                    )
                    rdd_preds.count()
                vectors.unpersist()

                self._send_sse(
                    "phase",
                    {
                        "phase": "rdd_eval",
                        "progress": 76,
                        "message": "Evaluating RDD clusters with ClusteringEvaluator (Silhouette & ARI)...",
                    },
                )
                rdd_sil = silhouette_score(rdd_preds, cfg.kmeans)
                rdd_ari = adjusted_rand_index(rdd_preds, "label", "prediction")
                rdd_sizes = cluster_sizes(rdd_preds)
                rdd_terms = top_terms_per_cluster(rdd_model.centers, tfidf.vocabulary, 10)
                rdd_sm = rdd_measure.metrics

                rdd_payload = {
                    "time": round(rdd_measure.elapsed_seconds, 3),
                    "iterations": rdd_model.num_iterations,
                    "converged": rdd_model.converged,
                    "silhouette": round(rdd_sil, 5),
                    "cost": round(rdd_model.training_cost, 4),
                    "ariTruth": round(rdd_ari, 4),
                    "shuffleRead": rdd_sm.shuffleReadBytes if rdd_sm else 0,
                    "shuffleWrite": rdd_sm.shuffleWriteBytes if rdd_sm else 0,
                    "inputBytes": rdd_sm.inputBytes if rdd_sm else 0,
                    "executorRunMs": rdd_sm.executorRunTime if rdd_sm else 0,
                    "executorCpuMs": round((rdd_sm.executorCpuTime / 1e6) if rdd_sm else 0.0, 2),
                    "gcMs": rdd_sm.jvmGcTime if rdd_sm else 0,
                    "stages": rdd_sm.num_stages if rdd_sm else 0,
                    "tasks": rdd_sm.num_tasks if rdd_sm else 0,
                    "history": rdd_history_payload,
                    "clusterSizes": rdd_sizes,
                    "topTerms": rdd_terms,
                }
                self._send_sse("rdd_done", {"progress": 80, "rdd": rdd_payload})

                # Phase 3: PySpark MLlib K-Means
                self._send_sse(
                    "phase",
                    {
                        "phase": "mllib_start",
                        "progress": 84,
                        "message": "Phase 3: Running PySpark MLlib KMeans (in-JVM BLAS / Tungsten)...",
                    },
                )
                with StageMetricsCollector(spark, settle_polls=6, poll_s=0.15) as mllib_measure:
                    mllib_res = fit_mllib_kmeans(tfidf.features_df, cfg.kmeans)

                mllib_sil = silhouette_score(mllib_res.predictions, cfg.kmeans)
                mllib_ari = adjusted_rand_index(mllib_res.predictions, "label", "prediction")
                mllib_sizes = cluster_sizes(mllib_res.predictions)
                mllib_centers = np.vstack(mllib_res.model.clusterCenters())
                mllib_terms = top_terms_per_cluster(mllib_centers, tfidf.vocabulary, 10)
                mllib_sm = mllib_measure.metrics

                mllib_payload = {
                    "time": round(mllib_measure.elapsed_seconds, 3),
                    "iterations": mllib_res.num_iterations,
                    "converged": mllib_res.num_iterations < max_iter,
                    "silhouette": round(mllib_sil, 5),
                    "cost": round(mllib_res.training_cost, 4),
                    "ariTruth": round(mllib_ari, 4),
                    "shuffleRead": mllib_sm.shuffleReadBytes if mllib_sm else 0,
                    "shuffleWrite": mllib_sm.shuffleWriteBytes if mllib_sm else 0,
                    "inputBytes": mllib_sm.inputBytes if mllib_sm else 0,
                    "executorRunMs": mllib_sm.executorRunTime if mllib_sm else 0,
                    "executorCpuMs": round(
                        (mllib_sm.executorCpuTime / 1e6) if mllib_sm else 0.0, 2
                    ),
                    "gcMs": mllib_sm.jvmGcTime if mllib_sm else 0,
                    "stages": mllib_sm.num_stages if mllib_sm else 0,
                    "tasks": mllib_sm.num_tasks if mllib_sm else 0,
                    "clusterSizes": mllib_sizes,
                    "topTerms": mllib_terms,
                }
                self._send_sse("mllib_done", {"progress": 92, "mllib": mllib_payload})

                # Phase 4: Cross-engine ARI & MLflow logging
                self._send_sse(
                    "phase",
                    {
                        "phase": "mlflow",
                        "progress": 96,
                        "message": "Phase 4: Computing cross-engine ARI & recording run in MLflow...",
                    },
                )
                ari_cross = agreement_between(rdd_preds, mllib_res.predictions)
                speedup = rdd_measure.elapsed_seconds / max(mllib_measure.elapsed_seconds, 1e-9)

                run_id = f"live_{int(time.time())}"
                try:
                    configure_mlflow(cfg.tracking)
                    with mlflow.start_run(run_name="kmeans_benchmark_live") as parent:
                        run_id = parent.info.run_id
                        log_benchmark_context(spark, cfg, tfidf)
                        mlflow.log_metrics(
                            {
                                "speedup_mllib_over_rdd": speedup,
                                "ari_rdd_vs_mllib": ari_cross,
                                "rdd_total_execution_time_seconds": rdd_measure.elapsed_seconds,
                                "mllib_total_execution_time_seconds": mllib_measure.elapsed_seconds,
                                "rdd_silhouette_score": rdd_sil,
                                "mllib_silhouette_score": mllib_sil,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("MLflow logging skipped during live stream: %s", exc)

                # Cache live clusterer for interactive document clustering
                _ACTIVE_CLUSTERER = DualDocumentClusterer(
                    tfidf=tfidf,
                    rdd_model=rdd_model,
                    mllib_model=mllib_res.model,
                    rdd_top_terms=rdd_terms,
                    mllib_top_terms=mllib_terms,
                )

                safe_unpersist(rdd_preds)
                safe_unpersist(mllib_res.predictions)

                run_key = f"run_live_{int(time.time())}"
                full_run = {
                    "id": run_id,
                    "label": f"LIVE · {distance.capitalize()} (N={tfidf.num_docs:,}, K={k}, maxIter={max_iter})",
                    "date": "Just now (Live Stream)",
                    "numDocs": tfidf.num_docs,
                    "numFeatures": tfidf.num_features,
                    "numPartitions": num_partitions,
                    "k": k,
                    "maxIter": max_iter,
                    "tol": tol,
                    "distance": distance,
                    "speedup": round(speedup, 2),
                    "ariAgreement": round(ari_cross, 4),
                    "finding": (
                        f"Live Run (N={tfidf.num_docs:,}, K={k}, {distance}): MLlib finished in "
                        f"{mllib_payload['time']:.2f}s ({mllib_payload['iterations']} iters) vs Custom RDD in "
                        f"{rdd_payload['time']:.2f}s ({rdd_payload['iterations']} iters) — {speedup:.2f}× speedup, "
                        f"Silhouette {rdd_payload['silhouette']:.4f} (RDD) vs {mllib_payload['silhouette']:.4f} (MLlib)."
                    ),
                    "rdd": rdd_payload,
                    "mllib": mllib_payload,
                }
                self._send_sse(
                    "complete",
                    {
                        "progress": 100,
                        "run_key": run_key,
                        "run": full_run,
                        "total_stream_seconds": round(time.perf_counter() - t_start, 2),
                    },
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("Client closed SSE stream")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in live benchmark stream")
            try:
                self._send_sse("error", {"message": str(exc)})
            except Exception:  # noqa: BLE001
                pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the K-Means Benchmark Frontend Dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), BenchmarkDashboardHandler)
    print(f"Benchmark Frontend Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
