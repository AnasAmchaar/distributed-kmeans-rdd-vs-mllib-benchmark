"""Real-time streaming HTTP + SSE server for Mini-Projet 10 (K-Means RDD L1/L2 vs MLlib).

Endpoints:
    * ``GET  /``                            -> serves ``frontend/index.html``
    * ``GET  /api/mp10``                    -> returns ``benchmark_results_58d.json``
    * ``GET  /kmeans_convergence_report.pdf`` -> serves the generated PDF report deliverable
    * ``GET  /api/benchmark/stream?...``    -> Server-Sent Events (SSE) stream that
      executes the Mini-Projet 10 benchmark live on PySpark (58D vectors, RDD L2
      Euclidean + RDD L1 Manhattan with ``reduceByKey`` & WSSE per iteration,
      MLlib ``k-means||``, Elbow method & Silhouette, 2D PCA projection, and
      live regeneration of ``kmeans_convergence_report.pdf``).
    * ``POST /api/cluster``                 -> clusters a 58D vector or raw text document
      with both RDD (L2 & L1) and MLlib models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import urllib.parse
from dataclasses import asdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_kmeans import (  # noqa: E402
    _evaluate_rdd_silhouette,
    compute_elbow_and_silhouette,
    compute_pca_2d_projection,
    generate_convergence_pdf_report,
)
from custom_kmeans_rdd import (  # noqa: E402
    CustomKMeansRDD,
    RDDIterationRecord,
    build_58d_dataframe,
    generate_58d_document_vectors,
)
from kmeans_benchmark.spark_session import (  # noqa: E402
    get_spark,
    safe_persist,
    safe_unpersist,
    ship_package_to_executors,
)

logger = logging.getLogger(__name__)

FRONTEND_DIR = ROOT_DIR / "frontend"
MP10_JSON = ROOT_DIR / "benchmark_results_58d.json"
MP10_PDF = ROOT_DIR / "kmeans_convergence_report.pdf"

_SPARK_LOCK = threading.Lock()


class BenchmarkDashboardHandler(SimpleHTTPRequestHandler):
    """Serves ``frontend/index.html``, ``/api/mp10``, PDF report, and SSE live stream."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/mp10":
            if MP10_JSON.exists():
                data = json.loads(MP10_JSON.read_text(encoding="utf-8"))
                self._send_json(200, data)
            else:
                self._send_json(200, {})
            return
        if parsed.path == "/kmeans_convergence_report.pdf" and MP10_PDF.exists():
            pdf_bytes = MP10_PDF.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition", 'inline; filename="kmeans_convergence_report.pdf"'
            )
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return
        if parsed.path == "/api/benchmark/stream":
            self._handle_mp10_stream(urllib.parse.parse_qs(parsed.query))
            return
        super().do_GET()

    def _handle_mp10_stream(self, qs: dict[str, list[str]]) -> None:
        def _get_int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(qs.get(key, [str(default)])[0])))
            except Exception:  # noqa: BLE001
                return default

        num_docs = _get_int("num_docs", 1000, 150, 20_000)
        k = _get_int("k", 5, 2, 15)
        max_iter = _get_int("max_iter", 10, 2, 30)
        seed = _get_int("seed", 42, 0, 999_999)

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
                        "phase": "SPARK_INIT",
                        "progress": 5,
                        "message": "Initialisation de la session Apache Spark 3.5 & exécuteurs...",
                    },
                )
                spark = get_spark(local_cores="2")
                ship_package_to_executors(spark)

                # Section 3: Représentation Vectorielle des Documents (58D)
                self._send_sse(
                    "phase",
                    {
                        "phase": "VECTEURS_58D",
                        "progress": 12,
                        "message": f"Génération et mise en cache de {num_docs:,} vecteurs de documents 58D ([0.0, 0.64, 0.64, ..., 3.78, 1.32, 45.0])...",
                    },
                )
                matrix_58d, _ = generate_58d_document_vectors(
                    num_docs=num_docs, k_true=k, seed=seed
                )
                sample_58d = [round(float(v), 2) for v in matrix_58d[0].tolist()]
                df_58d = safe_persist(
                    build_58d_dataframe(
                        spark, num_docs=num_docs, k_true=k, seed=seed, num_partitions=4
                    ),
                    StorageLevel.MEMORY_AND_DISK,
                )
                df_58d.count()

                # 1. K-Means Manuel sur RDD — Distance Euclidienne (L2)
                self._send_sse(
                    "phase",
                    {
                        "phase": "RDD_L2_START",
                        "progress": 18,
                        "message": f"Démarrage K-Means Manuel RDD — Distance Euclidienne (L2), K={k}, maxIter={max_iter} (Map + reduceByKey + WSSE)...",
                    },
                )

                def _on_l2_iter(rec: RDDIterationRecord) -> None:
                    pct = 18 + int((rec.iteration / max_iter) * 24)
                    self._send_sse(
                        "rdd_l2_iter",
                        {
                            "iteration": rec.iteration,
                            "max_iter": max_iter,
                            "wsse": round(rec.wsse, 2),
                            "max_shift": rec.max_shift,
                            "duration_seconds": round(rec.duration_seconds, 3),
                            "progress": min(pct, 42),
                        },
                    )

                rdd_l2 = CustomKMeansRDD(
                    k=k,
                    max_iter=max_iter,
                    tol=1e-4,
                    distance="euclidean",
                    seed=seed,
                    iteration_callback=_on_l2_iter,
                ).fit(spark, df_58d)
                rdd_l2_sil = _evaluate_rdd_silhouette(spark, df_58d, rdd_l2)

                # 2. K-Means Manuel sur RDD — Distance Manhattan (L1)
                self._send_sse(
                    "phase",
                    {
                        "phase": "RDD_L1_START",
                        "progress": 45,
                        "message": f"Démarrage K-Means Manuel RDD — Distance Manhattan (L1), K={k}, maxIter={max_iter} (Map + reduceByKey + WSSE)...",
                    },
                )

                def _on_l1_iter(rec: RDDIterationRecord) -> None:
                    pct = 45 + int((rec.iteration / max_iter) * 24)
                    self._send_sse(
                        "rdd_l1_iter",
                        {
                            "iteration": rec.iteration,
                            "max_iter": max_iter,
                            "wsse": round(rec.wsse, 2),
                            "max_shift": rec.max_shift,
                            "duration_seconds": round(rec.duration_seconds, 3),
                            "progress": min(pct, 69),
                        },
                    )

                rdd_l1 = CustomKMeansRDD(
                    k=k,
                    max_iter=max_iter,
                    tol=1e-4,
                    distance="manhattan",
                    seed=seed,
                    iteration_callback=_on_l1_iter,
                ).fit(spark, df_58d)
                rdd_l1_sil = _evaluate_rdd_silhouette(spark, df_58d, rdd_l1)

                # 3. KMeans PySpark MLlib API (Initialisation parallèle K-Means||)
                self._send_sse(
                    "phase",
                    {
                        "phase": "MLLIB_KMEANS_II",
                        "progress": 73,
                        "message": "Exécution de PySpark MLlib KMeans (initMode='k-means||', Catalyst / Tungsten DataFrames)...",
                    },
                )
                t_ml = time.perf_counter()
                mllib_km = KMeans(
                    featuresCol="features",
                    predictionCol="prediction",
                    k=k,
                    maxIter=max_iter,
                    tol=1e-4,
                    seed=seed,
                    initMode="k-means||",
                    distanceMeasure="euclidean",
                )
                mllib_model = mllib_km.fit(df_58d)
                mllib_preds = mllib_model.transform(df_58d)
                mllib_preds.count()
                mllib_time = time.perf_counter() - t_ml

                evaluator = ClusteringEvaluator(
                    featuresCol="features",
                    predictionCol="prediction",
                    metricName="silhouette",
                    distanceMeasure="squaredEuclidean",
                )
                mllib_sil = float(evaluator.evaluate(mllib_preds))
                mllib_wsse = float(mllib_model.summary.trainingCost)
                mllib_iters = int(mllib_model.summary.numIter)

                # 4. Méthode du coude (Elbow) & Silhouette
                self._send_sse(
                    "phase",
                    {
                        "phase": "ELBOW_SILHOUETTE",
                        "progress": 82,
                        "message": "Calcul de la Méthode du Coude (Elbow WSSE) & Silhouette Score pour K in {2, 3, 4, 5, 6, 8, 10}...",
                    },
                )
                elbow = compute_elbow_and_silhouette(
                    df_58d, k_values=(2, 3, 4, 5, 6, 8, 10), max_iter=max_iter, seed=seed
                )

                # 5. Projection PCA 2D + Génération du PDF
                self._send_sse(
                    "phase",
                    {
                        "phase": "PCA_AND_PDF",
                        "progress": 93,
                        "message": "Projection PCA 2D (58D -> 2D) et génération de kmeans_convergence_report.pdf...",
                    },
                )
                pca_points, pca_centroids, explained = compute_pca_2d_projection(
                    spark, df_58d, rdd_l2
                )
                safe_unpersist(df_58d)

                speedup_l2 = rdd_l2.total_time_seconds / max(mllib_time, 1e-9)
                speedup_l1 = rdd_l1.total_time_seconds / max(mllib_time, 1e-9)

                payload = {
                    "num_docs": num_docs,
                    "dim": 58,
                    "k": k,
                    "max_iter": max_iter,
                    "sample_vector_58d": sample_58d,
                    "rdd_l2": {
                        "distance": "Euclidean (L2)",
                        "num_iterations": rdd_l2.num_iterations,
                        "converged": rdd_l2.converged,
                        "final_wsse": round(rdd_l2.final_wsse, 4),
                        "silhouette": round(rdd_l2_sil, 5),
                        "total_time_seconds": round(rdd_l2.total_time_seconds, 3),
                        "avg_iter_seconds": round(
                            rdd_l2.total_time_seconds / max(rdd_l2.num_iterations, 1), 3
                        ),
                        "history": [asdict(h) for h in rdd_l2.history],
                    },
                    "rdd_l1": {
                        "distance": "Manhattan (L1)",
                        "num_iterations": rdd_l1.num_iterations,
                        "converged": rdd_l1.converged,
                        "final_wsse": round(rdd_l1.final_wsse, 4),
                        "silhouette": round(rdd_l1_sil, 5),
                        "total_time_seconds": round(rdd_l1.total_time_seconds, 3),
                        "avg_iter_seconds": round(
                            rdd_l1.total_time_seconds / max(rdd_l1.num_iterations, 1), 3
                        ),
                        "history": [asdict(h) for h in rdd_l1.history],
                    },
                    "mllib": {
                        "init_mode": "k-means||",
                        "num_iterations": mllib_iters,
                        "final_wsse": round(mllib_wsse, 4),
                        "silhouette": round(mllib_sil, 5),
                        "total_time_seconds": round(mllib_time, 3),
                        "avg_iter_seconds": round(mllib_time / max(mllib_iters, 1), 3),
                    },
                    "speedup_vs_rdd_l2": round(speedup_l2, 2),
                    "speedup_vs_rdd_l1": round(speedup_l1, 2),
                    "elbow_curve": [asdict(e) for e in elbow],
                    "pca_explained_variance": explained,
                    "pca_centroids_l2": pca_centroids,
                    "pca_points": pca_points,
                }

                MP10_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                generate_convergence_pdf_report(payload, MP10_PDF)

                self._send_sse(
                    "complete",
                    {
                        "progress": 100,
                        "report": payload,
                        "total_stream_seconds": round(time.perf_counter() - t_start, 2),
                    },
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("Client closed SSE stream")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error in live MP10 stream")
            try:
                self._send_sse("error", {"message": str(exc)})
            except Exception:  # noqa: BLE001
                pass

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Serve the Mini-Projet 10 Benchmark Frontend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), BenchmarkDashboardHandler)
    print(f"Mini-Projet 10 Frontend Dashboard running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
