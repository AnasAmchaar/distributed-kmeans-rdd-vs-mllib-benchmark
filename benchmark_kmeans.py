"""Mini-Projet 10 — Livrable 2 & 3 : ``benchmark_kmeans.py`` -> ``kmeans_convergence_report.pdf``

Benchmark comparatif approfondi entre :
    1. **K-Means Manuel sur RDDs** (``custom_kmeans_rdd.py``) :
       - Distance Euclidienne (L2) & Distance Manhattan (L1)
       - Étape Map (assignation des centroïdes) & Étape Reduce (``reduceByKey``)
       - Calcul du coût intra-cluster **WSSE** et du temps d'exécution à chaque itération
    2. **KMeans PySpark MLlib API** :
       - Initialisation parallèle **K-Means||** (``initMode="k-means||"``)
       - **Méthode du coude (Elbow)** & **Silhouette score** (``ClusteringEvaluator``)
       - Optimisation Catalyst / Tungsten sur DataFrames
    3. **Représentation Vectorielle des Documents (58D)** :
       ``[0.0, 0.64, 0.64, 0.0, 0.32, 0.0, 0.0, 0.0, 0.96, ..., 3.78, 1.32, 45.0]``
    4. **Génération automatique du rapport PDF** : ``kmeans_convergence_report.pdf``
       (Courbes de convergence WSSE L1/L2, temps par itération, Elbow/Silhouette, et
       projection PCA 2D des clusters).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from custom_kmeans_rdd import (  # noqa: E402
    CustomKMeansRDD,
    CustomKMeansRDDResult,
    build_58d_dataframe,
    generate_58d_document_vectors,
)
from kmeans_benchmark.spark_session import get_spark, safe_persist, safe_unpersist  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ElbowPoint:
    """Point de la méthode du coude (Elbow) & Silhouette pour un K donné."""

    k: int
    wsse: float
    silhouette: float
    time_seconds: float


@dataclass(slots=True)
class MiniProjet10Report:
    """Résultats consolidés du Mini-Projet 10 (58D, L1/L2, MLlib K-Means||, Elbow, PCA 2D)."""

    num_docs: int
    dim: int
    k: int
    max_iter: int
    sample_vector_58d: list[float]
    rdd_l2: CustomKMeansRDDResult
    rdd_l1: CustomKMeansRDDResult
    rdd_l2_silhouette: float
    rdd_l1_silhouette: float
    mllib_wsse: float
    mllib_silhouette: float
    mllib_iterations: int
    mllib_time_seconds: float
    mllib_per_iter_seconds: float
    speedup_vs_rdd_l2: float
    speedup_vs_rdd_l1: float
    elbow_curve: list[ElbowPoint]
    pca_points: list[dict[str, float | int]]
    pca_centroids_l2: list[dict[str, float | int]]
    pca_explained_variance: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_docs": self.num_docs,
            "dim": self.dim,
            "k": self.k,
            "max_iter": self.max_iter,
            "sample_vector_58d": self.sample_vector_58d,
            "rdd_l2": {
                "distance": "Euclidean (L2)",
                "num_iterations": self.rdd_l2.num_iterations,
                "converged": self.rdd_l2.converged,
                "final_wsse": round(self.rdd_l2.final_wsse, 4),
                "silhouette": round(self.rdd_l2_silhouette, 5),
                "total_time_seconds": round(self.rdd_l2.total_time_seconds, 3),
                "avg_iter_seconds": round(
                    self.rdd_l2.total_time_seconds / max(self.rdd_l2.num_iterations, 1), 3
                ),
                "history": [asdict(h) for h in self.rdd_l2.history],
            },
            "rdd_l1": {
                "distance": "Manhattan (L1)",
                "num_iterations": self.rdd_l1.num_iterations,
                "converged": self.rdd_l1.converged,
                "final_wsse": round(self.rdd_l1.final_wsse, 4),
                "silhouette": round(self.rdd_l1_silhouette, 5),
                "total_time_seconds": round(self.rdd_l1.total_time_seconds, 3),
                "avg_iter_seconds": round(
                    self.rdd_l1.total_time_seconds / max(self.rdd_l1.num_iterations, 1), 3
                ),
                "history": [asdict(h) for h in self.rdd_l1.history],
            },
            "mllib": {
                "init_mode": "k-means||",
                "num_iterations": self.mllib_iterations,
                "final_wsse": round(self.mllib_wsse, 4),
                "silhouette": round(self.mllib_silhouette, 5),
                "total_time_seconds": round(self.mllib_time_seconds, 3),
                "avg_iter_seconds": round(self.mllib_per_iter_seconds, 3),
            },
            "speedup_vs_rdd_l2": round(self.speedup_vs_rdd_l2, 2),
            "speedup_vs_rdd_l1": round(self.speedup_vs_rdd_l1, 2),
            "elbow_curve": [asdict(e) for e in self.elbow_curve],
            "pca_explained_variance": self.pca_explained_variance,
            "pca_centroids_l2": self.pca_centroids_l2,
            "pca_points": self.pca_points,
        }


def _evaluate_rdd_silhouette(
    spark: SparkSession, df_58d: DataFrame, result: CustomKMeansRDDResult
) -> float:
    """Calcule le Silhouette score d'un modèle RDD via ``ClusteringEvaluator``."""
    rows = df_58d.select("doc_id", "features").collect()
    scored = [(int(r[0]), r[1], int(result.predict_one(r[1]))) for r in rows]
    pred_df = spark.createDataFrame(scored, ["doc_id", "features", "prediction"])
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    return float(evaluator.evaluate(pred_df))


def compute_elbow_and_silhouette(
    df_58d: DataFrame,
    k_values: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10),
    max_iter: int = 12,
    seed: int = 42,
) -> list[ElbowPoint]:
    """Méthode du coude (Elbow WSSE) & Silhouette score sur PySpark MLlib (K-Means||)."""
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )
    curve: list[ElbowPoint] = []
    for k_val in k_values:
        t0 = time.perf_counter()
        km = KMeans(
            featuresCol="features",
            predictionCol="prediction",
            k=k_val,
            maxIter=max_iter,
            tol=1e-4,
            seed=seed,
            initMode="k-means||",
            distanceMeasure="euclidean",
        )
        model = km.fit(df_58d)
        preds = model.transform(df_58d)
        sil = float(evaluator.evaluate(preds))
        wsse = float(model.summary.trainingCost)
        dt = time.perf_counter() - t0
        curve.append(ElbowPoint(k=k_val, wsse=round(wsse, 2), silhouette=round(sil, 4), time_seconds=round(dt, 3)))
        logger.info("[Elbow K=%d] WSSE=%.2f Silhouette=%.4f (%.2fs)", k_val, wsse, sil, dt)
    return curve


def compute_pca_2d_projection(
    spark: SparkSession,
    df_58d: DataFrame,
    rdd_l2: CustomKMeansRDDResult,
    max_points: int = 400,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]], list[float]]:
    """Projette les vecteurs 58D et les centroïdes K-Means en 2D via ``pyspark.ml.feature.PCA(k=2)``."""
    pca_model = PCA(k=2, inputCol="features", outputCol="pca_features").fit(df_58d)
    explained = [round(float(v) * 100.0, 2) for v in pca_model.explainedVariance.toArray()]

    projected_rows = (
        pca_model.transform(df_58d).select("doc_id", "features", "pca_features").limit(max_points).collect()
    )
    points: list[dict[str, float | int]] = []
    for row in projected_rows:
        xy = row["pca_features"].toArray()
        cluster_id = int(rdd_l2.predict_one(row["features"]))
        points.append(
            {
                "doc_id": int(row["doc_id"]),
                "x": round(float(xy[0]), 4),
                "y": round(float(xy[1]), 4),
                "cluster": cluster_id,
            }
        )

    # Project the K centroids using the same PCA transformation matrix
    centroid_rows = [(i, Vectors.dense(rdd_l2.centers[i].tolist())) for i in range(rdd_l2.k)]
    centroid_df = spark.createDataFrame(centroid_rows, ["cluster", "features"])
    proj_centroids = pca_model.transform(centroid_df).select("cluster", "pca_features").collect()
    centroids_2d = [
        {
            "cluster": int(r["cluster"]),
            "x": round(float(r["pca_features"].toArray()[0]), 4),
            "y": round(float(r["pca_features"].toArray()[1]), 4),
        }
        for r in proj_centroids
    ]
    return points, centroids_2d, explained


def run_mini_projet_10_benchmark(
    spark: SparkSession,
    num_docs: int = 1_200,
    k: int = 5,
    max_iter: int = 12,
    seed: int = 42,
) -> MiniProjet10Report:
    """Exécute le benchmark complet du Mini-Projet 10 (RDD L2, RDD L1, MLlib K-Means||, Elbow, PCA 2D)."""
    matrix_58d, _ = generate_58d_document_vectors(num_docs=num_docs, k_true=k, seed=seed)
    sample_58d = [round(float(v), 2) for v in matrix_58d[0].tolist()]

    df_58d = safe_persist(
        build_58d_dataframe(spark, num_docs=num_docs, k_true=k, seed=seed, num_partitions=4),
        StorageLevel.MEMORY_AND_DISK,
    )
    df_58d.count()

    # 1. K-Means Manuel sur RDD — Distance Euclidienne (L2)
    rdd_l2 = CustomKMeansRDD(
        k=k, max_iter=max_iter, tol=1e-4, distance="euclidean", seed=seed
    ).fit(spark, df_58d)
    rdd_l2_sil = _evaluate_rdd_silhouette(spark, df_58d, rdd_l2)

    # 2. K-Means Manuel sur RDD — Distance Manhattan (L1)
    rdd_l1 = CustomKMeansRDD(
        k=k, max_iter=max_iter, tol=1e-4, distance="manhattan", seed=seed
    ).fit(spark, df_58d)
    rdd_l1_sil = _evaluate_rdd_silhouette(spark, df_58d, rdd_l1)

    # 3. PySpark MLlib KMeans API (Initialisation parallèle K-Means|| + Catalyst/Tungsten)
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
    elbow = compute_elbow_and_silhouette(df_58d, k_values=(2, 3, 4, 5, 6, 8, 10), max_iter=max_iter, seed=seed)

    # 5. Projection PCA 2D des clusters (58D -> 2D)
    pca_points, pca_centroids, explained = compute_pca_2d_projection(spark, df_58d, rdd_l2)

    safe_unpersist(df_58d)
    return MiniProjet10Report(
        num_docs=num_docs,
        dim=58,
        k=k,
        max_iter=max_iter,
        sample_vector_58d=sample_58d,
        rdd_l2=rdd_l2,
        rdd_l1=rdd_l1,
        rdd_l2_silhouette=rdd_l2_sil,
        rdd_l1_silhouette=rdd_l1_sil,
        mllib_wsse=mllib_wsse,
        mllib_silhouette=mllib_sil,
        mllib_iterations=mllib_iters,
        mllib_time_seconds=mllib_time,
        mllib_per_iter_seconds=mllib_time / max(mllib_iters, 1),
        speedup_vs_rdd_l2=rdd_l2.total_time_seconds / max(mllib_time, 1e-9),
        speedup_vs_rdd_l1=rdd_l1.total_time_seconds / max(mllib_time, 1e-9),
        elbow_curve=elbow,
        pca_points=pca_points,
        pca_centroids_l2=pca_centroids,
        pca_explained_variance=explained,
    )


def generate_convergence_pdf_report(
    report: MiniProjet10Report | dict[str, Any],
    output_pdf_path: Path = Path("kmeans_convergence_report.pdf"),
) -> Path:
    """Génère le livrable 3 : ``kmeans_convergence_report.pdf`` (Courbes WSSE L1/L2,
    Projection PCA 2D des clusters, Méthode du Coude / Silhouette, et Temps par Itération)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    data = report.to_dict() if isinstance(report, MiniProjet10Report) else report
    rdd_l2 = data["rdd_l2"]
    rdd_l1 = data["rdd_l1"]
    mllib = data["mllib"]
    elbow = data["elbow_curve"]
    pca_pts = data["pca_points"]
    pca_cnt = data["pca_centroids_l2"]
    pca_var = data["pca_explained_variance"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.edgecolor": "#d4d4d8",
            "axes.linewidth": 0.8,
        }
    )

    with PdfPages(str(output_pdf_path)) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), dpi=150)  # A4 Landscape
        fig.patch.set_facecolor("#ffffff")

        # Title Header Banner
        fig.text(
            0.04,
            0.95,
            "MINI-PROJET 10 — Clustering K-Means : RDD (L1/L2) vs PySpark MLlib (K-Means||)",
            fontsize=15,
            fontweight="bold",
            color="#09090b",
        )
        fig.text(
            0.04,
            0.92,
            f"Représentation Vectorielle des Documents (58D) · N = {data['num_docs']:,} documents · K = {data['k']} clusters · "
            f"Speed-up MLlib = {data['speedup_vs_rdd_l2']:.2f}× (vs RDD L2) / {data['speedup_vs_rdd_l1']:.2f}× (vs RDD L1)",
            fontsize=10,
            color="#52525b",
        )

        gs = fig.add_gridspec(
            2, 2, left=0.06, right=0.96, top=0.86, bottom=0.08, wspace=0.28, hspace=0.35
        )

        # ------------------------------------------------------------------ #
        # Plot 1: Courbes de Convergence WSSE par Itération (L2 vs L1 vs MLlib)
        # ------------------------------------------------------------------ #
        ax1 = fig.add_subplot(gs[0, 0])
        l2_iters = [h["iteration"] for h in rdd_l2["history"]]
        l2_wsse = [h["wsse"] for h in rdd_l2["history"]]
        l1_iters = [h["iteration"] for h in rdd_l1["history"]]
        l1_wsse = [h["wsse"] for h in rdd_l1["history"]]

        ax1.plot(
            l2_iters,
            l2_wsse,
            marker="o",
            linewidth=2.2,
            color="#10b981",
            label=f"RDD Euclidienne L2 (WSSE final = {rdd_l2['final_wsse']:.1f})",
        )
        ax1.plot(
            l1_iters,
            l1_wsse,
            marker="s",
            linewidth=2.0,
            linestyle="--",
            color="#f59e0b",
            label=f"RDD Manhattan L1 (WSSE final = {rdd_l1['final_wsse']:.1f})",
        )
        ax1.axhline(
            mllib["final_wsse"],
            color="#2563eb",
            linestyle=":",
            linewidth=1.8,
            label=f"MLlib K-Means|| Référence (WSSE = {mllib['final_wsse']:.1f})",
        )
        ax1.set_title("1. Courbes de Convergence du Coût Intra-Cluster WSSE (L1 vs L2)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Itération t (Map + reduceByKey)", fontsize=9)
        ax1.set_ylabel("Coût Intra-Cluster WSSE", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(fontsize=8, loc="upper right", frameon=True)

        # ------------------------------------------------------------------ #
        # Plot 2: Projection PCA 2D des Clusters de Documents (58D -> 2D)
        # ------------------------------------------------------------------ #
        ax2 = fig.add_subplot(gs[0, 1])
        palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16", "#ef4444"]
        for c_id in range(data["k"]):
            pts_c = [p for p in pca_pts if p["cluster"] == c_id]
            if pts_c:
                ax2.scatter(
                    [p["x"] for p in pts_c],
                    [p["y"] for p in pts_c],
                    s=18,
                    alpha=0.72,
                    color=palette[c_id % len(palette)],
                    label=f"Cluster #{c_id} (n={len(pts_c)})",
                )
        if pca_cnt:
            ax2.scatter(
                [c["x"] for c in pca_cnt],
                [c["y"] for c in pca_cnt],
                s=130,
                marker="X",
                color="#09090b",
                edgecolors="#ffffff",
                linewidths=1.2,
                zorder=5,
                label="Centroïdes Finaux (2D PCA)",
            )
        v1 = pca_var[0] if len(pca_var) > 0 else 0.0
        v2 = pca_var[1] if len(pca_var) > 1 else 0.0
        ax2.set_title("2. Projection PCA 2D des Clusters de Documents (58D -> 2D)", fontsize=11, fontweight="bold")
        ax2.set_xlabel(f"Composante Principale PC1 ({v1:.1f}% var.)", fontsize=9)
        ax2.set_ylabel(f"Composante Principale PC2 ({v2:.1f}% var.)", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.35)
        ax2.legend(fontsize=7.5, loc="best", ncol=2, frameon=True)

        # ------------------------------------------------------------------ #
        # Plot 3: Méthode du Coude (Elbow WSSE) & Silhouette Score vs K
        # ------------------------------------------------------------------ #
        ax3 = fig.add_subplot(gs[1, 0])
        ks = [e["k"] for e in elbow]
        wsses = [e["wsse"] for e in elbow]
        sils = [e["silhouette"] for e in elbow]

        ax3.plot(ks, wsses, marker="o", linewidth=2.2, color="#2563eb", label="WSSE (Elbow)")
        ax3.axvline(data["k"], color="#10b981", linestyle="--", linewidth=1.2, label=f"Coude optimal K={data['k']}")
        ax3.set_title("3. Méthode du Coude (Elbow WSSE) & Silhouette Score", fontsize=10.5, fontweight="bold")
        ax3.set_xlabel("Nombre de Clusters K", fontsize=9)
        ax3.set_ylabel("Coût WSSE", fontsize=9, color="#2563eb")
        ax3.grid(True, linestyle="--", alpha=0.35)

        ax3_twin = ax3.twinx()
        ax3_twin.plot(ks, sils, marker="D", linewidth=1.8, linestyle="-.", color="#10b981", label="Silhouette Score")
        ax3_twin.set_ylabel("Silhouette Score [-1, 1]", fontsize=9, color="#10b981")

        lines_1, labels_1 = ax3.get_legend_handles_labels()
        lines_2, labels_2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8, loc="center right", frameon=True)

        # ------------------------------------------------------------------ #
        # Plot 4: Surcoût de Sérialisation Python/RDD vs Catalyst/Tungsten MLlib
        # ------------------------------------------------------------------ #
        ax4 = fig.add_subplot(gs[1, 1])
        engines = [
            f"RDD L2 (Euclid.)\n{rdd_l2['num_iterations']} iters",
            f"RDD L1 (Manhat.)\n{rdd_l1['num_iterations']} iters",
            f"MLlib (K-Means||)\n{mllib['num_iterations']} iters",
        ]
        total_times = [
            rdd_l2["total_time_seconds"],
            rdd_l1["total_time_seconds"],
            mllib["total_time_seconds"],
        ]
        iter_times = [
            rdd_l2["avg_iter_seconds"],
            rdd_l1["avg_iter_seconds"],
            mllib["avg_iter_seconds"],
        ]
        x_pos = np.arange(len(engines))
        width = 0.36
        bars1 = ax4.bar(x_pos - width / 2, total_times, width, color=["#10b981", "#f59e0b", "#2563eb"], alpha=0.9, label="Temps Total (s)")
        bars2 = ax4.bar(x_pos + width / 2, iter_times, width, color=["#059669", "#d97706", "#1d4ed8"], alpha=0.55, label="Temps / Itération (s)")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(engines, fontsize=8.5)
        ax4.set_ylabel("Temps d'exécution (secondes)", fontsize=9)
        ax4.set_title("4. Performance : Sérialisation Python/RDD vs MLlib Tungsten", fontsize=10.5, fontweight="bold")
        ax4.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax4.legend(fontsize=8, loc="upper right", frameon=True)
        for b in bars1:
            h = b.get_height()
            ax4.annotate(
                f"{h:.2f}s",
                xy=(b.get_x() + b.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Rapport PDF généré avec succès : %s", output_pdf_path.resolve())
    return output_pdf_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout
    )
    spark = get_spark(local_cores="2")
    report = run_mini_projet_10_benchmark(spark, num_docs=1_200, k=5, max_iter=10, seed=42)
    payload = report.to_dict()

    json_path = Path("benchmark_results_58d.json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pdf_path = generate_convergence_pdf_report(payload, Path("kmeans_convergence_report.pdf"))

    print("=== MINI-PROJET 10 : RÉSULTATS DU BENCHMARK (58D) ===")
    print(f"Vecteur 58D exemple (doc #0) : {payload['sample_vector_58d'][:13]} ... {payload['sample_vector_58d'][-3:]}")
    print(f"RDD L2 (Euclidienne) : WSSE={payload['rdd_l2']['final_wsse']} | Silhouette={payload['rdd_l2']['silhouette']} | Temps={payload['rdd_l2']['total_time_seconds']}s ({payload['rdd_l2']['num_iterations']} iters)")
    print(f"RDD L1 (Manhattan)   : WSSE={payload['rdd_l1']['final_wsse']} | Silhouette={payload['rdd_l1']['silhouette']} | Temps={payload['rdd_l1']['total_time_seconds']}s ({payload['rdd_l1']['num_iterations']} iters)")
    print(f"MLlib (K-Means||)    : WSSE={payload['mllib']['final_wsse']} | Silhouette={payload['mllib']['silhouette']} | Temps={payload['mllib']['total_time_seconds']}s ({payload['mllib']['num_iterations']} iters)")
    print(f"Speed-up MLlib       : {payload['speedup_vs_rdd_l2']}x (vs RDD L2) | {payload['speedup_vs_rdd_l1']}x (vs RDD L1)")
    print(f"Livrable PDF généré  : {pdf_path.resolve()}")


if __name__ == "__main__":
    main()
