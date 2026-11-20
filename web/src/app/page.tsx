"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import {
  Activity,
  ArrowUpRight,
  Award,
  CheckCircle2,
  Clock,
  Database,
  FileText,
  Layers,
  Moon,
  Play,
  Square,
  Sun,
  Target,
  TrendingDown,
  X,
  Zap,
} from "lucide-react";

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

interface IterationRecord {
  iteration: number;
  wsse: number;
  max_shift: number;
  duration_seconds: number;
  empty_clusters?: number;
}

interface RDDReport {
  distance: string;
  num_iterations: number;
  converged: boolean;
  final_wsse: number;
  silhouette: number;
  total_time_seconds: number;
  avg_iter_seconds: number;
  history: IterationRecord[];
}

interface MLlibReport {
  init_mode: string;
  num_iterations: number;
  final_wsse: number;
  silhouette: number;
  total_time_seconds: number;
  avg_iter_seconds: number;
}

interface ElbowPoint {
  k: number;
  wsse: number;
  silhouette: number;
  time_seconds?: number;
}

interface PCAPoint {
  doc_id?: number;
  x: number;
  y: number;
  cluster: number;
}

interface BenchmarkPayload {
  num_docs: number;
  dim: number;
  k: number;
  max_iter: number;
  sample_vector_58d: number[];
  speedup_vs_rdd_l2: number;
  speedup_vs_rdd_l1: number;
  rdd_l2: RDDReport;
  rdd_l1: RDDReport;
  mllib: MLlibReport;
  elbow_curve: ElbowPoint[];
  pca_explained_variance: number[];
  pca_centroids_l2: PCAPoint[];
  pca_points: PCAPoint[];
}

const CHART_PALETTE = [
  "#3b82f6",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#06b6d4",
  "#84cc16",
  "#f97316",
  "#6366f1",
];

const INITIAL_REPORT: BenchmarkPayload = {
  num_docs: 1200,
  dim: 58,
  k: 5,
  max_iter: 10,
  sample_vector_58d: [
    0.0, 0.64, 0.64, 0.0, 0.32, 0.0, 0.0, 0.0, 0.96, 0.0, 0.0, 0.64, 0.0, 3.78,
    1.32, 4.5,
  ],
  speedup_vs_rdd_l2: 45.51,
  speedup_vs_rdd_l1: 86.6,
  rdd_l2: {
    distance: "Euclidean (L2)",
    num_iterations: 9,
    converged: true,
    final_wsse: 5551.4139,
    silhouette: 0.55167,
    total_time_seconds: 309.394,
    avg_iter_seconds: 34.377,
    history: [
      { iteration: 1, wsse: 17500.2, max_shift: 4.12, duration_seconds: 34.5 },
      { iteration: 2, wsse: 6012.4, max_shift: 1.85, duration_seconds: 34.2 },
      { iteration: 3, wsse: 5551.4, max_shift: 0.42, duration_seconds: 34.4 },
      { iteration: 4, wsse: 5551.4, max_shift: 0.08, duration_seconds: 34.3 },
      { iteration: 5, wsse: 5551.4, max_shift: 0.01, duration_seconds: 34.4 },
    ],
  },
  rdd_l1: {
    distance: "Manhattan (L1)",
    num_iterations: 10,
    converged: false,
    final_wsse: 5554.5259,
    silhouette: 0.55055,
    total_time_seconds: 588.749,
    avg_iter_seconds: 58.875,
    history: [
      { iteration: 1, wsse: 17500.2, max_shift: 11.4, duration_seconds: 58.9 },
      { iteration: 2, wsse: 6920.8, max_shift: 5.12, duration_seconds: 58.7 },
      { iteration: 3, wsse: 5554.5, max_shift: 0.84, duration_seconds: 58.8 },
      { iteration: 4, wsse: 5554.5, max_shift: 0.25, duration_seconds: 58.9 },
      { iteration: 5, wsse: 5554.5, max_shift: 0.13, duration_seconds: 58.8 },
    ],
  },
  mllib: {
    init_mode: "k-means||",
    num_iterations: 2,
    final_wsse: 1971.7798,
    silhouette: 0.89078,
    total_time_seconds: 6.798,
    avg_iter_seconds: 3.399,
  },
  elbow_curve: [
    { k: 2, wsse: 13549.27, silhouette: 0.5943, time_seconds: 3.38 },
    { k: 3, wsse: 8564.85, silhouette: 0.6331, time_seconds: 2.76 },
    { k: 4, wsse: 4970.92, silhouette: 0.7367, time_seconds: 2.79 },
    { k: 5, wsse: 1971.78, silhouette: 0.8908, time_seconds: 2.16 },
    { k: 6, wsse: 1969.45, silhouette: 0.7716, time_seconds: 2.65 },
    { k: 8, wsse: 1941.56, silhouette: 0.5846, time_seconds: 4.23 },
    { k: 10, wsse: 1906.67, silhouette: 0.1991, time_seconds: 4.47 },
  ],
  pca_explained_variance: [57.7, 14.4],
  pca_centroids_l2: [
    { cluster: 0, x: 8.0, y: -1.9 },
    { cluster: 1, x: 8.2, y: -1.8 },
    { cluster: 2, x: 5.8, y: 0.7 },
    { cluster: 3, x: 2.9, y: 1.4 },
    { cluster: 4, x: 11.5, y: 0.7 },
  ],
  pca_points: [],
};

type ActiveTab = "all" | "convergence" | "pca" | "elbow" | "table";
type ConvMetric = "wsse" | "max_shift" | "duration_seconds";

interface SelectedDetail {
  id: string;
  title: string;
  badge: string;
  subtitle: string;
  metrics: { label: string; value: string }[];
  notes: string;
}

export default function MiniProjet10Page() {
  const [isDark, setIsDark] = useState<boolean>(false);
  const [report, setReport] = useState<BenchmarkPayload>(INITIAL_REPORT);
  const [activeTab, setActiveTab] = useState<ActiveTab>("all");
  const [convMetric, setConvMetric] = useState<ConvMetric>("wsse");
  const [selectedDetail, setSelectedDetail] = useState<SelectedDetail | null>(null);

  // Live Streaming state
  const [paramDocs, setParamDocs] = useState<number>(1200);
  const [paramK, setParamK] = useState<number>(5);
  const [paramIter, setParamIter] = useState<number>(10);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamPct, setStreamPct] = useState<number>(100);
  const [streamPhase, setStreamPhase] = useState<string>("TERMINÉ");
  const [streamMessage, setStreamMessage] = useState<string>(
    "Benchmark 58D chargé depuis benchmark_results_58d.json · Prêt pour l'analyse ou un run live."
  );
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [isDark]);

  useEffect(() => {
    fetch("/api/mp10")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data && data.rdd_l2) {
          setReport(data);
        }
      })
      .catch(() => {});
  }, []);

  // Start Live SSE Stream
  const handleStartLiveStream = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    setIsStreaming(true);
    setStreamPct(5);
    setStreamPhase("INITIALISATION");
    setStreamMessage(
      `Connexion au moteur PySpark en direct (N=${paramDocs} docs 58D, K=${paramK}, maxIter=${paramIter})...`
    );

    setReport((prev) => ({
      ...prev,
      num_docs: paramDocs,
      k: paramK,
      max_iter: paramIter,
      rdd_l2: { ...prev.rdd_l2, num_iterations: 0, history: [] },
      rdd_l1: { ...prev.rdd_l1, num_iterations: 0, history: [] },
    }));

    const qs = new URLSearchParams({
      num_docs: String(paramDocs),
      k: String(paramK),
      max_iter: String(paramIter),
    });
    const es = new EventSource(`/api/benchmark/stream?${qs.toString()}`);
    esRef.current = es;

    es.addEventListener("phase", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setStreamPct(d.progress);
      setStreamPhase(d.phase);
      setStreamMessage(d.message);
    });

    es.addEventListener("rdd_l2_iter", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setStreamPct(d.progress);
      setStreamPhase(`RDD L2 (${d.iteration}/${d.max_iter})`);
      setStreamMessage(
        `[RDD Euclidienne L2] Itération ${d.iteration}/${d.max_iter} → WSSE = ${Number(
          d.wsse
        ).toFixed(1)} · Déplacement = ${Number(d.max_shift).toExponential(2)} (${
          d.duration_seconds
        }s)`
      );
      setReport((prev) => ({
        ...prev,
        rdd_l2: {
          ...prev.rdd_l2,
          num_iterations: d.iteration,
          final_wsse: d.wsse,
          history: [
            ...prev.rdd_l2.history,
            {
              iteration: d.iteration,
              wsse: d.wsse,
              max_shift: d.max_shift,
              duration_seconds: d.duration_seconds,
            },
          ],
        },
      }));
    });

    es.addEventListener("rdd_l1_iter", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setStreamPct(d.progress);
      setStreamPhase(`RDD L1 (${d.iteration}/${d.max_iter})`);
      setStreamMessage(
        `[RDD Manhattan L1] Itération ${d.iteration}/${d.max_iter} → WSSE = ${Number(
          d.wsse
        ).toFixed(1)} · Déplacement = ${Number(d.max_shift).toExponential(2)} (${
          d.duration_seconds
        }s)`
      );
      setReport((prev) => ({
        ...prev,
        rdd_l1: {
          ...prev.rdd_l1,
          num_iterations: d.iteration,
          final_wsse: d.wsse,
          history: [
            ...prev.rdd_l1.history,
            {
              iteration: d.iteration,
              wsse: d.wsse,
              max_shift: d.max_shift,
              duration_seconds: d.duration_seconds,
            },
          ],
        },
      }));
    });

    es.addEventListener("complete", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      es.close();
      esRef.current = null;
      setReport(d.report);
      setIsStreaming(false);
      setStreamPct(100);
      setStreamPhase("TERMINÉ");
      setStreamMessage(
        `Benchmark 58D terminé en ${d.total_stream_seconds}s ! Rapport PDF kmeans_convergence_report.pdf mis à jour.`
      );
    });

    es.addEventListener("error", () => {
      es.close();
      esRef.current = null;
      setIsStreaming(false);
      setStreamPhase("PRÊT");
    });
  };

  const handleStopStream = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setIsStreaming(false);
    setStreamPhase("ARRÊTÉ");
    setStreamMessage("Streaming interrompu par l'utilisateur.");
  };

  // =========================================================================
  // Apache ECharts Configurations (1 Chart Per Row, Large & High-Contrast)
  // =========================================================================
  const axisTextColor = isDark ? "#94a3b8" : "#475569";
  const gridLineColor = isDark ? "#1e293b" : "#e2e8f0";
  const tooltipStyle = {
    backgroundColor: isDark ? "rgba(15, 23, 42, 0.96)" : "rgba(255, 255, 255, 0.98)",
    borderColor: isDark ? "#334155" : "#e2e8f0",
    textStyle: { color: isDark ? "#f8fafc" : "#0f172a", fontSize: 13 },
  };

  // Chart 1: Convergence WSSE (L1 vs L2 vs MLlib)
  const convergenceOption = useMemo(() => {
    const h2 = report.rdd_l2.history || [];
    const h1 = report.rdd_l1.history || [];
    const maxLen = Math.max(h2.length, h1.length, 1);
    const xLabels = Array.from({ length: maxLen }, (_, i) => `Itération ${i + 1}`);

    const metricLabel =
      convMetric === "wsse"
        ? "Coût Intra-Cluster WSSE"
        : convMetric === "max_shift"
        ? "Déplacement Max ||Δc||"
        : "Temps par Itération (s)";

    return {
      color: ["#10b981", "#f59e0b", "#2563eb"],
      tooltip: {
        trigger: "axis",
        ...tooltipStyle,
      },
      legend: {
        top: 6,
        textStyle: { color: axisTextColor, fontSize: 13, fontWeight: 600 },
      },
      grid: { left: 72, right: 36, top: 52, bottom: 44 },
      xAxis: {
        type: "category",
        data: xLabels,
        axisLabel: { color: axisTextColor, fontSize: 12 },
        axisLine: { lineStyle: { color: gridLineColor } },
      },
      yAxis: {
        type: "value",
        name: metricLabel,
        nameTextStyle: { color: axisTextColor, fontSize: 12, padding: [0, 0, 8, 0] },
        axisLabel: { color: axisTextColor, fontSize: 12 },
        splitLine: { lineStyle: { color: gridLineColor, type: "dashed" } },
      },
      series: [
        {
          name: "RDD Euclidienne (L2)",
          type: "line",
          smooth: 0.15,
          symbolSize: 9,
          lineStyle: { width: 3.5 },
          data: h2.map((h) => Number(h[convMetric].toFixed(4))),
        },
        {
          name: "RDD Manhattan (L1)",
          type: "line",
          smooth: 0.15,
          symbol: "rect",
          symbolSize: 9,
          lineStyle: { width: 3, type: "dashed" },
          data: h1.map((h) => Number(h[convMetric].toFixed(4))),
        },
        ...(convMetric === "wsse" && report.mllib.final_wsse > 0
          ? [
              {
                name: "Référence MLlib (K-Means||)",
                type: "line",
                symbol: "none",
                lineStyle: { width: 2.5, type: "dotted", color: "#2563eb" },
                data: Array.from({ length: maxLen }, () =>
                  Number(report.mllib.final_wsse.toFixed(2))
                ),
              },
            ]
          : []),
      ],
    };
  }, [report, convMetric, isDark]);

  // Chart 2: 2D PCA Cluster Projection (58D -> 2D)
  const pcaOption = useMemo(() => {
    const pts = report.pca_points || [];
    const centroids = report.pca_centroids_l2 || [];
    const v1 = report.pca_explained_variance?.[0] ?? 57.7;
    const v2 = report.pca_explained_variance?.[1] ?? 14.4;

    const clusterSeries = Array.from({ length: report.k }, (_, cId) => {
      const cPts = pts
        .filter((p) => p.cluster === cId)
        .map((p) => [p.x, p.y, p.doc_id ?? 0]);
      return {
        name: `Cluster #${cId} (${cPts.length} docs)`,
        type: "scatter",
        symbolSize: 10,
        itemStyle: {
          color: CHART_PALETTE[cId % CHART_PALETTE.length],
          opacity: 0.82,
        },
        data: cPts,
      };
    });

    return {
      tooltip: {
        trigger: "item",
        ...tooltipStyle,
        formatter: (params: { seriesName: string; value: number[] }) => {
          if (!params.value) return "";
          return `<strong>${params.seriesName}</strong><br/>PC1 : ${params.value[0]}<br/>PC2 : ${params.value[1]}`;
        },
      },
      legend: {
        top: 6,
        textStyle: { color: axisTextColor, fontSize: 13, fontWeight: 600 },
      },
      grid: { left: 68, right: 36, top: 56, bottom: 48 },
      xAxis: {
        type: "value",
        name: `Composante Principale PC1 (${v1.toFixed(1)}% variance)`,
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: axisTextColor, fontSize: 13, fontWeight: 600 },
        axisLabel: { color: axisTextColor, fontSize: 12 },
        splitLine: { lineStyle: { color: gridLineColor, type: "dashed" } },
        scale: true,
      },
      yAxis: {
        type: "value",
        name: `PC2 (${v2.toFixed(1)}% variance)`,
        nameTextStyle: { color: axisTextColor, fontSize: 12, fontWeight: 600 },
        axisLabel: { color: axisTextColor, fontSize: 12 },
        splitLine: { lineStyle: { color: gridLineColor, type: "dashed" } },
        scale: true,
      },
      series: [
        ...clusterSeries,
        {
          name: "Centroïdes Finaux (2D PCA)",
          type: "scatter",
          symbol: "diamond",
          symbolSize: 20,
          itemStyle: {
            color: isDark ? "#ffffff" : "#09090b",
            borderColor: "#f97316",
            borderWidth: 2.5,
          },
          data: centroids.map((c) => [c.x, c.y, c.cluster]),
        },
      ],
    };
  }, [report, isDark]);

  // Chart 3: Elbow WSSE(K) & Silhouette Score(K)
  const elbowOption = useMemo(() => {
    const curve = report.elbow_curve || [];
    return {
      color: ["#2563eb", "#10b981"],
      tooltip: {
        trigger: "axis",
        ...tooltipStyle,
      },
      legend: {
        top: 6,
        textStyle: { color: axisTextColor, fontSize: 13, fontWeight: 600 },
      },
      grid: { left: 72, right: 72, top: 52, bottom: 44 },
      xAxis: {
        type: "category",
        data: curve.map((c) => `K = ${c.k}`),
        axisLabel: { color: axisTextColor, fontSize: 12, fontWeight: 600 },
        axisLine: { lineStyle: { color: gridLineColor } },
      },
      yAxis: [
        {
          type: "value",
          name: "Coût Intra-Cluster WSSE",
          nameTextStyle: { color: "#2563eb", fontSize: 12, fontWeight: 600 },
          axisLabel: { color: "#2563eb", fontSize: 12 },
          splitLine: { lineStyle: { color: gridLineColor, type: "dashed" } },
        },
        {
          type: "value",
          name: "Silhouette Score [-1, 1]",
          min: 0,
          max: 1,
          nameTextStyle: { color: "#10b981", fontSize: 12, fontWeight: 600 },
          axisLabel: { color: "#10b981", fontSize: 12 },
          splitLine: { show: false },
        },
      ],
      series: [
        {
          name: "Coût WSSE (Méthode du Coude)",
          type: "line",
          yAxisIndex: 0,
          symbolSize: 10,
          lineStyle: { width: 3.5 },
          data: curve.map((c) => c.wsse),
        },
        {
          name: "Score de Silhouette (Qualité)",
          type: "line",
          yAxisIndex: 1,
          symbol: "diamond",
          symbolSize: 11,
          lineStyle: { width: 3, type: "dashed" },
          data: curve.map((c) => c.silhouette),
        },
      ],
    };
  }, [report, isDark]);

  // Chart 4: Serialization Overhead Bar Chart
  const perfOption = useMemo(() => {
    const l2 = report.rdd_l2;
    const l1 = report.rdd_l1;
    const ml = report.mllib;

    return {
      color: ["#3b82f6", "#10b981"],
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        ...tooltipStyle,
      },
      legend: {
        top: 6,
        textStyle: { color: axisTextColor, fontSize: 13, fontWeight: 600 },
      },
      grid: { left: 72, right: 36, top: 52, bottom: 44 },
      xAxis: {
        type: "category",
        data: [
          `PySpark MLlib (K-Means||)\n(${ml.num_iterations} itérations)`,
          `RDD Manuel Euclidienne (L2)\n(${l2.num_iterations} itérations)`,
          `RDD Manuel Manhattan (L1)\n(${l1.num_iterations} itérations)`,
        ],
        axisLabel: { color: axisTextColor, fontSize: 12, fontWeight: 600 },
        axisLine: { lineStyle: { color: gridLineColor } },
      },
      yAxis: {
        type: "value",
        name: "Temps d'exécution (secondes)",
        nameTextStyle: { color: axisTextColor, fontSize: 12 },
        axisLabel: { color: axisTextColor, fontSize: 12 },
        splitLine: { lineStyle: { color: gridLineColor, type: "dashed" } },
      },
      series: [
        {
          name: "Temps Total (secondes)",
          type: "bar",
          barWidth: "28%",
          itemStyle: { borderRadius: [6, 6, 0, 0] },
          data: [
            { value: ml.total_time_seconds, itemStyle: { color: "#2563eb" } },
            { value: l2.total_time_seconds, itemStyle: { color: "#10b981" } },
            { value: l1.total_time_seconds, itemStyle: { color: "#f59e0b" } },
          ],
        },
        {
          name: "Temps Moyen / Itération (secondes)",
          type: "bar",
          barWidth: "28%",
          itemStyle: { borderRadius: [6, 6, 0, 0], opacity: 0.65 },
          data: [
            { value: ml.avg_iter_seconds, itemStyle: { color: "#60a5fa" } },
            { value: l2.avg_iter_seconds, itemStyle: { color: "#34d399" } },
            { value: l1.avg_iter_seconds, itemStyle: { color: "#fbbf24" } },
          ],
        },
      ],
    };
  }, [report, isDark]);

  // Table Rows for Primary Comparison Table (clicking opens the Investigation Side Panel)
  const engineComparisonRows = useMemo(() => {
    const l2 = report.rdd_l2;
    const l1 = report.rdd_l1;
    const ml = report.mllib;
    const maxTime = Math.max(l1.total_time_seconds, l2.total_time_seconds, 1);

    return [
      {
        id: "mllib",
        name: "PySpark MLlib KMeans API",
        file: "benchmark_kmeans.py",
        init: "Parallèle K-Means||",
        primitive: "Catalyst / Tungsten DataFrame",
        distance: "Euclidienne (L2)",
        iterations: ml.num_iterations,
        wsse: ml.final_wsse,
        silhouette: ml.silhouette,
        totalTime: ml.total_time_seconds,
        iterTime: ml.avg_iter_seconds,
        speedup: `${report.speedup_vs_rdd_l2.toFixed(1)}× vs L2`,
        timePct: Math.max((ml.total_time_seconds / maxTime) * 100, 4),
        barColor: "bg-emerald-500",
        notes:
          "Exécution native dans la JVM via Catalyst et Tungsten. L'initialisation parallèle K-Means|| sélectionne des centroïdes initiaux quasi-optimaux, permettant une convergence en seulement 2 itérations avec un Silhouette score de 0.8908.",
      },
      {
        id: "rdd_l2",
        name: "K-Means Manuel sur RDD (L2)",
        file: "custom_kmeans_rdd.py",
        init: "Échantillon Aléatoire (takeSample)",
        primitive: "RDD Map + reduceByKey + Broadcast",
        distance: "Euclidienne (L2)",
        iterations: l2.num_iterations,
        wsse: l2.final_wsse,
        silhouette: l2.silhouette,
        totalTime: l2.total_time_seconds,
        iterTime: l2.avg_iter_seconds,
        speedup: "1.0× (Base L2)",
        timePct: Math.max((l2.total_time_seconds / maxTime) * 100, 8),
        barColor: "bg-orange-400",
        notes:
          "Implémentation MapReduce from scratch sur RDDs : les centroïdes sont broadcastés à chaque itération, l'étape Map assigne chaque document 58D et calcule l'erreur quadratique, et reduceByKey agrège les sommes vectorielles et le WSSE.",
      },
      {
        id: "rdd_l1",
        name: "K-Means Manuel sur RDD (L1)",
        file: "custom_kmeans_rdd.py",
        init: "Échantillon Aléatoire (takeSample)",
        primitive: "RDD Map + reduceByKey + Broadcast",
        distance: "Manhattan (L1)",
        iterations: l1.num_iterations,
        wsse: l1.final_wsse,
        silhouette: l1.silhouette,
        totalTime: l1.total_time_seconds,
        iterTime: l1.avg_iter_seconds,
        speedup: `${(l2.total_time_seconds / Math.max(l1.total_time_seconds, 0.01)).toFixed(2)}× vs L2`,
        timePct: Math.max((l1.total_time_seconds / maxTime) * 100, 8),
        barColor: "bg-rose-500",
        notes:
          "Utilise la norme L1 (Manhattan ||x - c_k||_1) lors de l'étape Map pour l'assignation aux centroïdes, tout en évaluant le coût intra-cluster WSSE via reduceByKey pour comparer la robustesse de L1 face à L2.",
      },
    ];
  }, [report]);

  return (
    <div className="max-w-[1600px] mx-auto p-6 space-y-6">
      {/* 1. HEADER ROW */}
      <header className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-3">
            <span className="px-3 py-1 rounded-md bg-orange-600 text-white font-mono text-xs font-bold tracking-wider">
              MINI-PROJET 10
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-zinc-950 dark:text-zinc-50">
              Clustering K-Means : RDD (L1/L2) vs PySpark MLlib
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 font-mono text-xs font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Next.js · Spark 3.5.9 · 58D
            </span>
          </div>
          <p className="text-sm sm:text-base text-zinc-500 dark:text-zinc-400">
            Primitives distribuées Spark RDD (<code className="font-mono text-xs bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">reduceByKey</code>) • PySpark MLlib (<code className="font-mono text-xs bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">K-Means||</code>) • Convergence WSSE • Distances L1/L2 • Projection PCA 2D
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href="/api/report"
            target="_blank"
            rel="noreferrer"
            className="bg-white dark:bg-[#262626] hover:bg-[#f8fafc] dark:hover:bg-zinc-800 text-[#334155] dark:text-zinc-200 border border-zinc-200 dark:border-zinc-700 rounded-[8px] px-[16px] py-[10px] font-medium text-sm transition-colors shadow-sm inline-flex items-center gap-2"
          >
            <FileText className="w-4 h-4 text-orange-600" />
            <span>kmeans_convergence_report.pdf</span>
            <ArrowUpRight className="w-4 h-4 text-zinc-400" />
          </a>

          <button
            type="button"
            onClick={() => setIsDark(!isDark)}
            className="bg-white dark:bg-[#262626] hover:bg-[#f8fafc] dark:hover:bg-zinc-800 text-[#334155] dark:text-zinc-200 border border-zinc-200 dark:border-zinc-700 rounded-[8px] px-[14px] py-[10px] font-medium text-sm transition-colors shadow-sm inline-flex items-center gap-2"
          >
            {isDark ? (
              <>
                <Sun className="w-4 h-4 text-amber-400" />
                <span>Clair</span>
              </>
            ) : (
              <>
                <Moon className="w-4 h-4 text-zinc-600" />
                <span>Sombre</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* 2. LIVE STREAMING BENCHMARK RUNNER */}
      <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2.5">
              <Activity className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <h2 className="text-lg font-bold text-zinc-950 dark:text-zinc-50">
                Exécuter le Benchmark en Direct (Streaming SSE Temps Réel)
              </h2>
            </div>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Lancez le calcul PySpark depuis l&apos;interface : chaque itération <code className="font-mono text-xs">Map + reduceByKey</code> (Distances L2 &amp; L1) diffuse son coût WSSE en direct dans les graphiques ECharts.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">
                Vecteurs Documents (58D)
              </label>
              <select
                value={paramDocs}
                onChange={(e) => setParamDocs(Number(e.target.value))}
                disabled={isStreaming}
                className="bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-[8px] px-[12px] py-[8px] text-sm font-mono font-semibold text-zinc-950 dark:text-zinc-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value={600}>N = 600 docs</option>
                <option value={1200}>N = 1 200 docs</option>
                <option value={2000}>N = 2 000 docs</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">
                Clusters (K)
              </label>
              <select
                value={paramK}
                onChange={(e) => setParamK(Number(e.target.value))}
                disabled={isStreaming}
                className="bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-[8px] px-[12px] py-[8px] text-sm font-mono font-semibold text-zinc-950 dark:text-zinc-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value={3}>K = 3</option>
                <option value={5}>K = 5 (Optimal)</option>
                <option value={6}>K = 6</option>
                <option value={8}>K = 8</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-zinc-500 mb-1">
                Max Itérations
              </label>
              <select
                value={paramIter}
                onChange={(e) => setParamIter(Number(e.target.value))}
                disabled={isStreaming}
                className="bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-[8px] px-[12px] py-[8px] text-sm font-mono font-semibold text-zinc-950 dark:text-zinc-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50"
              >
                <option value={6}>6 itérations</option>
                <option value={10}>10 itérations</option>
                <option value={12}>12 itérations</option>
              </select>
            </div>

            <div className="pt-5 flex items-center gap-2">
              <button
                type="button"
                onClick={handleStartLiveStream}
                disabled={isStreaming}
                className="bg-[#059669] hover:bg-[#047857] disabled:opacity-50 text-white rounded-[8px] px-[18px] py-[9px] font-semibold text-sm transition-colors shadow-sm inline-flex items-center gap-2"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>{isStreaming ? "Streaming en cours..." : "Lancer Benchmark Live"}</span>
              </button>

              {isStreaming && (
                <button
                  type="button"
                  onClick={handleStopStream}
                  className="bg-[#e11d48] hover:bg-[#be123c] text-white rounded-[8px] px-[14px] py-[9px] font-semibold text-sm transition-colors shadow-sm inline-flex items-center gap-1.5"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Stop</span>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Stream Progress Bar */}
        <div className="pt-3 border-t border-zinc-200 dark:border-zinc-800 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <div className="flex items-center gap-2.5">
              <span className="px-2.5 py-0.5 rounded bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 font-mono text-xs font-bold">
                {streamPhase}
              </span>
              <span className="text-zinc-700 dark:text-zinc-300 font-medium">
                {streamMessage}
              </span>
            </div>
            <span className="font-mono font-bold text-blue-600 dark:text-blue-400 tabular-nums">
              {streamPct}%
            </span>
          </div>
          <div className="w-full h-2.5 rounded-full bg-zinc-100 dark:bg-zinc-800 overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-300 rounded-full"
              style={{ width: `${streamPct}%` }}
            />
          </div>
        </div>
      </section>

      {/* 3. KPI CARDS ROW (4 Cards) */}
      <section className="grid grid-cols-1 md:grid-cols-4 gap-5">
        {/* KPI 1 */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
              Speed-up MLlib vs RDD
            </span>
            <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl sm:text-4xl font-extrabold font-mono tabular-nums text-zinc-950 dark:text-zinc-50">
              {report.speedup_vs_rdd_l2.toFixed(1)}×
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              ↑ {report.speedup_vs_rdd_l1.toFixed(1)}× vs L1
            </span>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 font-mono tabular-nums">
            MLlib : {report.mllib.total_time_seconds.toFixed(2)}s · RDD L2 :{" "}
            {report.rdd_l2.total_time_seconds.toFixed(1)}s
          </p>
        </div>

        {/* KPI 2 */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
              Coût Intra-Cluster WSSE
            </span>
            <TrendingDown className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl sm:text-4xl font-extrabold font-mono tabular-nums text-zinc-950 dark:text-zinc-50">
              {report.mllib.final_wsse.toLocaleString("fr-FR", {
                maximumFractionDigits: 1,
              })}
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              K-Means||
            </span>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 font-mono tabular-nums">
            RDD L2 : {report.rdd_l2.final_wsse.toFixed(1)} · RDD L1 :{" "}
            {report.rdd_l1.final_wsse.toFixed(1)}
          </p>
        </div>

        {/* KPI 3 */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
              Score de Silhouette (K={report.k})
            </span>
            <Award className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl sm:text-4xl font-extrabold font-mono tabular-nums text-zinc-950 dark:text-zinc-50">
              {report.mllib.silhouette.toFixed(4)}
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
              Optimal K=5
            </span>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 font-mono tabular-nums">
            RDD L2 : {report.rdd_l2.silhouette.toFixed(4)} · RDD L1 :{" "}
            {report.rdd_l1.silhouette.toFixed(4)}
          </p>
        </div>

        {/* KPI 4 */}
        <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-zinc-500 dark:text-zinc-400">
              Temps Moyen / Itération
            </span>
            <Clock className="w-5 h-5 text-amber-500" />
          </div>
          <div className="flex items-baseline gap-3">
            <span className="text-3xl sm:text-4xl font-extrabold font-mono tabular-nums text-zinc-950 dark:text-zinc-50">
              {report.mllib.avg_iter_seconds.toFixed(2)}s
            </span>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-400">
              RDD : {report.rdd_l2.avg_iter_seconds.toFixed(1)}s/it
            </span>
          </div>
          <p className="text-sm text-zinc-500 dark:text-zinc-400 font-mono tabular-nums">
            RDD L1 : {report.rdd_l1.avg_iter_seconds.toFixed(1)}s/it ({report.num_docs} docs 58D)
          </p>
        </div>
      </section>

      {/* 4. UNDERLINE NAVIGATION TABS (react_framework.md Section 6) */}
      <nav className="border-b border-zinc-200 dark:border-zinc-800 flex flex-wrap gap-6 text-sm font-semibold">
        {(
          [
            ["all", "Vue Complète (1 Graphique / Ligne)"],
            ["convergence", "1. Convergence WSSE (L1 vs L2)"],
            ["pca", "2. Projection PCA 2D (58D → 2D)"],
            ["elbow", "3. Méthode du Coude & Silhouette"],
            ["table", "4. Comparatif & Livrables Ingénieur"],
          ] as [ActiveTab, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setActiveTab(key)}
            className={`pb-3.5 transition-colors border-b-2 ${
              activeTab === key
                ? "border-[#3b82f6] text-[#2563eb] dark:text-[#60a5fa]"
                : "border-transparent text-zinc-500 hover:text-[#334155] dark:hover:text-[#e2e8f0] hover:border-[#3b82f6]"
            }`}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* 5. MAIN CONTENT AREA — STRICT 1 CHART PER ROW FOR MAXIMUM READABILITY */}
      <div className="grid grid-cols-1 gap-6">
        {/* CHART 1: CONVERGENCE WSSE (L1 vs L2) */}
        {(activeTab === "all" || activeTab === "convergence") && (
          <section className="bg-white dark:bg-[#0c0c0f] rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-xl font-bold text-zinc-950 dark:text-zinc-50">
                  1. Courbes de Convergence du Coût Intra-Cluster WSSE (Distances L1 vs L2)
                </h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Évolution du coût WSSE à chaque itération <code className="font-mono text-xs">Map + reduceByKey</code> face à la référence PySpark MLlib (<code className="font-mono text-xs">K-Means||</code>).
                </p>
              </div>

              {/* Segmented Control (react_framework.md Section 5) */}
              <div className="bg-zinc-100 dark:bg-zinc-900 p-1 rounded-[8px] inline-flex text-xs font-semibold">
                {(
                  [
                    ["wsse", "Coût WSSE(t)"],
                    ["max_shift", "Déplacement ||Δc||"],
                    ["duration_seconds", "Temps / Itération (s)"],
                  ] as [ConvMetric, string][]
                ).map(([mKey, mLbl]) => (
                  <button
                    key={mKey}
                    type="button"
                    onClick={() => setConvMetric(mKey)}
                    className={`px-3.5 py-1.5 rounded-[6px] transition-all ${
                      convMetric === mKey
                        ? "bg-white dark:bg-zinc-800 text-[#2563eb] dark:text-[#60a5fa] shadow-sm"
                        : "text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200"
                    }`}
                  >
                    {mLbl}
                  </button>
                ))}
              </div>
            </div>

            <div className="w-full h-[400px]">
              <ReactECharts
                option={convergenceOption}
                style={{ height: "100%", width: "100%" }}
                notMerge={true}
              />
            </div>
          </section>
        )}

        {/* CHART 2: PROJECTION PCA 2D DES CLUSTERS (58D -> 2D) */}
        {(activeTab === "all" || activeTab === "pca") && (
          <section className="bg-white dark:bg-[#0c0c0f] rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-xl font-bold text-zinc-950 dark:text-zinc-50">
                  2. Projection PCA 2D des Clusters de Documents (58D → 2D)
                </h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Projection des vecteurs 58D sur les deux premières composantes principales via <code className="font-mono text-xs">pyspark.ml.feature.PCA(k=2)</code> avec affichage des centroïdes finaux (<code className="font-mono text-xs">◆</code>).
                </p>
              </div>
              <span className="font-mono text-xs font-semibold text-zinc-500 bg-zinc-100 dark:bg-zinc-800 rounded px-2.5 py-1">
                Variance expliquée :{" "}
                {(
                  (report.pca_explained_variance?.[0] ?? 57.7) +
                  (report.pca_explained_variance?.[1] ?? 14.4)
                ).toFixed(1)}
                % (PC1 + PC2)
              </span>
            </div>

            <div className="w-full h-[440px]">
              <ReactECharts
                option={pcaOption}
                style={{ height: "100%", width: "100%" }}
                notMerge={true}
              />
            </div>
          </section>
        )}

        {/* CHART 3: MÉTHODE DU COUDE (ELBOW) & SILHOUETTE SCORE */}
        {(activeTab === "all" || activeTab === "elbow") && (
          <section className="bg-white dark:bg-[#0c0c0f] rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm p-6 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h3 className="text-xl font-bold text-zinc-950 dark:text-zinc-50">
                  3. Méthode du Coude (Elbow WSSE) &amp; Score de Silhouette (K ∈ &#123;2..10&#125;)
                </h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Identification automatique du nombre optimal de clusters <code className="font-mono text-xs">K = 5</code> par inflexion du WSSE et maximum du Silhouette Score (<code className="font-mono text-xs">0.8908</code>).
                </p>
              </div>
              <span className="px-3 py-1 rounded-md bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 font-mono text-xs font-bold">
                Coude Optimal : K = 5
              </span>
            </div>

            <div className="w-full h-[400px]">
              <ReactECharts
                option={elbowOption}
                style={{ height: "100%", width: "100%" }}
                notMerge={true}
              />
            </div>
          </section>
        )}

        {/* CHART 4: SURCOÛT DE SÉRIALISATION PYTHON/RDD VS MLLIB */}
        {(activeTab === "all" || activeTab === "table") && (
          <section className="bg-white dark:bg-[#0c0c0f] rounded-xl border border-zinc-200 dark:border-zinc-800 shadow-sm p-6 space-y-4">
            <div>
              <h3 className="text-xl font-bold text-zinc-950 dark:text-zinc-50">
                4. Convergence &amp; Performance Comparative : Surcoût Sérialisation Python/RDD vs MLlib DataFrame
              </h3>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">
                Mise en évidence du coût de sérialisation Py4J/Pickle par itération sur les RDDs face aux optimisations Catalyst / Tungsten de PySpark MLlib.
              </p>
            </div>

            <div className="w-full h-[380px]">
              <ReactECharts
                option={perfOption}
                style={{ height: "100%", width: "100%" }}
                notMerge={true}
              />
            </div>
          </section>
        )}

        {/* 6. MAIN CONTENT SPLIT: PRIMARY DATA TABLE + SLIDE-IN INVESTIGATION SIDE PANEL (react_framework.md Section 2 & 3) */}
        <section className="flex flex-col lg:flex-row gap-6 items-start">
          {/* Primary Data Table (w-full when unselected, shrinks to w-2/3 when row is clicked) */}
          <div
            className={`${
              selectedDetail ? "w-full lg:w-2/3" : "w-full"
            } transition-all duration-300 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 rounded-[12px] overflow-hidden shadow-sm`}
          >
            <div className="p-5 border-b border-zinc-200 dark:border-zinc-800 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-lg font-bold text-zinc-950 dark:text-zinc-50">
                  5. Tableau Comparatif des Implémentations &amp; Livrables (Cliquez sur une ligne pour inspecter)
                </h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">
                  Comparaison directe sur les vecteurs de documents 58D : cliquez sur un moteur pour ouvrir le panneau d&apos;investigation.
                </p>
              </div>
              <span className="font-mono text-xs text-zinc-500 bg-zinc-100 dark:bg-zinc-800 rounded px-2.5 py-1">
                Vecteurs 58D · N = {report.num_docs}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-sm">
                <thead className="sticky top-0 bg-zinc-100 dark:bg-zinc-900 z-10 text-zinc-500 dark:text-zinc-400 font-semibold">
                  <tr>
                    <th className="py-3.5 px-4">Moteur / Implémentation</th>
                    <th className="py-3.5 px-4">Livrable</th>
                    <th className="py-3.5 px-4">Distance &amp; Init</th>
                    <th className="py-3.5 px-4 text-right">Itérations</th>
                    <th className="py-3.5 px-4 text-right">WSSE Final</th>
                    <th className="py-3.5 px-4 text-right">Silhouette</th>
                    <th className="py-3.5 px-4">Temps &amp; Coût Relatif</th>
                  </tr>
                </thead>
                <tbody>
                  {engineComparisonRows.map((row) => {
                    const isSelected = selectedDetail?.id === row.id;
                    return (
                      <tr
                        key={row.id}
                        onClick={() =>
                          setSelectedDetail({
                            id: row.id,
                            title: row.name,
                            badge: row.file,
                            subtitle: `${row.primitive} · ${row.init}`,
                            metrics: [
                              { label: "Distance", value: row.distance },
                              { label: "Itérations", value: String(row.iterations) },
                              { label: "Coût Final WSSE", value: row.wsse.toFixed(2) },
                              { label: "Score de Silhouette", value: row.silhouette.toFixed(4) },
                              { label: "Temps Total", value: `${row.totalTime.toFixed(2)} s` },
                              { label: "Temps / Itération", value: `${row.iterTime.toFixed(2)} s` },
                            ],
                            notes: row.notes,
                          })
                        }
                        className={`border-b border-zinc-100 dark:border-zinc-800 cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-blue-50 dark:bg-blue-900/40 hover:bg-blue-100 dark:hover:bg-blue-900/60"
                            : "text-zinc-900 dark:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                        }`}
                      >
                        <td className="py-4 px-4 font-bold">{row.name}</td>
                        <td className="py-4 px-4">
                          <span className="font-mono text-xs text-zinc-600 dark:text-zinc-300 bg-zinc-100 dark:bg-zinc-800 rounded px-2 py-1">
                            {row.file}
                          </span>
                        </td>
                        <td className="py-4 px-4 text-zinc-600 dark:text-zinc-400">
                          {row.distance} · <span className="text-xs">{row.init}</span>
                        </td>
                        <td className="py-4 px-4 text-right font-mono tabular-nums font-semibold">
                          {row.iterations}
                        </td>
                        <td className="py-4 px-4 text-right font-mono tabular-nums font-semibold">
                          {row.wsse.toFixed(1)}
                        </td>
                        <td className="py-4 px-4 text-right font-mono tabular-nums font-bold text-emerald-600 dark:text-emerald-400">
                          {row.silhouette.toFixed(4)}
                        </td>
                        <td className="py-4 px-4 w-52">
                          <div className="flex items-center justify-between text-xs font-mono mb-1">
                            <span className="font-bold">{row.totalTime.toFixed(2)}s</span>
                            <span className="text-zinc-500">{row.speedup}</span>
                          </div>
                          <div className="w-full h-2 rounded-sm bg-zinc-200 dark:bg-zinc-700 overflow-hidden">
                            <div
                              className={`h-full ${row.barColor} rounded-sm`}
                              style={{ width: `${row.timePct}%` }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Conditional Investigation / Details Side Panel (w-1/3) */}
          {selectedDetail && (
            <aside className="w-full lg:w-1/3 transition-all duration-300 bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-md space-y-5">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <span className="font-mono text-xs text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-950/50 rounded px-2 py-0.5">
                    {selectedDetail.badge}
                  </span>
                  <h4 className="text-lg font-bold text-zinc-950 dark:text-zinc-50 mt-1">
                    {selectedDetail.title}
                  </h4>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">
                    {selectedDetail.subtitle}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedDetail(null)}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-2.5">
                {selectedDetail.metrics.map((m) => (
                  <div
                    key={m.label}
                    className="bg-zinc-100 dark:bg-zinc-800/90 border border-zinc-200 dark:border-zinc-700 rounded-md p-3"
                  >
                    <div className="text-xs text-zinc-500 dark:text-zinc-400">{m.label}</div>
                    <div className="text-sm font-bold font-mono text-zinc-950 dark:text-zinc-50 mt-0.5 tabular-nums">
                      {m.value}
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-3.5 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/30 text-blue-900 dark:text-blue-200 text-xs leading-relaxed">
                {selectedDetail.notes}
              </div>

              <div className="flex items-center gap-2 pt-1">
                <a
                  href="/api/report"
                  target="_blank"
                  rel="noreferrer"
                  className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-center rounded-[8px] px-4 py-2 text-xs font-semibold transition"
                >
                  Ouvrir Rapport PDF
                </a>
                <button
                  type="button"
                  onClick={() => setSelectedDetail(null)}
                  className="bg-zinc-200 dark:bg-zinc-800 hover:bg-zinc-300 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 rounded-[8px] px-4 py-2 text-xs font-semibold transition"
                >
                  Fermer
                </button>
              </div>
            </aside>
          )}
        </section>

        {/* 7. SECTION 3 DU DESCRIPTIF : REPRÉSENTATION VECTORIELLE DES DOCUMENTS (58D) */}
        <section className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-6 shadow-sm space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2.5">
              <Database className="w-5 h-5 text-orange-600" />
              <h3 className="text-base font-bold text-zinc-950 dark:text-zinc-50">
                Section 3 du Descriptif — Représentation Vectorielle des Documents (58D)
              </h3>
            </div>
            <span className="font-mono text-xs text-zinc-500 bg-zinc-100 dark:bg-zinc-800 rounded px-2 py-0.5">
              48 fréquences de mots + 6 fréquences de caractères + 4 statistiques de séquences = 58D
            </span>
          </div>
          <pre className="p-4 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 font-mono text-xs sm:text-sm text-zinc-800 dark:text-zinc-200 overflow-x-auto">
            {`[${report.sample_vector_58d.slice(0, 13).map((v) => Number(v).toFixed(2)).join(", ")}, ..., ${report.sample_vector_58d.slice(-3).map((v) => Number(v).toFixed(2)).join(", ")}]`}
          </pre>
        </section>
      </div>
    </div>
  );
}
