"""Execution-time and shuffle metrics extraction.

Shuffle bytes are read from Spark's documented **Monitoring REST API**
(``<spark-ui>/api/v1/applications/<app-id>/stages``) served by the driver. The
collector snapshots the set of completed stages before a code block and sums
``shuffleReadBytes`` / ``shuffleWriteBytes`` (and a few other counters) of all
stages that completed during the block.

If the REST endpoint is unreachable (e.g. UI disabled, Spark Connect,
restricted network), metrics degrade gracefully to ``None`` and only
wall-clock time is reported.

Caveat: attribution is "everything that ran on this SparkContext during the
block" — run the benchmark on an otherwise idle (dedicated) cluster.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any

from pyspark.sql import SparkSession

from kmeans_benchmark.spark_session import has_rdd_support

logger = logging.getLogger(__name__)

_STAGE_COUNTERS: tuple[str, ...] = (
    "shuffleReadBytes",
    "shuffleWriteBytes",
    "inputBytes",
    "executorRunTime",
    "executorCpuTime",
    "jvmGcTime",
    "memoryBytesSpilled",
    "diskBytesSpilled",
)


@dataclass(slots=True)
class StageMetrics:
    """Aggregated counters for the stages executed inside a measured block."""

    num_stages: int = 0
    num_tasks: int = 0
    shuffleReadBytes: int = 0
    shuffleWriteBytes: int = 0
    inputBytes: int = 0
    executorRunTime: int = 0  # ms, summed over tasks
    executorCpuTime: int = 0  # ns, summed over tasks
    jvmGcTime: int = 0  # ms
    memoryBytesSpilled: int = 0
    diskBytesSpilled: int = 0

    def as_mlflow_metrics(self) -> dict[str, float]:
        """snake_case metric dict (``shuffle_read_bytes``, ``shuffle_write_bytes``, ...)."""
        mapping = {
            "num_stages": "spark_num_stages",
            "num_tasks": "spark_num_tasks",
            "shuffleReadBytes": "shuffle_read_bytes",
            "shuffleWriteBytes": "shuffle_write_bytes",
            "inputBytes": "input_bytes",
            "executorRunTime": "executor_run_time_ms",
            "executorCpuTime": "executor_cpu_time_ns",
            "jvmGcTime": "jvm_gc_time_ms",
            "memoryBytesSpilled": "memory_bytes_spilled",
            "diskBytesSpilled": "disk_bytes_spilled",
        }
        return {mapping[k]: float(v) for k, v in asdict(self).items()}


class SparkRestClient:
    """Minimal client for Spark's Monitoring REST API (``/api/v1``)."""

    def __init__(self, spark: SparkSession, timeout_s: float = 10.0) -> None:
        self._timeout = timeout_s
        if not has_rdd_support(spark):
            self._sc = None
            self._base = ""
            self._app_id = ""
            return
        sc = spark.sparkContext
        self._sc = sc
        self._base = (sc.uiWebUrl or "").rstrip("/")
        self._app_id = sc.applicationId

    @property
    def available(self) -> bool:
        if not self._base:
            return False
        try:
            self._get(f"/applications/{self._app_id}/stages?status=complete")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spark REST API unavailable (%s); shuffle metrics disabled", exc)
            return False

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(f"{self._base}/api/v1{path}", timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def completed_stages(self) -> dict[tuple[int, int], dict[str, Any]]:
        """``{(stageId, attemptId): stage_json}`` for all completed stage attempts."""
        stages = self._get(f"/applications/{self._app_id}/stages?status=complete")
        return {(s["stageId"], s.get("attemptId", 0)): s for s in stages}

    def has_active_jobs(self) -> bool:
        return len(self._sc.statusTracker().getActiveJobsIds()) > 0


class StageMetricsCollector:
    """Context manager measuring wall-clock time + Spark stage counters.

    Example:
        >>> with StageMetricsCollector(spark) as m:
        ...     model = CustomRDDKMeans(params).fit(spark, rdd)
        >>> m.elapsed_seconds, m.metrics.shuffleWriteBytes
    """

    def __init__(self, spark: SparkSession, settle_polls: int = 20, poll_s: float = 0.25) -> None:
        self._client = SparkRestClient(spark)
        self._enabled = self._client.available
        self._settle_polls = settle_polls
        self._poll_s = poll_s
        self._before: set[tuple[int, int]] = set()
        self._t0 = 0.0
        self.elapsed_seconds: float = 0.0
        self.metrics: StageMetrics | None = None

    def __enter__(self) -> StageMetricsCollector:
        if self._enabled:
            self._before = set(self._client.completed_stages())
        self._t0 = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_seconds = time.perf_counter() - self._t0
        if exc_type is not None or not self._enabled:
            return
        try:
            self.metrics = self._collect()
        except Exception as err:  # noqa: BLE001
            logger.warning("Could not collect stage metrics: %s", err)

    def _collect(self) -> StageMetrics:
        # The status store is fed asynchronously by the listener bus: poll until
        # no job is active and the set of completed stages stops changing.
        previous: dict[tuple[int, int], dict[str, Any]] = {}
        current = self._client.completed_stages()
        for _ in range(self._settle_polls):
            if not self._client.has_active_jobs() and current.keys() == previous.keys():
                break
            time.sleep(self._poll_s)
            previous, current = current, self._client.completed_stages()

        out = StageMetrics()
        for key, stage in current.items():
            if key in self._before:
                continue
            out.num_stages += 1
            out.num_tasks += int(stage.get("numCompleteTasks", stage.get("numTasks", 0)))
            for counter in _STAGE_COUNTERS:
                setattr(out, counter, getattr(out, counter) + int(stage.get(counter, 0) or 0))
        return out

