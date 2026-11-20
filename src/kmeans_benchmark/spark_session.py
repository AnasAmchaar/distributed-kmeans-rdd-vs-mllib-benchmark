"""SparkSession acquisition that works both on Databricks and locally."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from typing import Any

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession


def is_databricks() -> bool:
    """Return ``True`` when running inside a Databricks Runtime."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def has_rdd_support(spark: SparkSession) -> bool:
    """Return ``True`` when the session exposes a JVM ``SparkContext`` and RDDs.

    Returns ``False`` on Databricks Serverless / Spark Connect sessions.
    """
    try:
        _ = spark.sparkContext
        return True
    except Exception:  # noqa: BLE001
        return False


def safe_persist(
    df: DataFrame, storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK
) -> DataFrame:
    """Persist ``df`` when supported; degrade to a no-op on Databricks Serverless."""
    try:
        return df.persist(storage_level)
    except Exception:  # noqa: BLE001
        return df


def safe_unpersist(df: DataFrame) -> None:
    """Unpersist ``df`` when supported; no-op on Databricks Serverless."""
    try:
        df.unpersist()
    except Exception:  # noqa: BLE001
        pass


def _ensure_local_windows_hadoop() -> None:
    """Configure a fallback HADOOP_HOME and JDK 17 on Windows when present."""
    if os.name != "nt":
        return
    # Spark 3.5 requires Java 11/17 (Java 24+ removed Subject.getSubject).
    java_home = os.environ.get("JAVA_HOME", "")
    if not java_home or "jdk-25" in java_home.lower() or "jdk-24" in java_home.lower():
        scratch_jdk = next(
            Path.home().glob(".gemini/antigravity/brain/*/scratch/jdk17/jdk-*"), None
        )
        for jdk_candidate in (
            Path(os.environ["LOCAL_JAVA_HOME"]) if "LOCAL_JAVA_HOME" in os.environ else None,
            scratch_jdk,
        ):
            if jdk_candidate and (jdk_candidate / "bin" / "java.exe").exists():
                os.environ["JAVA_HOME"] = str(jdk_candidate)
                os.environ["PATH"] = f"{jdk_candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
                break

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home and (Path(hadoop_home) / "bin" / "winutils.exe").exists():
        return
    scratch_hadoop = next(
        Path.home().glob(".gemini/antigravity/brain/*/scratch/hadoop"), None
    )
    for candidate in (
        Path(os.environ["LOCAL_HADOOP_HOME"]) if "LOCAL_HADOOP_HOME" in os.environ else None,
        Path("C:/hadoop"),
        Path.home() / ".hadoop",
        scratch_hadoop,
    ):
        if candidate and (candidate / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(candidate)
            os.environ["PATH"] = f"{candidate / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def get_spark(app_name: str = "kmeans-rdd-vs-mllib", local_cores: str = "4") -> SparkSession:
    """Return the active SparkSession (Databricks) or build a local one.

    On Databricks the session is pre-created and *must not* be reconfigured;
    ``getOrCreate`` simply returns it. Locally we build a session sized for a
    laptop with Arrow and adaptive execution enabled.

    Note:
        The RDD implementation needs ``spark.sparkContext``. On Databricks this
        requires a cluster in **Dedicated / Single-user** access mode — RDD APIs
        are not available on Serverless or Standard (shared) access mode.
    """
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    if is_databricks():
        return SparkSession.builder.getOrCreate()

    _ensure_local_windows_hadoop()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    src_root = str(Path(__file__).resolve().parents[1])
    existing_py_path = os.environ.get("PYTHONPATH", "")
    if src_root not in existing_py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{src_root}{os.pathsep}{existing_py_path}" if existing_py_path else src_root
        )

    return (
        SparkSession.builder.appName(app_name)
        .master(f"local[{local_cores}]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
        .config("spark.python.worker.reuse", "true")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def ship_package_to_executors(spark: SparkSession) -> str | None:
    """Zip the ``kmeans_benchmark`` package and register it with ``addPyFile``.

    Executor Python workers must be able to *import* the module-level map /
    reduce functions (cloudpickle serialises them by reference). When the
    package is only on the driver's ``sys.path`` (Databricks Git folder,
    local ``src/`` checkout) executors would fail with ``ModuleNotFoundError``.
    The zip name embeds a content hash, so repeated calls are idempotent.

    In ``local[...]`` mode on Windows without ``winutils.exe``, ``addPyFile``
    triggers Hadoop ``FileUtil.chmod``; in that case workers already inherit
    ``PYTHONPATH`` on the local host so we fall back cleanly.

    Returns:
        Path of the zip that was registered, or ``None`` if skipped in local mode.
    """
    pkg_dir = Path(__file__).resolve().parent
    src_root = str(pkg_dir.parent)
    existing_py_path = os.environ.get("PYTHONPATH", "")
    if src_root not in existing_py_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{src_root}{os.pathsep}{existing_py_path}" if existing_py_path else src_root
        )

    if not has_rdd_support(spark) or spark.sparkContext.master.startswith("local"):
        return None

    files = sorted(p for p in pkg_dir.rglob("*.py"))
    digest = hashlib.sha1(b"".join(p.read_bytes() for p in files)).hexdigest()[:12]
    zip_path = Path(tempfile.gettempdir()) / f"kmeans_benchmark_{digest}.zip"
    if not zip_path.exists():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                zf.write(file, (Path(pkg_dir.name) / file.relative_to(pkg_dir)).as_posix())
    spark.sparkContext.addPyFile(str(zip_path))
    return str(zip_path)


class _BroadcastShim:
    """In-process Broadcast stand-in for Spark Connect / Serverless sessions."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def destroy(self) -> None:
        self.value = None


class _AccumulatorShim:
    """In-process Accumulator stand-in for Spark Connect / Serverless sessions."""

    def __init__(self, initial: float = 0.0) -> None:
        self.value = initial

    def add(self, term: float) -> None:
        self.value += term


class SparkContextShim:
    """SparkContext stand-in when running on a Spark Connect / Serverless cluster."""

    defaultParallelism: int = max(2, os.cpu_count() or 4)

    def broadcast(self, value: Any) -> Any:
        return _BroadcastShim(value)

    def accumulator(self, initial: float = 0.0) -> Any:
        return _AccumulatorShim(initial)


def get_spark_context(spark: SparkSession) -> Any:
    """Return ``spark.sparkContext`` or a :class:`SparkContextShim` on Serverless."""
    if has_rdd_support(spark):
        return spark.sparkContext
    return SparkContextShim()
