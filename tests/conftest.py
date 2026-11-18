"""Shared pytest fixtures: a small local SparkSession (skipped if Spark can't start)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = (
    f"{SRC}{os.pathsep}{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else str(SRC)
)
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


@pytest.fixture(scope="session")
def spark() -> Iterator["SparkSession"]:  # noqa: F821
    pyspark_sql = pytest.importorskip("pyspark.sql")
    try:
        session = (
            pyspark_sql.SparkSession.builder.master("local[2]")
            .appName("kmeans-benchmark-tests")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
            .config("spark.python.worker.reuse", "true")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001 - e.g. unsupported Java version
        pytest.skip(f"Local Spark unavailable: {exc}")
    from kmeans_benchmark.spark_session import ship_package_to_executors

    ship_package_to_executors(session)
    yield session
    session.stop()

