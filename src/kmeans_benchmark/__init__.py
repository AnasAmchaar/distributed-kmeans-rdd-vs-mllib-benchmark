"""Distributed Document Clustering — K-Means benchmark (RDD from scratch vs PySpark MLlib)."""

from kmeans_benchmark.config import (
    BenchmarkConfig,
    DataConfig,
    FeatureConfig,
    KMeansParams,
    TrackingConfig,
)
from kmeans_benchmark.inference import DocumentClusterAssignment, DualDocumentClusterer

__all__ = [
    "BenchmarkConfig",
    "DataConfig",
    "DocumentClusterAssignment",
    "DualDocumentClusterer",
    "FeatureConfig",
    "KMeansParams",
    "TrackingConfig",
]
__version__ = "1.0.0"

