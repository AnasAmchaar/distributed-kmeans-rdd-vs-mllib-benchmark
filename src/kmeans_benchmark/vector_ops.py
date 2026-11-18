"""Pure NumPy kernels used by the RDD K-Means map / reduce / update phases.

Kept free of any Spark dependency so the numerical core is unit-testable
without a JVM. Vectors are duck-typed: anything exposing NumPy ``indices`` and
``values`` (e.g. ``pyspark.ml.linalg.SparseVector``) is treated as sparse,
anything exposing ``toArray()`` (``DenseVector``) or a raw ``np.ndarray`` is
treated as dense.

Shapes used throughout:
    * ``D``  — feature dimensionality (vocabulary size)
    * ``K``  — number of clusters
    * ``centers`` — ``float64[K, D]`` dense, C-contiguous
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class SparseLike(Protocol):
    """Structural type of a sparse vector (``pyspark.ml.linalg.SparseVector``)."""

    indices: NDArray[np.int32]
    values: FloatArray


def _is_sparse(vector: Any) -> bool:
    """Cheap structural check (``hasattr`` is ~30x faster than a runtime
    ``isinstance(x, Protocol)`` — this runs once per document per iteration)."""
    return hasattr(vector, "indices")


def to_dense(vector: Any, dim: int) -> FloatArray:
    """Densify a sparse / dense / array-like vector to ``float64[dim]``."""
    if _is_sparse(vector):
        out = np.zeros(dim, dtype=np.float64)
        out[vector.indices] = vector.values
        return out
    if hasattr(vector, "toArray"):
        return np.asarray(vector.toArray(), dtype=np.float64)
    return np.asarray(vector, dtype=np.float64)


def squared_norm(vector: Any) -> float:
    """``||x||^2`` — O(nnz) for sparse inputs."""
    if _is_sparse(vector):
        vals = vector.values
    elif hasattr(vector, "toArray"):
        vals = np.asarray(vector.toArray(), dtype=np.float64)
    else:
        vals = np.asarray(vector, dtype=np.float64)
    return float(np.dot(vals, vals))


def dot_all(centers: FloatArray, vector: Any) -> FloatArray:
    """Dot product of one vector with **every** centroid: ``centers @ x``.

    Sparse fast path gathers only the ``nnz`` touched columns:
    ``centers[:, idx] @ vals`` -> ``O(K * nnz)`` instead of ``O(K * D)``.

    Returns:
        ``float64[K]``.
    """
    if _is_sparse(vector):
        if vector.indices.size == 0:
            return np.zeros(centers.shape[0], dtype=np.float64)
        return centers[:, vector.indices] @ vector.values
    return centers @ to_dense(vector, centers.shape[1])


def closest_center(
    vector: Any,
    centers: FloatArray,
    center_sq_norms: FloatArray,
    distance_measure: str,
) -> tuple[int, float]:
    """Assign ``vector`` to its nearest centroid.

    Euclidean uses the expansion ``||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2``;
    ``||c||^2`` is pre-computed once per iteration on the driver and broadcast,
    so the per-document cost is a single sparse gather-dot ``O(K * nnz)``.
    Cosine assumes the centroids are already unit-norm (see :func:`centroid`).

    Args:
        vector: Document vector (sparse or dense, length ``D``).
        centers: ``float64[K, D]``.
        center_sq_norms: ``float64[K]`` — ``||c_k||^2``.
        distance_measure: ``"euclidean"`` or ``"cosine"``.

    Returns:
        ``(best_index, cost)`` where cost is the squared Euclidean distance or
        the cosine distance ``1 - cos(x, c)`` to the chosen centroid.
    """
    dots = dot_all(centers, vector)
    x_sq = squared_norm(vector)
    if distance_measure == "cosine":
        x_norm = np.sqrt(x_sq) or 1.0
        c_norms = np.sqrt(center_sq_norms)
        c_norms[c_norms == 0.0] = 1.0
        dist = 1.0 - dots / (x_norm * c_norms)
    else:
        dist = np.maximum(x_sq - 2.0 * dots + center_sq_norms, 0.0)
    best = int(np.argmin(dist))
    return best, float(dist[best])


def add_into(acc: FloatArray, vector: Any) -> FloatArray:
    """In-place ``acc += x`` (sparse scatter-add when possible). Returns ``acc``.

    ``SparseVector`` indices are unique, so plain fancy-index ``+=`` is safe
    (no need for the slower ``np.add.at``).
    """
    if _is_sparse(vector):
        acc[vector.indices] += vector.values
    else:
        acc += to_dense(vector, acc.shape[0])
    return acc


def centroid(vector_sum: FloatArray, count: float, distance_measure: str) -> FloatArray:
    """New centroid from a cluster's sum vector and size.

    Euclidean: arithmetic mean. Cosine: mean re-projected onto the unit sphere
    (identical to MLlib's ``CosineDistanceMeasure.centroid``).
    """
    center = vector_sum / count
    if distance_measure == "cosine":
        norm = np.linalg.norm(center)
        if norm > 0.0:
            center = center / norm
    return center


def center_shift(old: FloatArray, new: FloatArray, distance_measure: str) -> float:
    """Movement of a centroid between two iterations.

    Euclidean: L2 distance ``||old - new||``. Cosine: ``1 - cos(old, new)``.
    Comparing this value to ``tol`` reproduces MLlib's ``isCenterConverged``.
    """
    if distance_measure == "cosine":
        denom = np.linalg.norm(old) * np.linalg.norm(new)
        return 0.0 if denom == 0.0 else float(1.0 - np.dot(old, new) / denom)
    return float(np.linalg.norm(old - new))
