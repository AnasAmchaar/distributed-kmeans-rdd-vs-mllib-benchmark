"""Spark-free unit tests for the NumPy kernels and ARI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from kmeans_benchmark import vector_ops as vo
from kmeans_benchmark.evaluation import adjusted_rand_index_from_contingency


@dataclass
class FakeSparse:
    """Duck-typed stand-in for ``pyspark.ml.linalg.SparseVector``."""

    size: int
    indices: np.ndarray
    values: np.ndarray


def sparse(dense: list[float]) -> FakeSparse:
    arr = np.asarray(dense, dtype=np.float64)
    idx = np.flatnonzero(arr).astype(np.int32)
    return FakeSparse(arr.size, idx, arr[idx])


CENTERS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.6, 0.8]])
SQ = np.einsum("ij,ij->i", CENTERS, CENTERS)


def test_dot_all_sparse_matches_dense() -> None:
    x = [0.0, 2.0, 0.0, 3.0]
    np.testing.assert_allclose(vo.dot_all(CENTERS, sparse(x)), CENTERS @ np.array(x))


def test_closest_center_euclidean() -> None:
    x = [0.1, 0.0, 0.5, 0.9]
    best, cost = vo.closest_center(sparse(x), CENTERS, SQ, "euclidean")
    expected = np.sum((CENTERS - np.array(x)) ** 2, axis=1)
    assert best == int(np.argmin(expected))
    assert cost == pytest.approx(expected.min())


def test_closest_center_cosine_and_dense_input() -> None:
    x = np.array([0.0, 5.0, 0.1, 0.0])
    best, cost = vo.closest_center(x, CENTERS, SQ, "cosine")
    assert best == 1
    assert 0.0 <= cost < 0.01


def test_add_into_and_centroid() -> None:
    acc = np.zeros(4)
    vo.add_into(acc, sparse([1.0, 0.0, 2.0, 0.0]))
    vo.add_into(acc, sparse([3.0, 0.0, 0.0, 4.0]))
    np.testing.assert_allclose(vo.centroid(acc, 2, "euclidean"), [2.0, 0.0, 1.0, 2.0])
    assert np.linalg.norm(vo.centroid(acc, 2, "cosine")) == pytest.approx(1.0)


def test_center_shift() -> None:
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert vo.center_shift(a, b, "euclidean") == pytest.approx(5.0)
    assert vo.center_shift(b, 2 * b, "cosine") == pytest.approx(0.0)


def test_empty_sparse_vector() -> None:
    empty = FakeSparse(4, np.array([], dtype=np.int32), np.array([]))
    np.testing.assert_array_equal(vo.dot_all(CENTERS, empty), np.zeros(3))


def test_ari_perfect_and_permuted() -> None:
    perfect = np.diag([10.0, 20.0, 30.0])
    assert adjusted_rand_index_from_contingency(perfect) == pytest.approx(1.0)
    assert adjusted_rand_index_from_contingency(perfect[:, ::-1]) == pytest.approx(1.0)


def test_ari_against_reference_value() -> None:
    # sklearn.metrics.adjusted_rand_score([0,0,1,1],[0,0,1,2]) == 0.5714285714
    table = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    assert adjusted_rand_index_from_contingency(table) == pytest.approx(0.5714285714)


def test_cli_parse_args_defaults() -> None:
    from kmeans_benchmark.benchmark import _parse_args

    cfg, track = _parse_args(["--in-memory", "--num-docs", "500"])
    assert track is True
    assert isinstance(cfg.data.seed, int) and cfg.data.seed == 42
    assert isinstance(cfg.kmeans.k, int) and cfg.kmeans.k == 8
    assert isinstance(cfg.features.vocab_size, int) and cfg.features.vocab_size == 16384

