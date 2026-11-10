from __future__ import annotations

from typing import cast

import numpy as np

from optimal_transport.types import FloatArray


def shifted_points(
    n_source: int = 6, n_target: int = 8, *, seed: int = 0
) -> tuple[FloatArray, FloatArray]:
    """Return a 2-D source cloud and a target cloud shifted away from it."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((n_source, 2))
    target = rng.standard_normal((n_target, 2)) + np.array([3.0, 3.0])
    return source, target


def permuted_points(
    n_points: int = 6, *, seed: int = 0
) -> tuple[FloatArray, FloatArray, list[int]]:
    """Return a point set, a permutation of it, and the permutation indices."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((n_points, 2)) * 3.0
    order = rng.permutation(n_points)
    return source, source[order], cast(list[int], order.tolist())
