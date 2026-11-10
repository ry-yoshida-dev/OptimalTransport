"""Validation helpers that turn user input into owned, read-only float arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .types import FloatArray


def to_read_only_float_array(values: ArrayLike, name: str, ndim: int) -> FloatArray:
    """
    Copy values into a read-only float64 array and validate its shape and contents.

    Parameters
    ----------
    values : ArrayLike
        The values to copy.
    name : str
        The parameter name used in error messages.
    ndim : int
        The required number of dimensions.

    Returns
    -------
    FloatArray
        The owned, read-only array.

    Raises
    ------
    ValueError
        If the array has the wrong number of dimensions, is empty, or contains
        non-finite values.
    """
    array = np.array(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def to_marginal_weights(weights: ArrayLike | None, size: int, name: str) -> FloatArray:
    """
    Validate marginal weights, defaulting to the uniform distribution.

    Parameters
    ----------
    weights : ArrayLike | None
        Non-negative weights of each point, or None for uniform weights summing
        to 1.
    size : int
        The required number of weights.
    name : str
        The parameter name used in error messages.

    Returns
    -------
    FloatArray
        The owned, read-only weights with shape (size,).

    Raises
    ------
    ValueError
        If the weights have the wrong length, are negative, or sum to zero.
    """
    if weights is None:
        return to_read_only_float_array(np.full(size, 1.0 / size), name, ndim=1)
    array = to_read_only_float_array(weights, name, ndim=1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have {size} entries, got {array.shape[0]}")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    if not array.sum() > 0:
        raise ValueError(f"{name} must have a positive total mass")
    return array
