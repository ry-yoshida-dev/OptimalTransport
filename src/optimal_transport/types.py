"""Shared type aliases for optimal transport data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.float64]
type IntegerArray = NDArray[np.int64]
