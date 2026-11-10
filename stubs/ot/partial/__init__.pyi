from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

def partial_wasserstein(
    a: ArrayLike,
    b: ArrayLike,
    M: ArrayLike,
    m: float | None = None,
    nb_dummies: int = 1,
    log: Literal[False] = False,
) -> NDArray[np.float64]: ...
