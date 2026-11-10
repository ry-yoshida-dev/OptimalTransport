from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import gromov as gromov
from . import partial as partial
from . import unbalanced as unbalanced

def emd(
    a: ArrayLike,
    b: ArrayLike,
    M: ArrayLike,
    numItermax: int = 100000,
    log: Literal[False] = False,
    center_dual: bool = True,
    numThreads: int = 1,
    check_marginals: bool = True,
) -> NDArray[np.float64]: ...
def sinkhorn(
    a: ArrayLike,
    b: ArrayLike,
    M: ArrayLike,
    reg: float,
    method: Literal[
        "sinkhorn",
        "sinkhorn_log",
        "greenkhorn",
        "sinkhorn_stabilized",
        "sinkhorn_epsilon_scaling",
    ] = "sinkhorn",
    numItermax: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: Literal[False] = False,
    warn: bool = True,
) -> NDArray[np.float64]: ...
def dist(
    x1: ArrayLike,
    x2: ArrayLike | None = None,
    metric: str = "sqeuclidean",
) -> NDArray[np.float64]: ...
