from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

class GromovWassersteinLog(TypedDict):
    gw_dist: float

def gromov_wasserstein(
    C1: ArrayLike,
    C2: ArrayLike,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    loss_fun: Literal["square_loss", "kl_loss"] = "square_loss",
    symmetric: bool | None = None,
    *,
    log: Literal[True],
    max_iter: int = 10000,
    tol_rel: float = 1e-9,
    tol_abs: float = 1e-9,
) -> tuple[NDArray[np.float64], GromovWassersteinLog]: ...
def entropic_gromov_wasserstein(
    C1: ArrayLike,
    C2: ArrayLike,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
    loss_fun: Literal["square_loss", "kl_loss"] = "square_loss",
    epsilon: float = 0.1,
    symmetric: bool | None = None,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
    verbose: bool = False,
    log: Literal[True],
) -> tuple[NDArray[np.float64], GromovWassersteinLog]: ...
