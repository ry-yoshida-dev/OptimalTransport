from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

def sinkhorn_unbalanced(
    a: ArrayLike,
    b: ArrayLike,
    M: ArrayLike,
    reg: float,
    reg_m: float,
    method: Literal[
        "sinkhorn",
        "sinkhorn_stabilized",
        "sinkhorn_translation_invariant",
        "sinkhorn_reg_scaling",
    ] = "sinkhorn",
    reg_type: Literal["kl", "entropy"] = "kl",
    numItermax: int = 1000,
    stopThr: float = 1e-6,
    verbose: bool = False,
    log: Literal[False] = False,
) -> NDArray[np.float64]: ...
