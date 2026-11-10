from __future__ import annotations

from dataclasses import dataclass
from typing import override

import ot

from ...method import TransportMethod
from ...types import FloatArray
from ..problem import LinearTransportProblem
from ..sinkhorn_algorithm import SinkhornAlgorithm
from ..solver import LinearTransportSolver


@dataclass(frozen=True)
class SinkhornSolver(LinearTransportSolver):
    """
    Entropy-regularized optimal transport solved by POT's ``ot.sinkhorn``.

    Attributes
    ----------
    regularization : float
        The entropic regularization strength; larger values give smoother plans
        and faster convergence. Choose it relative to the scale of the costs.
    algorithm : SinkhornAlgorithm
        The iteration scheme.
    max_iterations : int
        The maximum number of iterations.
    tolerance : float
        The marginal violation at which iterations stop.
    """

    regularization: float
    algorithm: SinkhornAlgorithm = SinkhornAlgorithm.SINKHORN
    max_iterations: int = 1000
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be greater than 0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    @property
    @override
    def method(self) -> TransportMethod:
        return TransportMethod.SINKHORN

    @property
    @override
    def is_balanced_mass_required(self) -> bool:
        return True

    @override
    def _compute_plan(self, problem: LinearTransportProblem) -> FloatArray:
        return ot.sinkhorn(
            problem.source_weights,
            problem.target_weights,
            problem.cost_matrix,
            reg=self.regularization,
            method=self.algorithm.value,
            numItermax=self.max_iterations,
            stopThr=self.tolerance,
        )
