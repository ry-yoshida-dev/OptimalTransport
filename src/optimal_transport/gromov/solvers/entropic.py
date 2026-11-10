from __future__ import annotations

from dataclasses import dataclass
from typing import override

import ot

from ...method import TransportMethod
from ...types import FloatArray
from ..loss import GromovLoss
from ..problem import GromovWassersteinProblem
from ..solver import GromovWassersteinSolver


@dataclass(frozen=True)
class EntropicGromovWassersteinSolver(GromovWassersteinSolver):
    """
    Entropy-regularized Gromov-Wasserstein transport solved by POT's projected
    gradient descent (``ot.gromov.entropic_gromov_wasserstein``).

    Attributes
    ----------
    regularization : float
        The entropic regularization strength; choose it relative to the scale
        of the cost matrices.
    loss : GromovLoss
        The loss comparing source and target cost entries.
    max_iterations : int
        The maximum number of outer iterations.
    tolerance : float
        The change of the plan at which iterations stop.
    """

    regularization: float
    loss: GromovLoss = GromovLoss.SQUARE
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
        return TransportMethod.ENTROPIC_GROMOV_WASSERSTEIN

    @override
    def _compute_plan_and_cost(
        self, problem: GromovWassersteinProblem
    ) -> tuple[FloatArray, float]:
        matrix, log = ot.gromov.entropic_gromov_wasserstein(
            problem.source_cost_matrix,
            problem.target_cost_matrix,
            problem.source_weights,
            problem.target_weights,
            loss_fun=self.loss.value,
            epsilon=self.regularization,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            log=True,
        )
        return matrix, float(log["gw_dist"])
