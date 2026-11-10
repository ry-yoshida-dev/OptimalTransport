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
class ExactGromovWassersteinSolver(GromovWassersteinSolver):
    """
    Gromov-Wasserstein transport solved by POT's conditional gradient
    (``ot.gromov.gromov_wasserstein``).

    Attributes
    ----------
    loss : GromovLoss
        The loss comparing source and target cost entries.
    max_iterations : int
        The maximum number of conditional gradient iterations.
    relative_tolerance : float
        The relative change of the objective at which iterations stop.
    absolute_tolerance : float
        The absolute change of the objective at which iterations stop.
    """

    loss: GromovLoss = GromovLoss.SQUARE
    max_iterations: int = 10000
    relative_tolerance: float = 1e-9
    absolute_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.absolute_tolerance < 0:
            raise ValueError("absolute_tolerance must be non-negative")

    @property
    @override
    def method(self) -> TransportMethod:
        return TransportMethod.GROMOV_WASSERSTEIN

    @override
    def _compute_plan_and_cost(
        self, problem: GromovWassersteinProblem
    ) -> tuple[FloatArray, float]:
        matrix, log = ot.gromov.gromov_wasserstein(
            problem.source_cost_matrix,
            problem.target_cost_matrix,
            problem.source_weights,
            problem.target_weights,
            loss_fun=self.loss.value,
            log=True,
            max_iter=self.max_iterations,
            tol_rel=self.relative_tolerance,
            tol_abs=self.absolute_tolerance,
        )
        return matrix, float(log["gw_dist"])
