from __future__ import annotations

from dataclasses import dataclass
from typing import override

import ot

from ...method import TransportMethod
from ...types import FloatArray
from ..problem import LinearTransportProblem
from ..solver import LinearTransportSolver


@dataclass(frozen=True)
class ExactSolver(LinearTransportSolver):
    """
    Exact optimal transport solved by POT's network simplex (``ot.emd``).

    Attributes
    ----------
    max_iterations : int
        The maximum number of simplex iterations.
    n_threads : int
        The number of OpenMP threads.
    """

    max_iterations: int = 100000
    n_threads: int = 1

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.n_threads <= 0:
            raise ValueError("n_threads must be greater than 0")

    @property
    @override
    def method(self) -> TransportMethod:
        return TransportMethod.EXACT

    @property
    @override
    def is_balanced_mass_required(self) -> bool:
        return True

    @override
    def _compute_plan(self, problem: LinearTransportProblem) -> FloatArray:
        return ot.emd(
            problem.source_weights,
            problem.target_weights,
            problem.cost_matrix,
            numItermax=self.max_iterations,
            numThreads=self.n_threads,
            check_marginals=False,
        )
