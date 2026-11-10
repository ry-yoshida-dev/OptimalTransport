from __future__ import annotations

from abc import ABC, abstractmethod

from ..method import TransportMethod
from ..plan import TransportPlan
from ..types import FloatArray
from .problem import GromovWassersteinProblem


class GromovWassersteinSolver(ABC):
    """
    Base class for solvers of Gromov-Wasserstein problems.

    Gromov-Wasserstein is non-convex, so solvers return a local optimum.
    """

    def solve(self, problem: GromovWassersteinProblem) -> TransportPlan:
        """
        Solve the Gromov-Wasserstein problem.

        Parameters
        ----------
        problem : GromovWassersteinProblem
            The problem to solve.

        Returns
        -------
        TransportPlan
            The transport plan and its Gromov-Wasserstein discrepancy.
        """
        matrix, cost = self._compute_plan_and_cost(problem)
        return TransportPlan(matrix=matrix, cost=cost, method=self.method)

    @property
    @abstractmethod
    def method(self) -> TransportMethod:
        """
        Get the transport method this solver implements.

        Returns
        -------
        TransportMethod
            The transport method.
        """

    @abstractmethod
    def _compute_plan_and_cost(
        self, problem: GromovWassersteinProblem
    ) -> tuple[FloatArray, float]: ...
