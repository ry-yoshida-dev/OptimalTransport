from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..method import TransportMethod
from ..plan import TransportPlan
from ..types import FloatArray
from .problem import LinearTransportProblem


class LinearTransportSolver(ABC):
    """
    Base class for solvers of optimal transport problems with a cost matrix.

    Subclasses compute the coupling; this base class checks the mass balance
    the method requires and wraps the coupling with its linear cost.
    """

    def solve(self, problem: LinearTransportProblem) -> TransportPlan:
        """
        Solve the transport problem.

        Parameters
        ----------
        problem : LinearTransportProblem
            The problem to solve.

        Returns
        -------
        TransportPlan
            The transport plan and its linear cost.

        Raises
        ------
        ValueError
            If the method requires balanced marginals and the problem is not
            balanced, or if the problem is otherwise infeasible for the method.
        """
        if self.is_balanced_mass_required and not problem.is_balanced:
            raise ValueError(
                f"{self.method.value} requires source and target weights with the same total mass, got {problem.source_weights.sum()} and {problem.target_weights.sum()}"
            )
        matrix = self._compute_plan(problem)
        cost = float(np.sum(matrix * problem.cost_matrix))
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

    @property
    @abstractmethod
    def is_balanced_mass_required(self) -> bool:
        """
        Check whether the source and target must carry the same total mass.

        Returns
        -------
        bool
            True if the method enforces both marginals exactly.
        """

    @abstractmethod
    def _compute_plan(self, problem: LinearTransportProblem) -> FloatArray: ...
