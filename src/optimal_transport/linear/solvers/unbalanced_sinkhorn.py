from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import override

import ot

from ...method import TransportMethod
from ...types import FloatArray
from ..entropic_regularization import EntropicRegularization
from ..problem import LinearTransportProblem
from ..solver import LinearTransportSolver
from ..unbalanced_sinkhorn_algorithm import UnbalancedSinkhornAlgorithm


@dataclass(frozen=True)
class UnbalancedSinkhornSolver(LinearTransportSolver):
    """
    Entropy-regularized transport with KL-relaxed marginals, solved by POT's
    ``ot.unbalanced.sinkhorn_unbalanced``.

    The marginals need not carry the same total mass; mass can be created or
    destroyed at a price set by ``marginal_relaxation``.

    Attributes
    ----------
    regularization : float
        The entropic regularization strength.
    marginal_relaxation : float
        The weight of the KL penalty on deviating from each marginal; larger
        values enforce the marginals more strictly.
    regularization_type : EntropicRegularization
        The entropic regularization term.
    algorithm : UnbalancedSinkhornAlgorithm
        The iteration scheme.
    max_iterations : int
        The maximum number of iterations.
    tolerance : float
        The change in scaling vectors at which iterations stop.
    """

    regularization: float
    marginal_relaxation: float
    regularization_type: EntropicRegularization = EntropicRegularization.KL
    algorithm: UnbalancedSinkhornAlgorithm = UnbalancedSinkhornAlgorithm.SINKHORN
    max_iterations: int = 1000
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be greater than 0")
        if self.marginal_relaxation <= 0:
            raise ValueError("marginal_relaxation must be greater than 0")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")

    @property
    @override
    def method(self) -> TransportMethod:
        return TransportMethod.UNBALANCED_SINKHORN

    @property
    @override
    def is_balanced_mass_required(self) -> bool:
        return False

    @override
    def _compute_plan(self, problem: LinearTransportProblem) -> FloatArray:
        with warnings.catch_warnings():
            # POT always warns that its reference matrix c is replaced for the
            # entropy term; this API never passes c, so the warning is noise.
            warnings.filterwarnings(
                "ignore",
                message="If reg_type = entropy",
                category=UserWarning,
            )
            return ot.unbalanced.sinkhorn_unbalanced(
                problem.source_weights,
                problem.target_weights,
                problem.cost_matrix,
                reg=self.regularization,
                reg_m=self.marginal_relaxation,
                method=self.algorithm.value,
                reg_type=self.regularization_type.value,
                numItermax=self.max_iterations,
                stopThr=self.tolerance,
            )
