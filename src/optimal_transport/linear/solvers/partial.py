from __future__ import annotations

from dataclasses import dataclass
from typing import override

import ot

from ...method import TransportMethod
from ...types import FloatArray
from ..problem import LinearTransportProblem
from ..solver import LinearTransportSolver


@dataclass(frozen=True)
class PartialSolver(LinearTransportSolver):
    """
    Exact transport of a fixed amount of mass, solved by POT's
    ``ot.partial.partial_wasserstein``.

    Points that are expensive to match can be left out, which makes the plan
    robust to outliers and to points without a counterpart.

    Attributes
    ----------
    mass : float | None
        The total mass to transport, or None to transport the smaller of the
        two total masses.
    """

    mass: float | None = None

    def __post_init__(self) -> None:
        if self.mass is not None and self.mass <= 0:
            raise ValueError("mass must be greater than 0")

    @property
    @override
    def method(self) -> TransportMethod:
        return TransportMethod.PARTIAL

    @property
    @override
    def is_balanced_mass_required(self) -> bool:
        return False

    @override
    def _compute_plan(self, problem: LinearTransportProblem) -> FloatArray:
        max_mass = float(
            min(problem.source_weights.sum(), problem.target_weights.sum())
        )
        if self.mass is not None and self.mass > max_mass:
            raise ValueError(
                f"mass: {self.mass} exceeds the smaller total mass of the problem, {max_mass}"
            )
        # POT accepts only marginals within the simplex; scaling both by the larger
        # total keeps any mass unit and scales the plan back.
        scale = float(max(problem.source_weights.sum(), problem.target_weights.sum()))
        source_weights = problem.source_weights / scale
        target_weights = problem.target_weights / scale
        # Clip so that rounding in the scaling cannot push the mass above the
        # smaller scaled total when the mass equals the smaller total.
        mass = (
            None
            if self.mass is None
            else min(
                self.mass / scale,
                float(min(source_weights.sum(), target_weights.sum())),
            )
        )
        plan = ot.partial.partial_wasserstein(
            source_weights, target_weights, problem.cost_matrix, m=mass
        )
        return plan * scale
