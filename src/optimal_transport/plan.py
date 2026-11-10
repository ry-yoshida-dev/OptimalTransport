from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .array_validation import to_read_only_float_array
from .method import TransportMethod
from .types import FloatArray, IntegerArray, NumericArray


@dataclass(frozen=True, eq=False)
class TransportPlan:
    """
    Immutable transport plan produced by an optimal transport solver.

    Attributes
    ----------
    matrix : FloatArray
        The read-only coupling with shape (n_source, n_target); entry (i, j) is
        the mass moved from source point i to target point j.
    cost : float
        The transport cost. For linear transport this is the linear cost
        ``sum(matrix * cost_matrix)`` without any regularization term; for
        Gromov-Wasserstein transport it is the Gromov-Wasserstein discrepancy.
    method : TransportMethod
        The method that produced the plan.
    """

    UNASSIGNED: ClassVar[int] = -1
    """Assignment value of source points that send no mass."""

    matrix: FloatArray = field(init=False)
    cost: float = field(init=False)
    method: TransportMethod = field(init=False)

    def __init__(self, matrix: ArrayLike, cost: float, method: TransportMethod) -> None:
        owned_matrix = to_read_only_float_array(matrix, "matrix", ndim=2)
        if np.any(owned_matrix < 0):
            raise ValueError("matrix must be non-negative")
        object.__setattr__(self, "matrix", owned_matrix)
        object.__setattr__(self, "cost", float(cost))
        object.__setattr__(self, "method", method)

    @property
    def source_marginal(self) -> FloatArray:
        """
        Get the mass sent from each source point.

        Returns
        -------
        FloatArray
            The row sums with shape (n_source,).
        """
        return self.matrix.sum(axis=1)

    @property
    def target_marginal(self) -> FloatArray:
        """
        Get the mass received by each target point.

        Returns
        -------
        FloatArray
            The column sums with shape (n_target,).
        """
        return self.matrix.sum(axis=0)

    @property
    def total_mass(self) -> float:
        """
        Get the total transported mass.

        Returns
        -------
        float
            The sum of all entries of the plan.
        """
        return float(self.matrix.sum())

    def get_assignments(self) -> IntegerArray:
        """
        Get the target point receiving the most mass from each source point.

        Returns
        -------
        IntegerArray
            The target index per source point with shape (n_source,); source
            points that send no mass get ``UNASSIGNED``.
        """
        assignments = np.argmax(self.matrix, axis=1).astype(np.int64)
        assignments[self.source_marginal <= 0] = self.UNASSIGNED
        return assignments

    def barycentric_projection(self, target_points: NumericArray) -> FloatArray:
        """
        Map each source point to the mass-weighted mean of its target points.

        Parameters
        ----------
        target_points : NumericArray
            The target coordinates with shape (n_target, n_features).

        Returns
        -------
        FloatArray
            The projected source points with shape (n_source, n_features).

        Raises
        ------
        ValueError
            If target_points does not have one row per target point, or if a
            source point sends no mass and therefore has no projection.
        """
        if target_points.ndim != 2 or target_points.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"target_points must have shape ({self.matrix.shape[1]}, n_features), got {target_points.shape}"
            )
        source_marginal = self.source_marginal
        unassigned = np.flatnonzero(source_marginal <= 0)
        if unassigned.size > 0:
            raise ValueError(
                f"source points {unassigned.tolist()} send no mass and have no barycentric projection"
            )
        projected: FloatArray = (self.matrix @ target_points) / source_marginal[:, None]
        return projected
