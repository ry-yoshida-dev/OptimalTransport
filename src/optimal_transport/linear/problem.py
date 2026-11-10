from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ot
from numpy.typing import ArrayLike

from ..array_validation import to_marginal_weights, to_read_only_float_array
from ..cost_metric import CostMetric
from ..types import FloatArray, NumericArray


@dataclass(frozen=True, eq=False)
class LinearTransportProblem:
    """
    Immutable optimal transport problem between two weighted point sets.

    Attributes
    ----------
    cost_matrix : FloatArray
        The read-only cost of moving unit mass from source point i to target
        point j, with shape (n_source, n_target).
    source_weights : FloatArray
        The read-only source marginal with shape (n_source,).
    target_weights : FloatArray
        The read-only target marginal with shape (n_target,).
    """

    cost_matrix: FloatArray = field(init=False)
    source_weights: FloatArray = field(init=False)
    target_weights: FloatArray = field(init=False)

    def __init__(
        self,
        cost_matrix: ArrayLike,
        source_weights: ArrayLike | None = None,
        target_weights: ArrayLike | None = None,
    ) -> None:
        owned_cost_matrix = to_read_only_float_array(cost_matrix, "cost_matrix", ndim=2)
        n_source, n_target = len(owned_cost_matrix), len(owned_cost_matrix.T)
        object.__setattr__(self, "cost_matrix", owned_cost_matrix)
        object.__setattr__(
            self,
            "source_weights",
            to_marginal_weights(source_weights, n_source, "source_weights"),
        )
        object.__setattr__(
            self,
            "target_weights",
            to_marginal_weights(target_weights, n_target, "target_weights"),
        )

    @classmethod
    def from_samples(
        cls,
        source_points: NumericArray,
        target_points: NumericArray,
        metric: CostMetric = CostMetric.SQUARED_EUCLIDEAN,
        source_weights: ArrayLike | None = None,
        target_weights: ArrayLike | None = None,
    ) -> LinearTransportProblem:
        """
        Build a problem whose cost matrix is the pairwise distance between samples.

        Parameters
        ----------
        source_points : NumericArray
            The source coordinates with shape (n_source, n_features).
        target_points : NumericArray
            The target coordinates with shape (n_target, n_features).
        metric : CostMetric
            The ground metric between a source and a target point.
        source_weights : ArrayLike | None
            The source marginal, or None for uniform weights summing to 1.
        target_weights : ArrayLike | None
            The target marginal, or None for uniform weights summing to 1.

        Returns
        -------
        LinearTransportProblem
            The problem over the given samples.

        Raises
        ------
        ValueError
            If the point sets are not 2-D or differ in their number of features.
        """
        if source_points.ndim != 2 or target_points.ndim != 2:
            raise ValueError(
                f"source_points and target_points must be 2-D, got shapes {source_points.shape} and {target_points.shape}"
            )
        if source_points.shape[1] != target_points.shape[1]:
            raise ValueError(
                f"source_points and target_points must have the same number of features, got {source_points.shape[1]} and {target_points.shape[1]}"
            )
        cost_matrix = ot.dist(
            np.asarray(source_points, dtype=np.float64),
            np.asarray(target_points, dtype=np.float64),
            metric=metric.value,
        )
        return cls(cost_matrix, source_weights, target_weights)

    @property
    def is_balanced(self) -> bool:
        """
        Check whether both marginals carry the same total mass.

        Returns
        -------
        bool
            True if the source and target total masses agree to within 1e-6
            relative tolerance.
        """
        return bool(
            np.isclose(self.source_weights.sum(), self.target_weights.sum(), rtol=1e-6)
        )
