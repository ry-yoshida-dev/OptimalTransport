from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import ot
from numpy.typing import ArrayLike

from ..array_validation import to_marginal_weights, to_read_only_float_array
from ..cost_metric import CostMetric
from ..types import FloatArray, NumericArray


@dataclass(frozen=True, eq=False)
class GromovWassersteinProblem:
    """
    Immutable Gromov-Wasserstein problem between two metric-measure spaces.

    The two spaces are compared only through their internal cost matrices, so
    they may live in different feature spaces.

    Attributes
    ----------
    source_cost_matrix : FloatArray
        The read-only cost between source points, with shape
        (n_source, n_source).
    target_cost_matrix : FloatArray
        The read-only cost between target points, with shape
        (n_target, n_target).
    source_weights : FloatArray
        The read-only source marginal with shape (n_source,).
    target_weights : FloatArray
        The read-only target marginal with shape (n_target,).
    """

    source_cost_matrix: FloatArray = field(init=False)
    target_cost_matrix: FloatArray = field(init=False)
    source_weights: FloatArray = field(init=False)
    target_weights: FloatArray = field(init=False)

    def __init__(
        self,
        source_cost_matrix: ArrayLike,
        target_cost_matrix: ArrayLike,
        source_weights: ArrayLike | None = None,
        target_weights: ArrayLike | None = None,
    ) -> None:
        owned_source_cost = self._to_square_matrix(
            source_cost_matrix, "source_cost_matrix"
        )
        owned_target_cost = self._to_square_matrix(
            target_cost_matrix, "target_cost_matrix"
        )
        owned_source_weights = to_marginal_weights(
            source_weights, len(owned_source_cost), "source_weights"
        )
        owned_target_weights = to_marginal_weights(
            target_weights, len(owned_target_cost), "target_weights"
        )
        if not np.isclose(
            owned_source_weights.sum(), owned_target_weights.sum(), rtol=1e-6
        ):
            raise ValueError(
                f"source_weights and target_weights must have the same total mass, got {owned_source_weights.sum()} and {owned_target_weights.sum()}"
            )
        object.__setattr__(self, "source_cost_matrix", owned_source_cost)
        object.__setattr__(self, "target_cost_matrix", owned_target_cost)
        object.__setattr__(self, "source_weights", owned_source_weights)
        object.__setattr__(self, "target_weights", owned_target_weights)

    @classmethod
    def from_samples(
        cls,
        source_points: NumericArray,
        target_points: NumericArray,
        metric: CostMetric = CostMetric.EUCLIDEAN,
        source_weights: ArrayLike | None = None,
        target_weights: ArrayLike | None = None,
    ) -> GromovWassersteinProblem:
        """
        Build a problem from the pairwise distances within each point set.

        Parameters
        ----------
        source_points : NumericArray
            The source coordinates with shape (n_source, n_source_features).
        target_points : NumericArray
            The target coordinates with shape (n_target, n_target_features);
            the number of features may differ from the source.
        metric : CostMetric
            The metric between two points of the same set.
        source_weights : ArrayLike | None
            The source marginal, or None for uniform weights summing to 1.
        target_weights : ArrayLike | None
            The target marginal, or None for uniform weights summing to 1.

        Returns
        -------
        GromovWassersteinProblem
            The problem over the given samples.

        Raises
        ------
        ValueError
            If either point set is not 2-D.
        """
        if source_points.ndim != 2 or target_points.ndim != 2:
            raise ValueError(
                f"source_points and target_points must be 2-D, got shapes {source_points.shape} and {target_points.shape}"
            )
        source = np.asarray(source_points, dtype=np.float64)
        target = np.asarray(target_points, dtype=np.float64)
        return cls(
            ot.dist(source, source, metric=metric.value),
            ot.dist(target, target, metric=metric.value),
            source_weights,
            target_weights,
        )

    @staticmethod
    def _to_square_matrix(values: ArrayLike, name: str) -> FloatArray:
        matrix = to_read_only_float_array(values, name, ndim=2)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square, got shape {matrix.shape}")
        return matrix
