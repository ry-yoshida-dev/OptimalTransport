from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import CostMetric, LinearTransportProblem
from tests.helpers import shifted_points


class TestLinearTransportProblem(unittest.TestCase):
    def test_default_weights_are_uniform(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 4)))
        np.testing.assert_allclose(problem.source_weights, [0.5, 0.5])
        np.testing.assert_allclose(problem.target_weights, [0.25] * 4)
        self.assertTrue(problem.is_balanced)

    def test_arrays_are_read_only(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            problem.cost_matrix[0, 0] = 0.0
        with self.assertRaises(ValueError):
            problem.source_weights[0] = 0.0

    def test_unbalanced_weights(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)), [1.0, 1.0], [0.5, 0.5])
        self.assertFalse(problem.is_balanced)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem(np.ones(3))
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem(np.full((2, 2), np.inf))
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem(np.ones((2, 2)), source_weights=[1.0])
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem(np.ones((2, 2)), source_weights=[1.5, -0.5])
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem(np.ones((2, 2)), target_weights=[0.0, 0.0])

    def test_from_samples_uses_metric(self) -> None:
        source, target = shifted_points()
        squared = LinearTransportProblem.from_samples(source, target)
        euclidean = LinearTransportProblem.from_samples(
            source, target, metric=CostMetric.EUCLIDEAN
        )
        self.assertEqual(squared.cost_matrix.shape, (6, 8))
        np.testing.assert_allclose(squared.cost_matrix, euclidean.cost_matrix**2)

    def test_from_samples_rejects_mismatched_features(self) -> None:
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem.from_samples(np.zeros((3, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            _ = LinearTransportProblem.from_samples(np.zeros(3), np.zeros(3))


if __name__ == "__main__":
    _ = unittest.main()
