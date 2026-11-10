from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import (
    EntropicGromovWassersteinSolver,
    ExactGromovWassersteinSolver,
    GromovLoss,
    GromovWassersteinProblem,
    TransportMethod,
)
from tests.helpers import permuted_points


class TestGromovWassersteinProblem(unittest.TestCase):
    def test_from_samples_allows_different_dimensions(self) -> None:
        problem = GromovWassersteinProblem.from_samples(
            np.zeros((3, 2)), np.ones((4, 5))
        )
        self.assertEqual(problem.source_cost_matrix.shape, (3, 3))
        self.assertEqual(problem.target_cost_matrix.shape, (4, 4))

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = GromovWassersteinProblem(np.ones((2, 3)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            _ = GromovWassersteinProblem(
                np.ones((2, 2)), np.ones((2, 2)), target_weights=[1.0, 1.0]
            )
        with self.assertRaises(ValueError):
            _ = GromovWassersteinProblem.from_samples(np.zeros(3), np.zeros((3, 2)))


class TestGromovWassersteinSolvers(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = ExactGromovWassersteinSolver(max_iterations=0)
        with self.assertRaises(ValueError):
            _ = ExactGromovWassersteinSolver(relative_tolerance=-1.0)
        with self.assertRaises(ValueError):
            _ = ExactGromovWassersteinSolver(absolute_tolerance=-1.0)
        with self.assertRaises(ValueError):
            _ = EntropicGromovWassersteinSolver(regularization=0.0)
        with self.assertRaises(ValueError):
            _ = EntropicGromovWassersteinSolver(regularization=0.1, max_iterations=0)
        with self.assertRaises(ValueError):
            _ = EntropicGromovWassersteinSolver(regularization=0.1, tolerance=-1.0)

    def test_isometric_spaces_have_zero_discrepancy(self) -> None:
        source, target, _ = permuted_points()
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        problem = GromovWassersteinProblem.from_samples(source, target @ rotation)
        plan = ExactGromovWassersteinSolver().solve(problem)
        self.assertEqual(plan.method, TransportMethod.GROMOV_WASSERSTEIN)
        self.assertAlmostEqual(plan.cost, 0.0, places=6)

    def test_plans_respect_marginals(self) -> None:
        source, target, _ = permuted_points()
        problem = GromovWassersteinProblem.from_samples(source, target)
        for solver in (
            ExactGromovWassersteinSolver(loss=GromovLoss.SQUARE),
            EntropicGromovWassersteinSolver(regularization=5.0),
        ):
            plan = solver.solve(problem)
            np.testing.assert_allclose(
                plan.source_marginal, problem.source_weights, atol=1e-6
            )
            np.testing.assert_allclose(
                plan.target_marginal, problem.target_weights, atol=1e-6
            )
            self.assertGreaterEqual(plan.cost, -1e-9)


if __name__ == "__main__":
    _ = unittest.main()
