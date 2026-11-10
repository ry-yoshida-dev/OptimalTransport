from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import (
    ExactSolver,
    LinearTransportProblem,
    SinkhornAlgorithm,
    SinkhornSolver,
    TransportMethod,
)
from tests.helpers import shifted_points


class TestSinkhornSolver(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = SinkhornSolver(regularization=0.0)
        with self.assertRaises(ValueError):
            _ = SinkhornSolver(regularization=1.0, max_iterations=0)
        with self.assertRaises(ValueError):
            _ = SinkhornSolver(regularization=1.0, tolerance=-1.0)

    def test_plan_respects_marginals(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        plan = SinkhornSolver(regularization=1.0).solve(problem)
        np.testing.assert_allclose(
            plan.source_marginal, problem.source_weights, atol=1e-6
        )
        np.testing.assert_allclose(
            plan.target_marginal, problem.target_weights, atol=1e-6
        )
        self.assertEqual(plan.method, TransportMethod.SINKHORN)

    def test_small_regularization_approaches_exact_cost(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        exact = ExactSolver().solve(problem)
        entropic = SinkhornSolver(
            regularization=0.1,
            algorithm=SinkhornAlgorithm.LOG_DOMAIN,
            max_iterations=10000,
        ).solve(problem)
        self.assertAlmostEqual(entropic.cost, exact.cost, delta=0.01 * exact.cost)

    def test_unbalanced_problem_raises(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)), [1.0, 1.0], [0.5, 0.5])
        with self.assertRaises(ValueError):
            _ = SinkhornSolver(regularization=1.0).solve(problem)


if __name__ == "__main__":
    _ = unittest.main()
