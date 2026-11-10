from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import ExactSolver, LinearTransportProblem, TransportMethod
from tests.helpers import permuted_points, shifted_points


class TestExactSolver(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = ExactSolver(max_iterations=0)
        with self.assertRaises(ValueError):
            _ = ExactSolver(n_threads=0)

    def test_recovers_permutation(self) -> None:
        source, target, permutation = permuted_points()
        plan = ExactSolver().solve(LinearTransportProblem.from_samples(source, target))
        expected = np.argsort(permutation)
        self.assertEqual(plan.get_assignments().tolist(), expected.tolist())
        self.assertAlmostEqual(plan.cost, 0.0)
        self.assertEqual(plan.method, TransportMethod.EXACT)

    def test_plan_respects_marginals(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        plan = ExactSolver().solve(problem)
        np.testing.assert_allclose(plan.source_marginal, problem.source_weights)
        np.testing.assert_allclose(plan.target_marginal, problem.target_weights)
        self.assertAlmostEqual(
            plan.cost, float(np.sum(plan.matrix * problem.cost_matrix))
        )

    def test_unbalanced_problem_raises(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)), [1.0, 1.0], [0.5, 0.5])
        with self.assertRaises(ValueError):
            _ = ExactSolver().solve(problem)


if __name__ == "__main__":
    _ = unittest.main()
