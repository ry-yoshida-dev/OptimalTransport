from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import (
    LinearTransportProblem,
    PartialSolver,
    TransportMethod,
    TransportPlan,
)
from tests.helpers import shifted_points


class TestPartialSolver(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = PartialSolver(mass=0.0)

    def test_transports_requested_mass(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        plan = PartialSolver(mass=0.5).solve(problem)
        self.assertAlmostEqual(plan.total_mass, 0.5)
        self.assertEqual(plan.method, TransportMethod.PARTIAL)
        self.assertTrue(np.all(plan.source_marginal <= problem.source_weights + 1e-12))

    def test_leaves_outlier_unassigned(self) -> None:
        source = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
        target = np.array([[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]])
        problem = LinearTransportProblem.from_samples(source, target)
        plan = PartialSolver(mass=2.0 / 3.0).solve(problem)
        self.assertEqual(plan.get_assignments()[2], TransportPlan.UNASSIGNED)

    def test_default_mass_is_smaller_total(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)), [0.5, 0.5], [0.2, 0.2])
        plan = PartialSolver().solve(problem)
        self.assertAlmostEqual(plan.total_mass, 0.4)

    def test_weights_beyond_the_simplex(self) -> None:
        cost_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        problem = LinearTransportProblem(cost_matrix, [300.0, 300.0], [150.0, 250.0])
        plan = PartialSolver(mass=100.0).solve(problem)
        self.assertAlmostEqual(plan.total_mass, 100.0)
        self.assertAlmostEqual(plan.cost, 0.0)
        plan = PartialSolver().solve(problem)
        np.testing.assert_allclose(plan.target_marginal, [150.0, 250.0])

    def test_mass_equal_to_smaller_total(self) -> None:
        # These weights make the scaled mass round above the smaller scaled total.
        problem = LinearTransportProblem(np.ones((2, 2)), [0.1, 0.7], [4.1, 3.0])
        mass = float(problem.source_weights.sum())
        plan = PartialSolver(mass=mass).solve(problem)
        self.assertAlmostEqual(plan.total_mass, mass)

    def test_mass_above_total_raises(self) -> None:
        problem = LinearTransportProblem(np.ones((2, 2)))
        with self.assertRaises(ValueError):
            _ = PartialSolver(mass=1.5).solve(problem)


if __name__ == "__main__":
    _ = unittest.main()
