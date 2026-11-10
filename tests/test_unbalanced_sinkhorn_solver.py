from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import (
    EntropicRegularization,
    LinearTransportProblem,
    TransportMethod,
    UnbalancedSinkhornAlgorithm,
    UnbalancedSinkhornSolver,
)
from tests.helpers import shifted_points


class TestUnbalancedSinkhornSolver(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            _ = UnbalancedSinkhornSolver(regularization=0.0, marginal_relaxation=1.0)
        with self.assertRaises(ValueError):
            _ = UnbalancedSinkhornSolver(regularization=1.0, marginal_relaxation=0.0)
        with self.assertRaises(ValueError):
            _ = UnbalancedSinkhornSolver(
                regularization=1.0, marginal_relaxation=1.0, max_iterations=0
            )
        with self.assertRaises(ValueError):
            _ = UnbalancedSinkhornSolver(
                regularization=1.0, marginal_relaxation=1.0, tolerance=-1.0
            )

    def test_accepts_unbalanced_masses(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(
            source, target, source_weights=np.full(6, 1.0 / 3.0)
        )
        plan = UnbalancedSinkhornSolver(
            regularization=1.0, marginal_relaxation=10.0
        ).solve(problem)
        self.assertEqual(plan.method, TransportMethod.UNBALANCED_SINKHORN)
        self.assertGreater(plan.total_mass, 0.0)

    def test_stronger_relaxation_moves_more_mass(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        loose = UnbalancedSinkhornSolver(regularization=1.0, marginal_relaxation=0.1)
        strict = UnbalancedSinkhornSolver(regularization=1.0, marginal_relaxation=100.0)
        self.assertLess(
            loose.solve(problem).total_mass, strict.solve(problem).total_mass
        )

    def test_options_are_forwarded(self) -> None:
        source, target = shifted_points()
        problem = LinearTransportProblem.from_samples(source, target)
        plan = UnbalancedSinkhornSolver(
            regularization=1.0,
            marginal_relaxation=10.0,
            regularization_type=EntropicRegularization.ENTROPY,
            algorithm=UnbalancedSinkhornAlgorithm.STABILIZED,
        ).solve(problem)
        self.assertTrue(np.all(np.isfinite(plan.matrix)))


if __name__ == "__main__":
    _ = unittest.main()
