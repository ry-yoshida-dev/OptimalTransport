from __future__ import annotations

import unittest

import numpy as np

from optimal_transport import TransportMethod, TransportPlan


def partial_plan() -> TransportPlan:
    """Return a plan whose second source point sends no mass."""
    return TransportPlan(
        matrix=[[0.25, 0.25, 0.0], [0.0, 0.0, 0.0], [0.0, 0.1, 0.4]],
        cost=1.5,
        method=TransportMethod.PARTIAL,
    )


class TestTransportPlan(unittest.TestCase):
    def test_matrix_is_read_only_copy(self) -> None:
        source = np.eye(2)
        plan = TransportPlan(matrix=source, cost=0.0, method=TransportMethod.EXACT)
        source[0, 0] = 5.0
        self.assertEqual(plan.matrix[0, 0], 1.0)
        with self.assertRaises(ValueError):
            plan.matrix[0, 0] = 2.0

    def test_invalid_matrix_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = TransportPlan(matrix=[1.0, 2.0], cost=0.0, method=TransportMethod.EXACT)
        with self.assertRaises(ValueError):
            _ = TransportPlan(matrix=[[-1.0]], cost=0.0, method=TransportMethod.EXACT)
        with self.assertRaises(ValueError):
            _ = TransportPlan(matrix=[[np.nan]], cost=0.0, method=TransportMethod.EXACT)

    def test_marginals_and_total_mass(self) -> None:
        np.testing.assert_allclose(partial_plan().source_marginal, [0.5, 0.0, 0.5])
        np.testing.assert_allclose(partial_plan().target_marginal, [0.25, 0.35, 0.4])
        self.assertAlmostEqual(partial_plan().total_mass, 1.0)

    def test_assignments_mark_unassigned_rows(self) -> None:
        assignments = partial_plan().get_assignments()
        self.assertEqual(assignments.tolist(), [0, TransportPlan.UNASSIGNED, 2])
        self.assertEqual(assignments.dtype, np.int64)

    def test_barycentric_projection(self) -> None:
        plan = TransportPlan(
            matrix=[[0.25, 0.25], [0.0, 0.5]], cost=0.0, method=TransportMethod.EXACT
        )
        target = np.array([[0.0, 0.0], [2.0, 4.0]])
        np.testing.assert_allclose(
            plan.barycentric_projection(target), [[1.0, 2.0], [2.0, 4.0]]
        )

    def test_barycentric_projection_rejects_unassigned_rows(self) -> None:
        with self.assertRaises(ValueError):
            _ = partial_plan().barycentric_projection(np.zeros((3, 2)))

    def test_barycentric_projection_rejects_wrong_shape(self) -> None:
        with self.assertRaises(ValueError):
            _ = partial_plan().barycentric_projection(np.zeros((2, 2)))


if __name__ == "__main__":
    _ = unittest.main()
