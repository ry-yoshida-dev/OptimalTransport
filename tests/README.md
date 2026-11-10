# Tests

## Overview

Unit tests for the `optimal_transport` package, one file per solver or shared type.

## Components

| Component | Description |
| --- | --- |
| [`helpers.py`](helpers.py) | Shared synthetic point sets. |
| [`test_transport_plan.py`](test_transport_plan.py) | Tests for `TransportPlan` helpers and invariants. |
| [`test_linear_transport_problem.py`](test_linear_transport_problem.py) | Tests for `LinearTransportProblem` validation and construction from samples. |
| [`test_exact_solver.py`](test_exact_solver.py) | Tests for `ExactSolver`. |
| [`test_sinkhorn_solver.py`](test_sinkhorn_solver.py) | Tests for `SinkhornSolver`. |
| [`test_unbalanced_sinkhorn_solver.py`](test_unbalanced_sinkhorn_solver.py) | Tests for `UnbalancedSinkhornSolver`. |
| [`test_partial_solver.py`](test_partial_solver.py) | Tests for `PartialSolver`. |
| [`test_gromov_wasserstein.py`](test_gromov_wasserstein.py) | Tests for `GromovWassersteinProblem` and both Gromov-Wasserstein solvers. |
