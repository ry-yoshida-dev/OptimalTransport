# linear

## Overview

Optimal transport whose objective is linear in the plan: the cost of moving mass from a source point to a target point is given by a cost matrix. Solvers return a `TransportPlan` whose `cost` is the linear cost without regularization terms.

## Components

| Component | Description |
|-----------|-------------|
| [problem.py](./problem.py) | `LinearTransportProblem`, built from a cost matrix or from samples. |
| [solver.py](./solver.py) | Abstract base class for linear transport solvers. |
| [solvers/](./solvers/README.md) | Concrete solvers. |
| [sinkhorn_algorithm.py](./sinkhorn_algorithm.py) | `SinkhornAlgorithm` enum for `SinkhornSolver`. |
| [unbalanced_sinkhorn_algorithm.py](./unbalanced_sinkhorn_algorithm.py) | `UnbalancedSinkhornAlgorithm` enum for `UnbalancedSinkhornSolver`. |
| [entropic_regularization.py](./entropic_regularization.py) | `EntropicRegularization` enum for `UnbalancedSinkhornSolver`. |

## Examples

```python
from optimal_transport import LinearTransportProblem, SinkhornSolver

problem = LinearTransportProblem(cost_matrix, source_weights, target_weights)
plan = SinkhornSolver(regularization=0.1).solve(problem)
```
