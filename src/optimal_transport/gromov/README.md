# gromov

## Overview

Gromov-Wasserstein transport, which matches two point sets by comparing the costs within each set, so the sets may live in different feature spaces. Solvers return a `TransportPlan` whose `cost` is the Gromov-Wasserstein discrepancy.

## Components

| Component | Description |
|-----------|-------------|
| [problem.py](./problem.py) | `GromovWassersteinProblem`, built from intra-set cost matrices or from samples. |
| [solver.py](./solver.py) | Abstract base class for Gromov-Wasserstein solvers. |
| [solvers/](./solvers/README.md) | Concrete solvers. |
| [loss.py](./loss.py) | `GromovLoss` enum comparing source and target cost entries. |

## Examples

```python
from optimal_transport import EntropicGromovWassersteinSolver, GromovWassersteinProblem

problem = GromovWassersteinProblem.from_samples(source_points, target_points)
plan = EntropicGromovWassersteinSolver(regularization=0.1).solve(problem)
```
