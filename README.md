# OptimalTransport

## Overview

OptimalTransport (`optimal_transport`) wraps the solvers of [POT](https://pythonot.github.io/) behind one typed API: build an immutable problem from a cost matrix or from samples, pass it to a solver dataclass, and get back a `TransportPlan` with the coupling, its cost, and helpers such as per-point assignments and barycentric projection. Everything is exported from the top-level `optimal_transport` package.

| Solver | Problem | Marginals | Regularized | POT function |
|---|---|---|---|---|
| `ExactSolver` | `LinearTransportProblem` | Balanced | No | `ot.emd` |
| `SinkhornSolver` | `LinearTransportProblem` | Balanced | Entropic | `ot.sinkhorn` |
| `UnbalancedSinkhornSolver` | `LinearTransportProblem` | Relaxed (KL) | Entropic | `ot.unbalanced.sinkhorn_unbalanced` |
| `PartialSolver` | `LinearTransportProblem` | Fixed transported mass | No | `ot.partial.partial_wasserstein` |
| `ExactGromovWassersteinSolver` | `GromovWassersteinProblem` | Balanced | No | `ot.gromov.gromov_wasserstein` |
| `EntropicGromovWassersteinSolver` | `GromovWassersteinProblem` | Balanced | Entropic | `ot.gromov.entropic_gromov_wasserstein` |

## Installation

```bash
pip install .
```

## Usage

```python
import numpy as np

from optimal_transport import ExactSolver, LinearTransportProblem

rng = np.random.default_rng(0)
source = rng.standard_normal((40, 2))
target = rng.standard_normal((50, 2)) + 3.0

problem = LinearTransportProblem.from_samples(source, target)
plan = ExactSolver().solve(problem)
print(plan.cost)  # squared 2-Wasserstein distance
mapped = plan.barycentric_projection(target)
```

Compare point sets that live in different spaces with Gromov-Wasserstein:

```python
from optimal_transport import ExactGromovWassersteinSolver, GromovWassersteinProblem

problem = GromovWassersteinProblem.from_samples(source, rng.standard_normal((30, 3)))
plan = ExactGromovWassersteinSolver().solve(problem)
```
