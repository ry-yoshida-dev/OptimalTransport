# solvers

## Overview

Concrete `LinearTransportSolver` implementations, each a frozen dataclass of the method's parameters.

## Components

| Component | Description |
|-----------|-------------|
| [exact.py](./exact.py) | `ExactSolver`: exact transport by network simplex. |
| [sinkhorn.py](./sinkhorn.py) | `SinkhornSolver`: entropy-regularized transport. |
| [unbalanced_sinkhorn.py](./unbalanced_sinkhorn.py) | `UnbalancedSinkhornSolver`: entropy-regularized transport with relaxed marginals. |
| [partial.py](./partial.py) | `PartialSolver`: exact transport of a fixed amount of mass. |
