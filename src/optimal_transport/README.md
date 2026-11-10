# optimal_transport

## Overview

Unified optimal transport interface. `linear/` covers problems with a cost between source and target points, and `gromov/` problems that compare the internal geometry of two point sets.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases (`NumericArray`, `FloatArray`, `IntegerArray`). |
| [method.py](./method.py) | `TransportMethod` enum of the methods that produce a plan. |
| [cost_metric.py](./cost_metric.py) | `CostMetric` enum of ground metrics for building cost matrices from samples. |
| [array_validation.py](./array_validation.py) | Validation of input arrays and marginal weights. |
| [plan.py](./plan.py) | `TransportPlan`, the shared result type. |
| [linear/](./linear/README.md) | Optimal transport with a source-to-target cost matrix. |
| [gromov/](./gromov/README.md) | Gromov-Wasserstein transport. |
