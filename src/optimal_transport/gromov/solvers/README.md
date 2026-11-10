# solvers

## Overview

Concrete `GromovWassersteinSolver` implementations, each a frozen dataclass of the method's parameters.

## Components

| Component | Description |
|-----------|-------------|
| [exact.py](./exact.py) | `ExactGromovWassersteinSolver`: conditional gradient. |
| [entropic.py](./entropic.py) | `EntropicGromovWassersteinSolver`: entropy-regularized projected gradient descent. |
