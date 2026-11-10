from enum import Enum


class SinkhornAlgorithm(Enum):
    """
    Iteration scheme of POT's balanced Sinkhorn solver.

    Attributes
    ----------
    SINKHORN
        Sinkhorn-Knopp matrix scaling; fastest, but may underflow for small
        regularization.
    LOG_DOMAIN
        Sinkhorn iterations in the log domain; numerically stable for small
        regularization.
    STABILIZED
        Sinkhorn iterations with absorption of large scaling factors.
    EPSILON_SCALING
        Stabilized Sinkhorn with a decreasing regularization schedule.
    GREENKHORN
        Greedy coordinate-wise Sinkhorn updates.
    """

    SINKHORN = "sinkhorn"
    LOG_DOMAIN = "sinkhorn_log"
    STABILIZED = "sinkhorn_stabilized"
    EPSILON_SCALING = "sinkhorn_epsilon_scaling"
    GREENKHORN = "greenkhorn"
