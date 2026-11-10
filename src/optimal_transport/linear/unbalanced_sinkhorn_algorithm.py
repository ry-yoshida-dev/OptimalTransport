from enum import Enum


class UnbalancedSinkhornAlgorithm(Enum):
    """
    Iteration scheme of POT's unbalanced Sinkhorn solver.

    Attributes
    ----------
    SINKHORN
        Sinkhorn-Knopp matrix scaling.
    STABILIZED
        Sinkhorn iterations with absorption of large scaling factors.
    TRANSLATION_INVARIANT
        Translation-invariant Sinkhorn, which converges faster when the
        marginal relaxation is large.
    """

    SINKHORN = "sinkhorn"
    STABILIZED = "sinkhorn_stabilized"
    TRANSLATION_INVARIANT = "sinkhorn_translation_invariant"
