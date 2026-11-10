from enum import Enum


class EntropicRegularization(Enum):
    """
    Entropic regularization term of the unbalanced Sinkhorn objective.

    Attributes
    ----------
    KL
        Kullback-Leibler divergence to the product of the marginals.
    ENTROPY
        Negative entropy of the plan.
    """

    KL = "kl"
    ENTROPY = "entropy"
