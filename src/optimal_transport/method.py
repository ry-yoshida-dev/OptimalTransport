from enum import Enum


class TransportMethod(Enum):
    """
    Optimal transport method that produced a transport plan.

    Attributes
    ----------
    EXACT
        Exact optimal transport solved by network simplex.
    SINKHORN
        Entropy-regularized optimal transport.
    UNBALANCED_SINKHORN
        Entropy-regularized optimal transport with relaxed marginals.
    PARTIAL
        Exact optimal transport of only part of the mass.
    GROMOV_WASSERSTEIN
        Gromov-Wasserstein transport between intra-domain cost matrices.
    ENTROPIC_GROMOV_WASSERSTEIN
        Entropy-regularized Gromov-Wasserstein transport.
    """

    EXACT = "Exact"
    SINKHORN = "Sinkhorn"
    UNBALANCED_SINKHORN = "UnbalancedSinkhorn"
    PARTIAL = "Partial"
    GROMOV_WASSERSTEIN = "GromovWasserstein"
    ENTROPIC_GROMOV_WASSERSTEIN = "EntropicGromovWasserstein"
