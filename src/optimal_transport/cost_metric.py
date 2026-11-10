from enum import Enum


class CostMetric(Enum):
    """
    Ground metric used to build a cost matrix from sample coordinates.

    Attributes
    ----------
    SQUARED_EUCLIDEAN
        Squared Euclidean distance; with exact transport it yields the squared
        2-Wasserstein distance.
    EUCLIDEAN
        Euclidean distance.
    CITYBLOCK
        Manhattan (L1) distance.
    COSINE
        Cosine distance.
    """

    SQUARED_EUCLIDEAN = "sqeuclidean"
    EUCLIDEAN = "euclidean"
    CITYBLOCK = "cityblock"
    COSINE = "cosine"
