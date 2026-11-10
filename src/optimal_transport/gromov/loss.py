from enum import Enum


class GromovLoss(Enum):
    """
    Loss comparing a source cost entry with a target cost entry.

    Attributes
    ----------
    SQUARE
        Squared difference.
    KL
        Kullback-Leibler loss; requires positive cost matrices.
    """

    SQUARE = "square_loss"
    KL = "kl_loss"
