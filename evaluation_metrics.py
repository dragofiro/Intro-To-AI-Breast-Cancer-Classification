"""Evaluation helpers that do not load the dataset or training pipeline."""

import numpy as np
from sklearn.metrics import roc_curve


def threshold_for_sensitivity(y, p, target):
    """Return the highest cutoff whose measured sensitivity meets target."""
    if not 0 < target <= 1:
        raise ValueError("target sensitivity must be in (0, 1]")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("threshold selection requires both labels: 0 and 1")

    # Check every distinct score, not only the points retained for plotting.
    _, sensitivity, thresholds = roc_curve(y, p, drop_intermediate=False)
    # Thresholds descend, so the first qualifying cutoff flags the fewest cases.
    return float(thresholds[np.flatnonzero(sensitivity >= target)[0]])
