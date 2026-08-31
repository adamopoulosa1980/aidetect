"""Evaluation utilities.

Convention: higher score = more likely AI. (For Binoculars, pass the
NEGATED score, since raw Binoculars scores are lower for AI text.)

The metric that matters in deployment is FPR at your operating threshold —
falsely accusing a human is usually far more costly than missing AI text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class EvalReport:
    auroc: float
    threshold: float
    tpr_at_threshold: float
    fpr_at_threshold: float
    n_human: int
    n_ai: int

    def __str__(self) -> str:
        return (
            f"AUROC: {self.auroc:.4f} | threshold: {self.threshold:.4f} | "
            f"TPR: {self.tpr_at_threshold:.3f} | FPR: {self.fpr_at_threshold:.3f} "
            f"(n_human={self.n_human}, n_ai={self.n_ai})"
        )


def pick_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01) -> float:
    """Pick the threshold giving the best TPR subject to FPR <= max_fpr."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = fpr <= max_fpr
    if not valid.any():
        return float(thresholds[0])
    best = np.argmax(tpr[valid])
    return float(thresholds[valid][best])


def pick_threshold_max_accuracy(
    scores: np.ndarray, labels: np.ndarray, balanced: bool = True
) -> float:
    """Pick the threshold maximizing accuracy, ignoring any FPR budget.

    balanced=True maximizes Youden's J (TPR - FPR), i.e. balanced accuracy —
    robust when class ratios differ between eval set and deployment.
    balanced=False maximizes raw accuracy at the eval set's class ratio
    (only meaningful if that ratio matches real-world prevalence).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if balanced:
        return float(thresholds[np.argmax(tpr - fpr)])
    n_pos, n_neg = (labels == 1).sum(), (labels == 0).sum()
    acc = (tpr * n_pos + (1 - fpr) * n_neg) / (n_pos + n_neg)
    return float(thresholds[np.argmax(acc)])


def evaluate(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    threshold: float | None = None,
    max_fpr: float = 0.01,
) -> EvalReport:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    auroc = roc_auc_score(labels, scores)
    if threshold is None:
        threshold = pick_threshold(scores, labels, max_fpr=max_fpr)
    preds = scores >= threshold
    ai_mask = labels == 1
    tpr = preds[ai_mask].mean() if ai_mask.any() else float("nan")
    fpr = preds[~ai_mask].mean() if (~ai_mask).any() else float("nan")
    return EvalReport(
        auroc=float(auroc),
        threshold=float(threshold),
        tpr_at_threshold=float(tpr),
        fpr_at_threshold=float(fpr),
        n_human=int((~ai_mask).sum()),
        n_ai=int(ai_mask.sum()),
    )
