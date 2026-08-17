"""Reusable CRF training and evaluation logic.

Kept separate from data loading and file I/O so it can be unit tested
against small synthetic examples without touching the real dataset.
"""

import sklearn_crfsuite
from seqeval.metrics import classification_report, f1_score


def train_crf(
    x_train: list[list[dict]],
    y_train: list[list[str]],
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
) -> sklearn_crfsuite.CRF:
    """Fit a CRF with L-BFGS training.

    c1/c2 = 0.1 was chosen over stronger regularization after a small
    sweep: heavier regularization lowered dev F1 further rather than
    closing the train/dev gap, since the gap comes from WNUT-17's
    high-variance entities rather than from overfitting that
    regularization can fix.
    """
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    crf.fit(x_train, y_train)
    return crf


def evaluate_crf(
    crf: sklearn_crfsuite.CRF,
    x: list[list[dict]],
    y_true: list[list[str]],
) -> tuple[str, float]:
    """Return an entity-level classification report and micro F1."""
    y_pred = [list(sequence) for sequence in crf.predict(x)]
    report = classification_report(y_true, y_pred, digits=3)
    f1 = f1_score(y_true, y_pred)
    return report, f1
