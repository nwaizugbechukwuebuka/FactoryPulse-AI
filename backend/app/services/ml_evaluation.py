"""Evaluation utilities for manufacturing anomaly detection."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    actual_labels: pd.Series,
    predicted_labels: pd.Series,
) -> dict[str, Any]:
    """Evaluate binary anomaly predictions against ground truth."""

    if len(actual_labels) == 0:
        raise ValueError("Evaluation labels must not be empty.")

    if len(actual_labels) != len(predicted_labels):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    actual = actual_labels.astype(bool)
    predicted = predicted_labels.astype(bool)

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[False, True],
    )

    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )

    return {
        "sample_count": len(actual),
        "actual_anomaly_count": int(actual.sum()),
        "predicted_anomaly_count": int(predicted.sum()),
        "precision": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }