"""Decision fusion for hybrid anomaly detection."""

from collections.abc import Sequence
from typing import Literal

FusionStrategy = Literal["or", "and"]


def combine_predictions(
    rule_predictions: Sequence[int],
    ml_predictions: Sequence[int],
    strategy: FusionStrategy = "or",
) -> list[int]:
    """Combine rule-based and ML predictions.

    OR strategy:
        Flag an observation if either detector identifies an anomaly.

    AND strategy:
        Flag an observation only if both detectors identify an anomaly.

    Predictions use binary labels:
        0 = normal
        1 = anomaly
    """

    if strategy not in ("or", "and"):
        raise ValueError(
            "strategy must be either 'or' or 'and'."
        )

    if not rule_predictions or not ml_predictions:
        raise ValueError(
            "Prediction sequences cannot be empty."
        )

    if len(rule_predictions) != len(ml_predictions):
        raise ValueError(
            "Prediction sequences must have equal lengths."
        )

    if any(value not in (0, 1) for value in rule_predictions):
        raise ValueError(
            "Rule-based predictions must contain only 0 and 1."
        )

    if any(value not in (0, 1) for value in ml_predictions):
        raise ValueError(
            "ML predictions must contain only 0 and 1."
        )

    if strategy == "or":
        return [
            int(rule == 1 or ml == 1)
            for rule, ml in zip(
                rule_predictions,
                ml_predictions,
                strict=True,
            )
        ]

    return [
        int(rule == 1 and ml == 1)
        for rule, ml in zip(
            rule_predictions,
            ml_predictions,
            strict=True,
        )
    ]