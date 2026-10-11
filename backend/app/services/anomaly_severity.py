"""Classify production anomaly severity."""

import math
from collections.abc import Mapping
from typing import Any, Literal

Severity = Literal["none", "low", "medium", "high", "critical"]


DEFAULT_THRESHOLDS = {
    "medium": 10.0,
    "high": 25.0,
    "critical": 50.0,
}


def classify_severity(
    is_anomaly: bool,
    explanation: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assign severity using available evidence and configurable thresholds."""
    if not isinstance(is_anomaly, bool):
        raise TypeError("is_anomaly must be a boolean.")

    if not isinstance(explanation, Mapping):
        raise TypeError("explanation must be a mapping.")

    if not is_anomaly:
        return {
            "severity": "none",
            "reason": "No anomaly was detected.",
        }

    configured = dict(DEFAULT_THRESHOLDS)

    if thresholds is not None:
        if not isinstance(thresholds, Mapping):
            raise TypeError("thresholds must be a mapping or None.")

        configured.update(thresholds)

    required = {"medium", "high", "critical"}

    if set(configured) != required:
        raise ValueError(
            "Thresholds must contain medium, high, and critical."
        )

    values = list(configured.values())

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError("Severity thresholds must be numeric.")

    if any(
        not math.isfinite(value) or value < 0
        for value in values
    ):
        raise ValueError(
            "Severity thresholds must be finite and nonnegative."
        )

    if not (
        configured["medium"]
        < configured["high"]
        < configured["critical"]
    ):
        raise ValueError(
            "Thresholds must increase from medium to critical."
        )

    deviation = None
    evidence = explanation.get("evidence", [])

    if isinstance(evidence, list):
        for item in evidence:
            if (
                isinstance(item, Mapping)
                and item.get("field") == "value_deviation"
            ):
                deviation = item.get("deviation_percent")
                break

    if (
        isinstance(deviation, bool)
        or not isinstance(deviation, (int, float))
        or not math.isfinite(deviation)
        or deviation < 0
    ):
        return {
            "severity": "medium",
            "reason": (
                "An anomaly was detected, but quantitative evidence "
                "is insufficient for a more precise severity assessment."
            ),
        }

    if deviation >= configured["critical"]:
        severity: Severity = "critical"
    elif deviation >= configured["high"]:
        severity = "high"
    elif deviation >= configured["medium"]:
        severity = "medium"
    else:
        severity = "low"

    return {
        "severity": severity,
        "reason": (
            f"The measured deviation is {deviation:.2f}%, "
            f"which maps to the configured '{severity}' severity level."
        ),
    }