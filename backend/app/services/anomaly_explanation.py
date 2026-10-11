"""Generate evidence-based explanations for production anomalies."""

import math
from collections.abc import Mapping
from typing import Any


def _to_float(value: Any) -> float | None:
    """Convert a numeric value safely to a finite float."""
    if isinstance(value, bool) or value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(result):
        return None

    return result


def calculate_deviation_percent(
    observed: Any,
    expected: Any,
) -> float | None:
    """Calculate absolute percentage deviation when meaningful."""
    observed_value = _to_float(observed)
    expected_value = _to_float(expected)

    if observed_value is None or expected_value is None:
        return None

    if expected_value == 0:
        return None

    deviation = (
        abs(observed_value - expected_value)
        / abs(expected_value)
    ) * 100

    return round(deviation, 2)


def explain_anomaly(
    event: Mapping[str, Any],
    is_anomaly: bool,
    detection_method: str,
    rule_result: Mapping[str, Any] | None = None,
    ml_prediction: int | None = None,
) -> dict[str, Any]:
    """Return a structured, evidence-based anomaly explanation."""
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping.")

    allowed_methods = {
        "rule_based",
        "isolation_forest",
        "hybrid",
    }

    if detection_method not in allowed_methods:
        raise ValueError(
            "detection_method must be rule_based, "
            "isolation_forest, or hybrid."
        )

    if not isinstance(is_anomaly, bool):
        raise TypeError("is_anomaly must be a boolean.")

    if (
        ml_prediction is not None
        and (
            isinstance(ml_prediction, bool)
            or ml_prediction not in (0, 1)
        )
    ):
        raise ValueError("ml_prediction must be 0, 1, or None.")

    event_type = str(event.get("event_type", "unknown"))
    observed = _to_float(event.get("value"))
    expected = _to_float(event.get("target_value"))

    evidence: list[dict[str, Any]] = []

    if observed is not None:
        evidence.append(
            {
                "field": "value",
                "observed": observed,
            }
        )

    if expected is not None:
        evidence.append(
            {
                "field": "target_value",
                "expected": expected,
            }
        )

    deviation_percent = calculate_deviation_percent(
        observed,
        expected,
    )

    if deviation_percent is not None:
        evidence.append(
            {
                "field": "value_deviation",
                "deviation_percent": deviation_percent,
            }
        )

    rule_reason = None

    if rule_result is not None:
        if not isinstance(rule_result, Mapping):
            raise TypeError("rule_result must be a mapping or None.")

        candidate_reason = rule_result.get("reason")

        if isinstance(candidate_reason, str) and candidate_reason.strip():
            rule_reason = candidate_reason.strip()

    if not is_anomaly:
        summary = (
            f"No anomaly was detected for event type '{event_type}'."
        )

    elif detection_method == "isolation_forest":
        summary = (
            f"Event type '{event_type}' was flagged by the "
            "Isolation Forest model. The available measurements "
            "do not establish a specific rule violation."
        )

    elif rule_reason and detection_method in {
        "rule_based",
        "hybrid",
    }:
        summary = rule_reason

    elif deviation_percent is not None and deviation_percent > 0:
        summary = (
            f"Event type '{event_type}' was flagged as anomalous. "
            f"The observed value differs from its target by "
            f"{deviation_percent:.2f}%."
        )

    else:
        summary = (
            f"Event type '{event_type}' was flagged as anomalous, "
            "but the available evidence is insufficient to identify "
            "a specific measurement deviation."
        )

    return {
        "is_anomaly": is_anomaly,
        "detection_method": detection_method,
        "summary": summary,
        "evidence": evidence,
        "rule_reason": rule_reason,
        "ml_prediction": ml_prediction,
    }