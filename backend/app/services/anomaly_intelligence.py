"""Orchestrate anomaly explanation, severity, and recommendations."""

from collections.abc import Mapping
from typing import Any

from backend.app.services.anomaly_explanation import explain_anomaly
from backend.app.services.anomaly_recommendations import recommend_actions
from backend.app.services.anomaly_severity import classify_severity


def build_anomaly_intelligence(
    event: Mapping[str, Any],
    is_anomaly: bool,
    detection_method: str,
    rule_result: Mapping[str, Any] | None = None,
    ml_prediction: int | None = None,
) -> dict[str, Any]:
    """Build a complete intelligence response for one production event."""
    explanation = explain_anomaly(
        event=event,
        is_anomaly=is_anomaly,
        detection_method=detection_method,
        rule_result=rule_result,
        ml_prediction=ml_prediction,
    )

    severity_result = classify_severity(
        is_anomaly=is_anomaly,
        explanation=explanation,
    )

    recommendations = recommend_actions(
        severity=severity_result["severity"],
        event_type=str(event.get("event_type", "unknown")),
    )

    return {
        "event_id": event.get("event_id"),
        "event_type": event.get("event_type", "unknown"),
        "is_anomaly": is_anomaly,
        "detection_method": detection_method,
        "explanation": explanation,
        "severity": severity_result["severity"],
        "severity_reason": severity_result["reason"],
        "recommended_actions": recommendations["recommended_actions"],
    }