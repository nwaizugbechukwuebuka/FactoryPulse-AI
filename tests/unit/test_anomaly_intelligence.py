"""Tests for the anomaly intelligence orchestration service."""

import pytest

from backend.app.services.anomaly_intelligence import (
    build_anomaly_intelligence,
)


def test_builds_complete_anomaly_intelligence() -> None:
    result = build_anomaly_intelligence(
        event={
            "event_id": "EVT-001",
            "event_type": "temperature",
            "value": 145,
            "target_value": 100,
        },
        is_anomaly=True,
        detection_method="hybrid",
        rule_result={
            "is_anomaly": True,
            "reason": "Temperature exceeds the configured target.",
        },
        ml_prediction=1,
    )

    assert result["event_id"] == "EVT-001"
    assert result["is_anomaly"] is True
    assert result["detection_method"] == "hybrid"
    assert result["severity"] == "high"
    assert result["explanation"]["evidence"]
    assert result["recommended_actions"]


def test_normal_event_has_no_recommendation_for_anomaly_response() -> None:
    result = build_anomaly_intelligence(
        event={
            "event_id": "EVT-002",
            "event_type": "temperature",
            "value": 100,
            "target_value": 100,
        },
        is_anomaly=False,
        detection_method="rule_based",
    )

    assert result["is_anomaly"] is False
    assert result["severity"] == "none"
    assert result["recommended_actions"] == [
        "Continue routine production monitoring."
    ]


def test_invalid_detection_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_anomaly_intelligence(
            event={"event_type": "temperature"},
            is_anomaly=True,
            detection_method="unknown",
        )