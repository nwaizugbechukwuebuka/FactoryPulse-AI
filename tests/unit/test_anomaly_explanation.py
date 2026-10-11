"""Tests for anomaly explanations."""

import pytest

from backend.app.services.anomaly_explanation import (
    calculate_deviation_percent,
    explain_anomaly,
)


def test_calculates_percentage_deviation() -> None:
    assert calculate_deviation_percent(145, 100) == 45.0


def test_zero_expected_value_returns_none() -> None:
    assert calculate_deviation_percent(10, 0) is None


def test_missing_expected_value_returns_none() -> None:
    assert calculate_deviation_percent(10, None) is None


def test_detected_anomaly_includes_measurement_evidence() -> None:
    result = explain_anomaly(
        event={
            "event_id": "EVT-001",
            "event_type": "temperature",
            "value": 145,
            "target_value": 100,
        },
        is_anomaly=True,
        detection_method="hybrid",
        ml_prediction=1,
    )

    assert result["is_anomaly"] is True
    assert result["detection_method"] == "hybrid"
    assert "45.00%" in result["summary"]
    assert result["evidence"]


def test_normal_event_is_not_described_as_an_anomaly() -> None:
    result = explain_anomaly(
        event={
            "event_type": "temperature",
            "value": 100,
            "target_value": 100,
        },
        is_anomaly=False,
        detection_method="rule_based",
    )

    assert result["is_anomaly"] is False
    assert "No anomaly was detected" in result["summary"]


def test_ml_explanation_does_not_invent_rule_violation() -> None:
    result = explain_anomaly(
        event={
            "event_type": "production_rate",
            "value": 100,
            "target_value": 100,
        },
        is_anomaly=True,
        detection_method="isolation_forest",
        ml_prediction=1,
    )

    assert "Isolation Forest" in result["summary"]
    assert result["rule_reason"] is None


def test_invalid_detection_method_is_rejected() -> None:
    with pytest.raises(ValueError):
        explain_anomaly(
            event={"event_type": "temperature"},
            is_anomaly=True,
            detection_method="unknown",
        )


def test_invalid_ml_prediction_is_rejected() -> None:
    with pytest.raises(ValueError):
        explain_anomaly(
            event={"event_type": "temperature"},
            is_anomaly=True,
            detection_method="hybrid",
            ml_prediction=2,
        )


def test_non_finite_measurement_is_handled_safely() -> None:
    result = explain_anomaly(
        event={
            "event_type": "temperature",
            "value": float("nan"),
            "target_value": 100,
        },
        is_anomaly=True,
        detection_method="isolation_forest",
    )

    assert result["evidence"] == [
        {
            "field": "target_value",
            "expected": 100.0,
        }
    ]