"""Tests for manufacturing anomaly detection."""

from datetime import UTC, datetime

import pytest

from backend.app.models.production_event import ProductionEvent
from backend.app.services.anomaly_detection import (
    classify_measurement,
    detect_anomalies,
)


@pytest.fixture
def temperature_event() -> ProductionEvent:
    """Return a normal synthetic temperature measurement."""

    return ProductionEvent(
        event_id="EVT-TEST-001",
        timestamp=datetime(
            2026,
            10,
            1,
            9,
            0,
            tzinfo=UTC,
        ),
        production_line="LINE-01",
        equipment_id="MOTOR-01",
        event_type="temperature",
        value=72.0,
        unit="celsius",
        target_value=85.0,
        status="normal",
    )


def test_normal_temperature_is_not_an_anomaly(
    temperature_event: ProductionEvent,
) -> None:
    result = classify_measurement(temperature_event)

    assert result["severity"] == "normal"
    assert result["is_anomaly"] is False


def test_high_temperature_produces_warning(
    temperature_event: ProductionEvent,
) -> None:
    event = temperature_event.model_copy(
        update={"value": 89.0}
    )

    result = classify_measurement(event)

    assert result["severity"] == "warning"
    assert result["is_anomaly"] is True


def test_critical_temperature_is_detected(
    temperature_event: ProductionEvent,
) -> None:
    event = temperature_event.model_copy(
        update={"value": 90.0}
    )

    result = classify_measurement(event)

    assert result["severity"] == "critical"
    assert result["is_anomaly"] is True


def test_low_production_output_is_critical(
    temperature_event: ProductionEvent,
) -> None:
    event = temperature_event.model_copy(
        update={
            "event_type": "production_output",
            "value": 79.0,
            "unit": "units_per_hour",
            "target_value": 95.0,
        }
    )

    result = classify_measurement(event)

    assert result["severity"] == "critical"
    assert result["is_anomaly"] is True


def test_high_defect_rate_is_critical(
    temperature_event: ProductionEvent,
) -> None:
    event = temperature_event.model_copy(
        update={
            "event_type": "defect_rate",
            "value": 4.8,
            "unit": "percent",
            "target_value": 2.0,
        }
    )

    result = classify_measurement(event)

    assert result["severity"] == "critical"
    assert result["is_anomaly"] is True


def test_deviation_percentage_is_calculated(
    temperature_event: ProductionEvent,
) -> None:
    result = classify_measurement(temperature_event)

    expected_deviation_pct = (
        (72.0 - 85.0) / 85.0
    ) * 100

    assert result["deviation_pct"] == pytest.approx(
        round(expected_deviation_pct, 2)
    )


def test_detection_summary_counts_anomalies(
    temperature_event: ProductionEvent,
) -> None:
    critical_event = temperature_event.model_copy(
        update={
            "event_id": "EVT-TEST-002",
            "value": 93.0,
        }
    )

    result = detect_anomalies(
        [temperature_event, critical_event]
    )

    assert result["total_events"] == 2
    assert result["anomaly_count"] == 1
    assert result["normal_count"] == 1
    assert result["critical_count"] == 1
    assert result["warning_count"] == 0


def test_empty_input_returns_empty_summary() -> None:
    result = detect_anomalies([])

    assert result["total_events"] == 0
    assert result["anomaly_count"] == 0
    assert result["normal_count"] == 0
    assert result["anomalies"] == []
    assert result["results"] == []