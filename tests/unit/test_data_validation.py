"""Tests for FactoryPulse AI manufacturing data validation."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.app.models.production_event import ProductionEvent
from backend.app.services.data_validation import (
    load_and_validate_csv,
    validate_event_records,
)


@pytest.fixture
def valid_event() -> dict[str, object]:
    """Return a valid synthetic temperature event."""

    return {
        "event_id": "EVT-TEST-001",
        "timestamp": datetime(2026, 10, 1, 9, 0, tzinfo=UTC),
        "production_line": "LINE-01",
        "equipment_id": "MOTOR-01",
        "event_type": "temperature",
        "value": 72.0,
        "unit": "celsius",
        "target_value": 85.0,
        "status": "normal",
    }


def test_valid_event_is_accepted(
    valid_event: dict[str, object],
) -> None:
    event = ProductionEvent.model_validate(valid_event)

    assert event.event_id == "EVT-TEST-001"
    assert event.value == 72.0


def test_naive_timestamp_is_rejected(
    valid_event: dict[str, object],
) -> None:
    valid_event["timestamp"] = datetime(2026, 10, 1, 9, 0)

    with pytest.raises(ValidationError):
        ProductionEvent.model_validate(valid_event)


def test_negative_measurement_is_rejected(
    valid_event: dict[str, object],
) -> None:
    valid_event["value"] = -5.0

    with pytest.raises(ValidationError):
        ProductionEvent.model_validate(valid_event)


def test_defect_rate_above_100_is_rejected(
    valid_event: dict[str, object],
) -> None:
    valid_event["event_type"] = "defect_rate"
    valid_event["unit"] = "percent"
    valid_event["value"] = 101.0

    with pytest.raises(ValidationError):
        ProductionEvent.model_validate(valid_event)


def test_mismatched_unit_is_rejected(
    valid_event: dict[str, object],
) -> None:
    valid_event["unit"] = "percent"

    with pytest.raises(ValidationError):
        ProductionEvent.model_validate(valid_event)


def test_duplicate_event_ids_are_rejected(
    valid_event: dict[str, object],
) -> None:
    valid_events, errors = validate_event_records(
        [valid_event, valid_event.copy()]
    )

    assert len(valid_events) == 1
    assert len(errors) == 1
    assert "Duplicate event ID" in errors[0]["error"]


def test_csv_dataset_passes_validation() -> None:
    result = load_and_validate_csv()

    assert result["total_records"] == 12
    assert result["valid_count"] == 12
    assert result["invalid_count"] == 0