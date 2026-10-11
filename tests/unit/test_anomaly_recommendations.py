"""Tests for anomaly recommendations."""

import pytest

from backend.app.services.anomaly_recommendations import (
    recommend_actions,
)


@pytest.mark.parametrize(
    "severity",
    ["none", "low", "medium", "high", "critical"],
)
def test_returns_actions_for_each_severity(
    severity: str,
) -> None:
    result = recommend_actions(
        severity=severity,
        event_type="temperature",
    )

    assert result["severity"] == severity
    assert result["event_type"] == "temperature"
    assert isinstance(result["recommended_actions"], list)
    assert result["recommended_actions"]


def test_critical_recommendations_include_safety_procedures() -> None:
    result = recommend_actions(
        severity="critical",
        event_type="temperature",
    )

    assert any(
        "safety" in action.lower()
        for action in result["recommended_actions"]
    )


def test_invalid_severity_is_rejected() -> None:
    with pytest.raises(ValueError):
        recommend_actions(
            severity="urgent",
            event_type="temperature",
        )