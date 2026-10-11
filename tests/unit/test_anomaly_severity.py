"""Tests for anomaly severity classification."""

import pytest

from backend.app.services.anomaly_severity import (
    classify_severity,
)


def make_explanation(deviation: float) -> dict:
    return {
        "evidence": [
            {
                "field": "value_deviation",
                "deviation_percent": deviation,
            }
        ]
    }


@pytest.mark.parametrize(
    ("deviation", "expected"),
    [
        (0, "low"),
        (9.99, "low"),
        (10, "medium"),
        (24.99, "medium"),
        (25, "high"),
        (49.99, "high"),
        (50, "critical"),
        (75, "critical"),
    ],
)
def test_classifies_severity_thresholds(
    deviation: float,
    expected: str,
) -> None:
    result = classify_severity(
        is_anomaly=True,
        explanation=make_explanation(deviation),
    )

    assert result["severity"] == expected


def test_normal_event_has_no_severity() -> None:
    result = classify_severity(
        is_anomaly=False,
        explanation={"evidence": []},
    )

    assert result["severity"] == "none"


def test_missing_deviation_defaults_to_medium() -> None:
    result = classify_severity(
        is_anomaly=True,
        explanation={"evidence": []},
    )

    assert result["severity"] == "medium"


def test_invalid_threshold_order_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_severity(
            is_anomaly=True,
            explanation=make_explanation(30),
            thresholds={
                "medium": 25,
                "high": 10,
                "critical": 50,
            },
        )