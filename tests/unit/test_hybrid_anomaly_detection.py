"""Tests for hybrid anomaly detection decision fusion."""

import pytest

from backend.app.services.hybrid_anomaly_detection import (
    combine_predictions,
)


def test_or_fusion_flags_anomalies_detected_by_either_model() -> None:
    rule_predictions = [0, 1, 0, 1]
    ml_predictions = [0, 0, 1, 1]

    result = combine_predictions(
        rule_predictions,
        ml_predictions,
        strategy="or",
    )

    assert result == [0, 1, 1, 1]


def test_and_fusion_requires_both_models_to_agree() -> None:
    rule_predictions = [0, 1, 0, 1]
    ml_predictions = [0, 0, 1, 1]

    result = combine_predictions(
        rule_predictions,
        ml_predictions,
        strategy="and",
    )

    assert result == [0, 0, 0, 1]


def test_or_fusion_is_the_default_strategy() -> None:
    result = combine_predictions(
        [0, 1, 0],
        [0, 0, 1],
    )

    assert result == [0, 1, 1]


def test_fusion_rejects_empty_predictions() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        combine_predictions([], [])


def test_fusion_rejects_different_prediction_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        combine_predictions([0, 1], [0])


def test_fusion_rejects_invalid_rule_predictions() -> None:
    with pytest.raises(ValueError, match="Rule-based predictions"):
        combine_predictions([0, 2], [0, 1])


def test_fusion_rejects_invalid_ml_predictions() -> None:
    with pytest.raises(ValueError, match="ML predictions"):
        combine_predictions([0, 1], [0, -1])


def test_fusion_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy"):
        combine_predictions(
            [0, 1],
            [0, 1],
            strategy="xor",  # type: ignore[arg-type]
        )