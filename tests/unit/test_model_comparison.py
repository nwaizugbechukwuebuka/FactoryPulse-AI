"""Tests for FactoryPulse-AI model comparison metrics."""

import pytest

from scripts.compare_anomaly_models import calculate_metrics


def test_calculate_metrics_returns_expected_confusion_matrix() -> None:
    actual = [0, 0, 0, 0, 1, 1, 1, 1]
    predicted = [0, 0, 0, 1, 1, 1, 1, 0]

    metrics = calculate_metrics(actual, predicted)

    assert metrics["true_negative"] == 3
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_positive"] == 3


def test_calculate_metrics_computes_expected_scores() -> None:
    actual = [0, 0, 0, 0, 1, 1, 1, 1]
    predicted = [0, 0, 0, 1, 1, 1, 1, 0]

    metrics = calculate_metrics(actual, predicted)

    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["f1_score"] == 0.75
    assert metrics["false_positive_rate"] == 0.25


def test_calculate_metrics_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_metrics([], [])


def test_calculate_metrics_rejects_different_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        calculate_metrics([0, 1], [0])


def test_calculate_metrics_rejects_invalid_actual_labels() -> None:
    with pytest.raises(ValueError, match="Actual labels"):
        calculate_metrics([0, 2], [0, 1])


def test_calculate_metrics_rejects_invalid_predictions() -> None:
    with pytest.raises(ValueError, match="Predicted labels"):
        calculate_metrics([0, 1], [0, -1])


def test_calculate_metrics_handles_no_predicted_anomalies() -> None:
    actual = [0, 0, 1, 1]
    predicted = [0, 0, 0, 0]

    metrics = calculate_metrics(actual, predicted)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["true_negative"] == 2
    assert metrics["false_negative"] == 2