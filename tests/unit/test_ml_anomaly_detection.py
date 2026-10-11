"""Tests for machine-learning anomaly detection and evaluation."""

import pandas as pd
import pytest

from backend.app.services.ml_anomaly_detection import (
    IsolationForestDetector,
)
from backend.app.services.ml_evaluation import evaluate_predictions
from backend.app.services.ml_features import build_features


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Return synthetic observations for testing."""

    return pd.DataFrame(
        {
            "event_type": [
                "temperature",
                "temperature",
                "temperature",
                "production_output",
                "defect_rate",
            ],
            "value": [
                76.0,
                77.0,
                75.0,
                98.0,
                1.2,
            ],
            "target_value": [
                85.0,
                85.0,
                85.0,
                95.0,
                2.0,
            ],
            "known_anomaly": [
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_feature_engineering_excludes_ground_truth(
    sample_data: pd.DataFrame,
) -> None:
    features = build_features(sample_data)

    assert "known_anomaly" not in features.columns
    assert "relative_deviation" in features.columns
    assert len(features) == len(sample_data)


def test_feature_engineering_creates_measurement_type_features(
    sample_data: pd.DataFrame,
) -> None:
    features = build_features(sample_data)

    assert "event_type_temperature" in features.columns
    assert "event_type_production_output" in features.columns
    assert "event_type_defect_rate" in features.columns


def test_feature_engineering_rejects_missing_columns() -> None:
    dataframe = pd.DataFrame({"value": [10.0]})

    with pytest.raises(ValueError, match="Missing required feature columns"):
        build_features(dataframe)


def test_detector_requires_fit_before_prediction(
    sample_data: pd.DataFrame,
) -> None:
    detector = IsolationForestDetector()

    with pytest.raises(RuntimeError, match="must be fitted"):
        detector.predict(sample_data)


def test_detector_returns_predictions(
    sample_data: pd.DataFrame,
) -> None:
    detector = IsolationForestDetector(
        contamination=0.20,
        random_state=42,
    )

    detector.fit(sample_data)
    predictions = detector.predict(sample_data)

    assert len(predictions) == len(sample_data)
    assert predictions["predicted_anomaly"].dtype == bool
    assert predictions["anomaly_score"].notna().all()


def test_evaluation_metrics_are_calculated() -> None:
    actual = pd.Series([False, False, True, True])
    predicted = pd.Series([False, True, True, True])

    metrics = evaluate_predictions(actual, predicted)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(0.8)

    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_evaluation_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_predictions(
            pd.Series([], dtype=bool),
            pd.Series([], dtype=bool),
        )


def test_evaluation_rejects_mismatched_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        evaluate_predictions(
            pd.Series([True, False]),
            pd.Series([True]),
        )


def test_invalid_contamination_is_rejected() -> None:
    with pytest.raises(ValueError, match="contamination"):
        IsolationForestDetector(contamination=0.0)