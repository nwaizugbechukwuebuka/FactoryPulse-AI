"""Compare rule-based and Isolation Forest anomaly detection."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.production_event import ProductionEvent
from backend.app.services.anomaly_detection import classify_measurement
from backend.app.services.ml_anomaly_detection import (
    IsolationForestDetector,
)
from backend.app.services.ml_features import build_features

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_events.csv"
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "anomaly_model_comparison.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def calculate_metrics(
    actual: list[int],
    predicted: list[int],
) -> dict[str, float | int]:
    """Calculate binary classification metrics with input validation."""

    if not actual or not predicted:
        raise ValueError("Actual and predicted labels cannot be empty.")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    if any(label not in (0, 1) for label in actual):
        raise ValueError("Actual labels must contain only 0 and 1.")

    if any(label not in (0, 1) for label in predicted):
        raise ValueError("Predicted labels must contain only 0 and 1.")

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": round(
            float(precision_score(actual, predicted, zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(actual, predicted, zero_division=0)),
            4,
        ),
        "f1_score": round(
            float(f1_score(actual, predicted, zero_division=0)),
            4,
        ),
        "false_positive_rate": round(
            float(fp / (fp + tn)) if (fp + tn) else 0.0,
            4,
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main() -> None:
    """Evaluate both detectors on identical held-out observations."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Manufacturing dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    required_columns = {
        "known_anomaly",
        "event_type",
        "value",
        "target_value",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Cannot compare models on an empty dataset.")

    if dataframe["known_anomaly"].isna().any():
        raise ValueError("Ground-truth anomaly labels cannot be missing.")

    labels = dataframe["known_anomaly"].astype(int)

    if not labels.isin([0, 1]).all():
        raise ValueError("known_anomaly must contain only 0 and 1.")

    if labels.nunique() != 2:
        raise ValueError(
            "Comparison requires both normal and anomalous observations."
        )

    train_indices, test_indices = train_test_split(
        dataframe.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train_data = dataframe.loc[train_indices].copy()
    test_data = dataframe.loc[test_indices].copy()

    y_test = labels.loc[test_indices].tolist()

    # Validate feature engineering without passing transformed features
    # into a detector that performs feature engineering internally.
    build_features(train_data)
    build_features(test_data)

    # The detector accepts raw data and builds its own features.
    # Ground-truth labels must never be used as model input.
    detector = IsolationForestDetector()
    detector.fit(train_data)

    ml_predictions_frame = detector.predict(test_data)

    # Support the existing detector's documented DataFrame result.
    if isinstance(ml_predictions_frame, pd.DataFrame):
        if "predicted_anomaly" not in ml_predictions_frame.columns:
            raise ValueError(
                "ML prediction output must contain 'predicted_anomaly'."
            )

        ml_predictions = (
            ml_predictions_frame["predicted_anomaly"]
            .astype(int)
            .tolist()
        )
    else:
        ml_predictions = [
            int(value)
            for value in ml_predictions_frame
        ]

    # Apply the existing rule-based detector to exactly the same test rows.
    model_columns = list(ProductionEvent.model_fields)
    rule_predictions: list[int] = []

    for _, row in test_data.iterrows():
        event_data = row[model_columns].to_dict()
        event = ProductionEvent.model_validate(event_data)

        result = classify_measurement(event)
        rule_predictions.append(int(result["is_anomaly"]))

    if not (
        len(y_test)
        == len(ml_predictions)
        == len(rule_predictions)
    ):
        raise ValueError(
            "Ground-truth and prediction counts do not match."
        )

    results = {
        "project": "FactoryPulse-AI",
        "evaluation_type": "synthetic_manufacturing_anomaly_detection",
        "random_state": RANDOM_STATE,
        "dataset_rows": len(dataframe),
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "ground_truth_column": "known_anomaly",
        "models": {
            "rule_based": calculate_metrics(
                y_test,
                rule_predictions,
            ),
            "isolation_forest": calculate_metrics(
                y_test,
                ml_predictions,
            ),
        },
    }

    results["comparison"] = {
        metric: round(
            results["models"]["isolation_forest"][metric]
            - results["models"]["rule_based"][metric],
            4,
        )
        for metric in (
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        )
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nFACTORYPULSE AI — MODEL COMPARISON")
    print("=" * 60)
    print(f"Dataset rows: {len(dataframe)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")

    print("\nMETRICS")
    print("-" * 60)

    for model_name, metrics in results["models"].items():
        print(f"\n{model_name.replace('_', ' ').title()}")

        for metric in (
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ):
            print(f"  {metric}: {metrics[metric]:.4f}")

        print(
            "  Confusion matrix counts: "
            f"TN={metrics['true_negative']}, "
            f"FP={metrics['false_positive']}, "
            f"FN={metrics['false_negative']}, "
            f"TP={metrics['true_positive']}"
        )

    print("\nISOLATION FOREST MINUS RULE-BASED")
    print("-" * 60)

    for metric, difference in results["comparison"].items():
        print(f"  {metric}: {difference:+.4f}")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()