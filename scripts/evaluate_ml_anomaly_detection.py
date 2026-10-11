"""Train and evaluate Isolation Forest on synthetic manufacturing data."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure the project root is importable when this file runs directly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.ml_anomaly_detection import (
    IsolationForestDetector,
)
from backend.app.services.ml_evaluation import (
    evaluate_predictions,
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manufacturing_events.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_anomaly_evaluation.json"
)

RANDOM_STATE = 42
CONTAMINATION = 0.10
TEST_SIZE = 0.25


def main() -> None:
    """Train the detector and evaluate held-out observations."""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Manufacturing dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(DATASET_PATH)

    required_columns = {
        "event_type",
        "value",
        "target_value",
        "known_anomaly",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Cannot evaluate an empty dataset.")

    raw_labels = dataframe["known_anomaly"]

    if raw_labels.isna().any():
        raise ValueError(
            "Ground-truth labels must not contain missing values."
        )

    # Accept only explicit Boolean values or 0/1 labels.
    if not raw_labels.isin([True, False, 0, 1]).all():
        raise ValueError(
            "Ground-truth labels must contain only Boolean values "
            "or 0/1."
        )

    labels = raw_labels.astype(bool)

    if labels.nunique() != 2:
        raise ValueError(
            "Evaluation requires both normal and anomalous observations."
        )

    if len(dataframe) < 4:
        raise ValueError(
            "At least four observations are required for a train/test split."
        )

    train_indices, test_indices = train_test_split(
        dataframe.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    training_data = dataframe.loc[train_indices]
    testing_data = dataframe.loc[test_indices]

    # Ground-truth labels are deliberately excluded from model training.
    detector = IsolationForestDetector(
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )

    detector.fit(training_data)

    predictions = detector.predict(testing_data)

    metrics = evaluate_predictions(
        labels.loc[test_indices],
        predictions["predicted_anomaly"],
    )

    report = {
        "project": "FactoryPulse AI",
        "model": "IsolationForest",
        "dataset": DATASET_PATH.name,
        "dataset_rows": len(dataframe),
        "training_rows": len(training_data),
        "test_rows": len(testing_data),
        "random_state": RANDOM_STATE,
        "contamination": CONTAMINATION,
        "test_size": TEST_SIZE,
        "features": detector.feature_columns,
        "metrics": metrics,
        "limitations": [
            "Evaluation uses synthetic manufacturing data.",
            "Ground-truth labels represent injected anomalies.",
            "Performance does not establish real-world factory accuracy.",
            "Contamination is configured as a modeling assumption.",
            (
                "The test set is held out from fitting, but the synthetic "
                "data distribution may not represent real factory operations."
            ),
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nFACTORYPULSE AI — ISOLATION FOREST EVALUATION")
    print("=" * 52)
    print(f"Dataset rows:   {len(dataframe)}")
    print(f"Training rows:  {len(training_data)}")
    print(f"Test rows:      {len(testing_data)}")
    print(
        "Features:       "
        f"{', '.join(detector.feature_columns or [])}"
    )

    print("\nEVALUATION METRICS")
    print("=" * 52)
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1_score']:.3f}")

    print("\nCONFUSION MATRIX")
    print("=" * 52)

    for name, value in metrics["confusion_matrix"].items():
        print(f"{name.replace('_', ' ').title()}: {value}")

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()