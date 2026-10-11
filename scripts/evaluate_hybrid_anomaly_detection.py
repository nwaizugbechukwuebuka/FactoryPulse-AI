"""Evaluate hybrid anomaly detection strategies."""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models.production_event import ProductionEvent
from backend.app.services.anomaly_detection import classify_measurement
from backend.app.services.hybrid_anomaly_detection import (
    combine_predictions,
)
from backend.app.services.ml_anomaly_detection import (
    IsolationForestDetector,
)
from scripts.compare_anomaly_models import calculate_metrics

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_events.csv"

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hybrid_anomaly_evaluation.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.25


def main() -> None:
    """Evaluate individual detectors and hybrid fusion strategies."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
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
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("The evaluation dataset cannot be empty.")

    if dataframe["known_anomaly"].isna().any():
        raise ValueError(
            "Ground-truth labels cannot contain missing values."
        )

    labels = dataframe["known_anomaly"]

    if not labels.isin([0, 1]).all():
        raise ValueError(
            "Ground-truth labels must contain only 0 and 1."
        )

    labels = labels.astype(int)

    if labels.nunique() != 2:
        raise ValueError(
            "Evaluation requires both normal and anomalous observations."
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

    # Pass raw event columns to the detector.
    # The detector performs feature engineering internally.
    # Ground-truth labels are not passed to the model.
    detector = IsolationForestDetector()
    detector.fit(train_data)

    ml_output = detector.predict(test_data)

    if isinstance(ml_output, pd.DataFrame):
        if "predicted_anomaly" not in ml_output.columns:
            raise ValueError(
                "ML output is missing the predicted_anomaly column."
            )

        ml_predictions = (
            ml_output["predicted_anomaly"]
            .astype(int)
            .tolist()
        )
    else:
        ml_predictions = [
            int(value)
            for value in ml_output
        ]

    # Generate rule-based predictions for the same test observations.
    model_columns = list(ProductionEvent.model_fields)
    rule_predictions = []

    for _, row in test_data.iterrows():
        event = ProductionEvent.model_validate(
            row[model_columns].to_dict()
        )

        result = classify_measurement(event)
        rule_predictions.append(int(result["is_anomaly"]))

    if not (
        len(y_test)
        == len(rule_predictions)
        == len(ml_predictions)
    ):
        raise ValueError(
            "Prediction counts do not match the test-set size."
        )

    predictions = {
        "rule_based": rule_predictions,
        "isolation_forest": ml_predictions,
        "hybrid_or": combine_predictions(
            rule_predictions,
            ml_predictions,
            strategy="or",
        ),
        "hybrid_and": combine_predictions(
            rule_predictions,
            ml_predictions,
            strategy="and",
        ),
    }

    model_metrics = {
        name: calculate_metrics(y_test, model_predictions)
        for name, model_predictions in predictions.items()
    }

    results = {
        "project": "FactoryPulse-AI",
        "evaluation_type": "synthetic_manufacturing_anomaly_detection",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "dataset_rows": len(dataframe),
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "ground_truth_column": "known_anomaly",
        "fusion_strategies": {
            "hybrid_or": (
                "Anomaly if either detector predicts an anomaly."
            ),
            "hybrid_and": (
                "Anomaly only if both detectors predict an anomaly."
            ),
        },
        "models": model_metrics,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nFACTORYPULSE AI — HYBRID ANOMALY EVALUATION")
    print("=" * 68)
    print(f"Dataset rows: {len(dataframe)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Test rows: {len(test_data)}")

    print("\nMODEL COMPARISON")
    print("-" * 68)

    for model_name, metrics in model_metrics.items():
        print(f"\n{model_name.replace('_', ' ').title()}")

        for metric in (
            "precision",
            "recall",
            "f1_score",
            "false_positive_rate",
        ):
            print(f"  {metric}: {metrics[metric]:.4f}")

        print(
            "  Confusion matrix: "
            f"TN={metrics['true_negative']}, "
            f"FP={metrics['false_positive']}, "
            f"FN={metrics['false_negative']}, "
            f"TP={metrics['true_positive']}"
        )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()