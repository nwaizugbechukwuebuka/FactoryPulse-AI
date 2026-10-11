"""Feature engineering for manufacturing anomaly detection."""

import pandas as pd

REQUIRED_COLUMNS = {
    "event_type",
    "value",
    "target_value",
}

SUPPORTED_EVENT_TYPES = {
    "temperature",
    "production_output",
    "defect_rate",
}


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert manufacturing measurements into ML-ready features.

    Ground-truth labels and operational status are deliberately excluded
    to prevent target leakage.
    """

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Missing required feature columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError("Cannot build features from an empty dataset.")

    features = dataframe[
        ["event_type", "value", "target_value"]
    ].copy()

    if features.isnull().any().any():
        raise ValueError("Feature inputs must not contain missing values.")

    if not features["event_type"].isin(
        SUPPORTED_EVENT_TYPES
    ).all():
        raise ValueError("Dataset contains unsupported event types.")

    if not pd.api.types.is_numeric_dtype(features["value"]):
        raise ValueError("'value' must contain numeric measurements.")

    if not pd.api.types.is_numeric_dtype(
        features["target_value"]
    ):
        raise ValueError("'target_value' must be numeric.")

    if not features["value"].map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    ).all():
        raise ValueError("'value' must contain finite measurements.")

    if not features["target_value"].map(
        lambda value: (
            pd.notna(value)
            and value > 0
            and abs(value) != float("inf")
        )
    ).all():
        raise ValueError("'target_value' must contain finite positive values.")

    features["relative_deviation"] = (
        features["value"] - features["target_value"]
    ) / features["target_value"]

    event_type_features = pd.get_dummies(
        features["event_type"],
        prefix="event_type",
        dtype=int,
    )

    event_type_features = event_type_features.reindex(
        columns=[
            "event_type_defect_rate",
            "event_type_production_output",
            "event_type_temperature",
        ],
        fill_value=0,
    )

    result = pd.concat(
        [
            features[["relative_deviation"]].reset_index(drop=True),
            event_type_features.reset_index(drop=True),
        ],
        axis=1,
    )

    return result.astype(float)