"""Isolation Forest model for manufacturing anomaly detection."""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest

from backend.app.services.ml_features import build_features


@dataclass
class IsolationForestDetector:
    """Train and apply an Isolation Forest anomaly detector."""

    contamination: float = 0.10
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate configuration and initialize the model."""

        if not 0.0 < self.contamination <= 0.5:
            raise ValueError(
                "contamination must be greater than 0 and at most 0.5."
            )

        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
        )

        self.feature_columns: list[str] | None = None
        self.is_fitted = False

    def fit(self, dataframe: pd.DataFrame) -> "IsolationForestDetector":
        """Fit the model using measurements without ground-truth labels."""

        features = build_features(dataframe)

        if len(features) < 2:
            raise ValueError(
                "At least two observations are required to train the model."
            )

        self.model.fit(features)
        self.feature_columns = list(features.columns)
        self.is_fitted = True

        return self

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return anomaly predictions and scores for each observation."""

        if not self.is_fitted or self.feature_columns is None:
            raise RuntimeError(
                "The detector must be fitted before prediction."
            )

        features = build_features(dataframe)

        if list(features.columns) != self.feature_columns:
            raise ValueError(
                "Prediction features do not match the training features."
            )

        predictions = self.model.predict(features)
        scores = self.model.decision_function(features)

        result = pd.DataFrame(index=dataframe.index)

        result["predicted_anomaly"] = predictions == -1
        result["anomaly_score"] = scores

        return result