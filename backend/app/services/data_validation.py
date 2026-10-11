"""CSV loading and production-event validation."""

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from backend.app.models.production_event import ProductionEvent

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "manufacturing_events.csv"


def validate_event_records(
    records: list[dict[str, Any]],
) -> tuple[list[ProductionEvent], list[dict[str, Any]]]:
    """Validate records and return valid events and structured errors."""

    valid_events: list[ProductionEvent] = []
    errors: list[dict[str, Any]] = []

    seen_event_ids: set[str] = set()

    for row_number, record in enumerate(records, start=2):
        try:
            event = ProductionEvent.model_validate(record)

            if event.event_id in seen_event_ids:
                raise ValueError(
                    f"Duplicate event ID: {event.event_id}"
                )

            seen_event_ids.add(event.event_id)
            valid_events.append(event)

        except (ValidationError, ValueError) as exc:
            errors.append(
                {
                    "row_number": row_number,
                    "event_id": record.get("event_id"),
                    "error": str(exc),
                }
            )

    return valid_events, errors


def load_and_validate_csv(
    file_path: str | Path = DEFAULT_DATASET,
) -> dict[str, Any]:
    """Load a CSV dataset and validate every record."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError("Dataset must contain at least one event.")

    required_columns = set(ProductionEvent.model_fields)

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.astype(object).where(
        pd.notna(dataframe),
        None,
    )

    valid_events, errors = validate_event_records(
        dataframe.to_dict(orient="records")
    )

    return {
        "source": str(path),
        "total_records": len(dataframe),
        "valid_count": len(valid_events),
        "invalid_count": len(errors),
        "valid_events": valid_events,
        "errors": errors,
    }