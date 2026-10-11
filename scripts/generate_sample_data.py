"""Generate synthetic manufacturing events for FactoryPulse AI."""

from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "manufacturing_events.csv"

TIMESTAMPS = [
    "2026-10-01T09:00:00Z",
    "2026-10-01T09:05:00Z",
    "2026-10-01T09:10:00Z",
    "2026-10-01T09:15:00Z",
]

MEASUREMENTS: list[dict[str, Any]] = [
    {
        "production_line": "LINE-01",
        "equipment_id": "MOTOR-01",
        "event_type": "temperature",
        "values": [72.0, 74.0, 89.0, 93.0],
        "unit": "celsius",
        "target_value": 85.0,
        "statuses": ["normal", "normal", "warning", "critical"],
    },
    {
        "production_line": "LINE-02",
        "equipment_id": "PRESS-01",
        "event_type": "production_output",
        "values": [100.0, 98.0, 79.0, 68.0],
        "unit": "units_per_hour",
        "target_value": 95.0,
        "statuses": ["normal", "normal", "warning", "critical"],
    },
    {
        "production_line": "LINE-03",
        "equipment_id": "INSPECT-01",
        "event_type": "defect_rate",
        "values": [1.2, 1.4, 4.8, 6.1],
        "unit": "percent",
        "target_value": 2.0,
        "statuses": ["normal", "normal", "warning", "critical"],
    },
]


def generate_events() -> pd.DataFrame:
    """Create a reproducible dataset of simulated production measurements."""
    records: list[dict[str, object]] = []
    event_number = 1

    for measurement in MEASUREMENTS:
        values = measurement["values"]
        statuses = measurement["statuses"]

        for index, value in enumerate(values):
            records.append(
                {
                    "event_id": f"EVT-{event_number:03d}",
                    "timestamp": TIMESTAMPS[index],
                    "production_line": measurement["production_line"],
                    "equipment_id": measurement["equipment_id"],
                    "event_type": measurement["event_type"],
                    "value": value,
                    "unit": measurement["unit"],
                    "target_value": measurement["target_value"],
                    "status": statuses[index],
                }
            )
            event_number += 1

    return pd.DataFrame(records)


def main() -> None:
    """Generate and save the synthetic manufacturing dataset."""
    events = generate_events()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(events)} manufacturing events.")
    print(f"Saved dataset to: {OUTPUT_PATH}")
    print("\nEvents by production line:")
    print(events.groupby("production_line").size().to_string())
    print(f"\nTimestamp example: {events.iloc[0]['timestamp']}")


if __name__ == "__main__":
    main()