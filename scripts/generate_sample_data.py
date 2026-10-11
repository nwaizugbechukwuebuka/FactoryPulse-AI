"""Generate reproducible synthetic manufacturing events."""

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manufacturing_events.csv"
)

RANDOM_SEED = 42
NUMBER_OF_EVENTS = 600
ANOMALY_RATE = 0.10

PRODUCTION_LINES = [
    "LINE-01",
    "LINE-02",
    "LINE-03",
]

EQUIPMENT = {
    "LINE-01": ["MOTOR-01", "PRESS-01"],
    "LINE-02": ["MOTOR-02", "CONVEYOR-02"],
    "LINE-03": ["MOTOR-03", "ROBOT-03"],
}

TARGETS = {
    "temperature": {
        "target": 85.0,
        "unit": "celsius",
    },
    "production_output": {
        "target": 95.0,
        "unit": "units_per_hour",
    },
    "defect_rate": {
        "target": 2.0,
        "unit": "percent",
    },
}


def generate_measurement(
    event_type: str,
    is_anomaly: bool,
    rng: random.Random,
) -> tuple[float, str]:
    """Generate a measurement and its synthetic severity."""

    target = TARGETS[event_type]["target"]

    if event_type == "temperature":
        if is_anomaly:
            value = rng.uniform(91.0, 102.0)
        else:
            value = rng.gauss(76.0, 3.0)

        value = max(60.0, value)

    elif event_type == "production_output":
        if is_anomaly:
            value = rng.uniform(55.0, 77.0)
        else:
            value = rng.gauss(98.0, 4.0)

        value = max(0.0, value)

    elif event_type == "defect_rate":
        if is_anomaly:
            value = rng.uniform(4.5, 9.0)
        else:
            value = rng.gauss(1.2, 0.35)

        value = max(0.0, value)

    else:
        raise ValueError(f"Unsupported event type: {event_type}")

    if event_type == "temperature":
        if value >= target + 5:
            severity = "critical"
        elif value > target:
            severity = "warning"
        else:
            severity = "normal"

    elif event_type == "production_output":
        if value < target * 0.85:
            severity = "critical"
        elif value < target:
            severity = "warning"
        else:
            severity = "normal"

    else:
        if value >= target * 2:
            severity = "critical"
        elif value > target:
            severity = "warning"
        else:
            severity = "normal"

    return round(value, 2), severity


def generate_events() -> pd.DataFrame:
    """Generate synthetic manufacturing events."""

    rng = random.Random(RANDOM_SEED)

    start_time = datetime(
        2026,
        10,
        1,
        9,
        0,
        tzinfo=UTC,
    )

    event_types = list(TARGETS)
    records = []

    for index in range(NUMBER_OF_EVENTS):
        production_line = PRODUCTION_LINES[
            index % len(PRODUCTION_LINES)
        ]

        equipment_options = EQUIPMENT[production_line]

        equipment_id = equipment_options[
            (index // len(PRODUCTION_LINES))
            % len(equipment_options)
        ]

        event_type = event_types[
            index % len(event_types)
        ]

        is_anomaly = rng.random() < ANOMALY_RATE

        value, severity = generate_measurement(
            event_type,
            is_anomaly,
            rng,
        )

        target = TARGETS[event_type]["target"]
        unit = TARGETS[event_type]["unit"]

        timestamp = start_time + timedelta(minutes=5 * index)

        records.append(
            {
                "event_id": f"EVT-{index + 1:04d}",
                "timestamp": timestamp.isoformat().replace(
                    "+00:00",
                    "Z",
                ),
                "production_line": production_line,
                "equipment_id": equipment_id,
                "event_type": event_type,
                "value": value,
                "unit": unit,
                "target_value": target,
                "status": severity,
                "known_anomaly": is_anomaly,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Generate and save the synthetic dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = generate_events()
    dataframe.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(dataframe)} manufacturing events.")
    print(f"Saved dataset to: {OUTPUT_PATH}")

    print("\nEvents by production line:")
    print(dataframe["production_line"].value_counts().sort_index())

    print("\nEvents by measurement type:")
    print(dataframe["event_type"].value_counts().sort_index())

    print("\nGround-truth anomaly distribution:")
    print(dataframe["known_anomaly"].value_counts().sort_index())

    print("\nSynthetic severity distribution:")
    print(dataframe["status"].value_counts().sort_index())


if __name__ == "__main__":
    main()