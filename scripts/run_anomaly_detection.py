"""Run anomaly detection against the manufacturing CSV dataset."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.anomaly_detection import detect_anomalies
from backend.app.services.data_validation import load_and_validate_csv


def main() -> None:
    """Validate the dataset, detect anomalies, and print a report."""

    dataset = load_and_validate_csv()

    if dataset["invalid_count"] > 0:
        raise ValueError(
            "Dataset contains invalid records. "
            "Resolve validation errors before anomaly detection."
        )

    events = dataset["valid_events"]
    report = detect_anomalies(events)

    print("\nFACTORYPULSE AI — ANOMALY DETECTION REPORT")
    print("=" * 48)
    print(f"Total events:    {report['total_events']}")
    print(f"Normal events:   {report['normal_count']}")
    print(f"Anomalies:       {report['anomaly_count']}")
    print(f"Warnings:        {report['warning_count']}")
    print(f"Critical events: {report['critical_count']}")

    print("\nDETECTED ANOMALIES")
    print("=" * 48)

    if not report["anomalies"]:
        print("No anomalies detected.")
    else:
        for anomaly in report["anomalies"]:
            print(
                f"\nEvent: {anomaly['event_id']}"
                f"\nLine: {anomaly['production_line']}"
                f"\nEquipment: {anomaly['equipment_id']}"
                f"\nMeasurement: {anomaly['event_type']}"
                f"\nObserved: {anomaly['observed_value']} "
                f"{anomaly['unit']}"
                f"\nTarget: {anomaly['target_value']} "
                f"{anomaly['unit']}"
                f"\nDeviation: {anomaly['deviation_pct']}%"
                f"\nSeverity: {anomaly['severity'].upper()}"
                f"\nReason: {anomaly['reason']}"
            )

    output_path = PROJECT_ROOT / "data" / "processed" / (
        "anomaly_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()