"""Rule-based anomaly detection for manufacturing measurements."""

from typing import Any

from backend.app.models.production_event import ProductionEvent


def classify_measurement(
    event: ProductionEvent,
) -> dict[str, Any]:
    """Classify a manufacturing measurement against its target."""

    deviation = event.value - event.target_value

    deviation_pct = (
        deviation / event.target_value
    ) * 100

    absolute_deviation_pct = abs(deviation_pct)

    severity = "normal"
    reason = "Measurement is within the defined operational limits."

    if event.event_type == "temperature":
        if event.value >= event.target_value + 5:
            severity = "critical"
            reason = (
                "Temperature exceeds the target by at least "
                "5 degrees Celsius."
            )
        elif event.value > event.target_value:
            severity = "warning"
            reason = "Temperature exceeds the target."

    elif event.event_type == "production_output":
        if event.value < event.target_value * 0.85:
            severity = "critical"
            reason = (
                "Production output is more than 15% below target."
            )
        elif event.value < event.target_value:
            severity = "warning"
            reason = "Production output is below target."

    elif event.event_type == "defect_rate":
        if event.value >= event.target_value * 2:
            severity = "critical"
            reason = (
                "Defect rate is at least twice the target."
            )
        elif event.value > event.target_value:
            severity = "warning"
            reason = "Defect rate exceeds the target."

    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat(),
        "production_line": event.production_line,
        "equipment_id": event.equipment_id,
        "event_type": event.event_type,
        "observed_value": event.value,
        "target_value": event.target_value,
        "unit": event.unit,
        "deviation": round(deviation, 3),
        "deviation_pct": round(deviation_pct, 2),
        "absolute_deviation_pct": round(
            absolute_deviation_pct,
            2,
        ),
        "severity": severity,
        "is_anomaly": severity != "normal",
        "reason": reason,
    }


def detect_anomalies(
    events: list[ProductionEvent],
) -> dict[str, Any]:
    """Analyze production events and summarize detected anomalies."""

    results = [
        classify_measurement(event)
        for event in events
    ]

    anomalies = [
        result
        for result in results
        if result["is_anomaly"]
    ]

    critical_count = sum(
        result["severity"] == "critical"
        for result in anomalies
    )

    warning_count = sum(
        result["severity"] == "warning"
        for result in anomalies
    )

    return {
        "total_events": len(results),
        "anomaly_count": len(anomalies),
        "normal_count": len(results) - len(anomalies),
        "warning_count": warning_count,
        "critical_count": critical_count,
        "anomalies": anomalies,
        "results": results,
    }