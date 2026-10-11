"""Generate operational recommendations for production anomalies."""

from typing import Any

RECOMMENDATIONS: dict[str, list[str]] = {
    "none": [
        "Continue routine production monitoring.",
    ],
    "low": [
        "Continue monitoring the affected measurement.",
        "Verify that the configured target remains appropriate.",
    ],
    "medium": [
        "Verify the measurement against the configured operating limits.",
        "Review recent measurements for a sustained deviation.",
        "Document the event for operational review.",
    ],
    "high": [
        "Notify the responsible production operator.",
        "Verify the measurement using an approved inspection procedure.",
        "Review related events for recurring deviations.",
        "Escalate according to the site's operating procedure.",
    ],
    "critical": [
        "Follow the site's approved safety and emergency procedures.",
        "Notify the designated responsible operator immediately.",
        "Verify the event using approved procedures before taking corrective action.",
        "Preserve the event data for subsequent investigation.",
    ],
}


def recommend_actions(
    severity: str,
    event_type: str = "unknown",
) -> dict[str, Any]:
    """Return recommendations appropriate to the supplied severity."""
    if severity not in RECOMMENDATIONS:
        raise ValueError(
            "severity must be none, low, medium, high, or critical."
        )

    if not isinstance(event_type, str) or not event_type.strip():
        event_type = "unknown"

    return {
        "severity": severity,
        "event_type": event_type,
        "recommended_actions": list(RECOMMENDATIONS[severity]),
    }