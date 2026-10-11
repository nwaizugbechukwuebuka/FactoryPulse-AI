"""Validated production-event model for FactoryPulse AI."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EventType = Literal[
    "temperature",
    "production_output",
    "defect_rate",
]

EventStatus = Literal[
    "normal",
    "warning",
    "critical",
]

EXPECTED_UNITS = {
    "temperature": "celsius",
    "production_output": "units_per_hour",
    "defect_rate": "percent",
}


class ProductionEvent(BaseModel):
    """Represent one validated manufacturing measurement."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=50)
    timestamp: datetime
    production_line: str = Field(min_length=1, max_length=50)
    equipment_id: str = Field(min_length=1, max_length=50)

    event_type: EventType
    value: float = Field(ge=0, allow_inf_nan=False)
    unit: str = Field(min_length=1, max_length=30)
    target_value: float = Field(gt=0, allow_inf_nan=False)
    status: EventStatus

    @field_validator("event_id", "production_line", "equipment_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        """Reject identifiers containing only whitespace."""

        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Identifier must not be empty or whitespace.")

        return cleaned

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        """Require a timestamp with timezone information."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must include timezone information.")

        return value

    @model_validator(mode="after")
    def validate_measurement(self) -> "ProductionEvent":
        """Ensure the measurement uses the expected unit."""

        expected_unit = EXPECTED_UNITS[self.event_type]

        if self.unit != expected_unit:
            raise ValueError(
                f"Event type '{self.event_type}' requires unit "
                f"'{expected_unit}', not '{self.unit}'."
            )

        if self.event_type == "defect_rate" and self.value > 100:
            raise ValueError("Defect rate cannot exceed 100 percent.")

        return self