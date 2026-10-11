"""FastAPI endpoints for anomaly intelligence."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from backend.app.services.anomaly_intelligence import (
    build_anomaly_intelligence,
)

router = APIRouter(
    prefix="/api/v1/intelligence",
    tags=["Anomaly Intelligence"],
)


class IntelligenceRequest(BaseModel):
    """Request containing event data and detector results."""

    model_config = ConfigDict(extra="forbid")

    event: dict[str, Any] = Field(
        ...,
        description="Production event data.",
    )

    is_anomaly: bool

    detection_method: str = Field(
        ...,
        description=(
            "Detector identifier: rule_based, isolation_forest, or hybrid."
        ),
    )

    rule_result: dict[str, Any] | None = None

    ml_prediction: int | None = Field(
        default=None,
        ge=0,
        le=1,
    )


@router.post("/analyze")
def analyze_anomaly(
    request: IntelligenceRequest,
) -> dict[str, Any]:
    """Return an explanation, severity, and recommendations."""
    try:
        return build_anomaly_intelligence(
            event=request.event,
            is_anomaly=request.is_anomaly,
            detection_method=request.detection_method,
            rule_result=request.rule_result,
            ml_prediction=request.ml_prediction,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc