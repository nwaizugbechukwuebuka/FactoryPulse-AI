"""FastAPI application entry point for FactoryPulse-AI."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.app.api.routes.intelligence import router as intelligence_router

app = FastAPI(
    title="FactoryPulse-AI",
    description=(
        "AI-powered factory operations monitoring with anomaly detection, "
        "evidence-based explanations, severity assessment, and "
        "operational recommendations."
    ),
    version="0.4.0",
)


@app.get(
    "/",
    tags=["Health"],
    summary="Application information",
)
def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "name": "FactoryPulse-AI",
        "version": "0.4.0",
        "status": "running",
        "message": "Factory operations intelligence API is available.",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
)
def health_check() -> dict[str, str]:
    """Report whether the API process is responding."""
    return {
        "status": "healthy",
        "service": "factorypulse-ai",
    }


@app.get(
    "/ready",
    tags=["Health"],
    summary="Readiness check",
)
def readiness_check() -> JSONResponse:
    """Report basic application readiness."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "service": "factorypulse-ai",
        },
    )


app.include_router(intelligence_router)