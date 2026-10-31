"""
FastAPI application entry point for Claims Fraud Analytics.

Run from the project root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

This module configures Loguru, registers the claims router, and exposes
basic root and liveness endpoints.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.config.settings import settings
logger = settings.get_logger()



LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = settings.log_dir
LOG_FILE = LOG_DIR / "claims_fraud_analytics.log"


def configure_logging() -> None:
    """Configure console and rotating file logging with Loguru."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()

    log_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
        "request_id={extra[request_id]} | {name}:{function}:{line} | {message}"
    )

    logger.configure(extra={"request_id": "-"})

    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format=log_format,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        str(LOG_FILE),
        level=LOG_LEVEL,
        format=log_format,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info(
        "claims_fraud_api_starting app_name={} environment={}",
        "Claims Fraud Analytics API",
        os.getenv("APP_ENV", "development"),
    )
    yield
    logger.info("claims_fraud_api_stopped")


app = FastAPI(
    title="Claims Fraud Analytics API",
    description=(
        "API for claim fraud-risk scoring and model evaluation. "
        "Predictions are decision-support signals and require appropriate review."
    ),
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next: Any):
    """Log request method/path/status/duration without logging payloads."""
    import time
    from uuid import uuid4

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    started = time.perf_counter()

    with logger.contextualize(request_id=request_id):
        logger.info(
            "http_request_started method={} path={}",
            request.method,
            request.url.path,
        )
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            elapsed = time.perf_counter() - started
            logger.info(
                "http_request_completed method={} path={} status_code={} "
                "duration_seconds={:.4f}",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
            )
            return response
        except Exception:
            logger.exception(
                "http_request_failed method={} path={}",
                request.method,
                request.url.path,
            )
            raise


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unhandled exceptions and return a generic safe response."""
    logger.exception(
        "unhandled_api_exception method={} path={}",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected server error occurred.",
            "request_id": request.headers.get("X-Request-ID"),
        },
    )


@app.get("/", tags=["System"], summary="API information")
def root() -> dict[str, str]:
    """Return basic API metadata."""
    return {
        "application": "Claims Fraud Analytics API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["System"], summary="Application liveness")
def health() -> dict[str, str]:
    """Liveness endpoint; does not force model loading."""
    return {"status": "UP", "application": "claims-fraud-analytics"}


# Register application routers. Router-level prefix is defined in claims_router.py.
try:
    from app.routers.claims_router import router as claims_router

    app.include_router(claims_router, prefix="/api/v1")
    logger.info("claims_router_registered prefix=/api/v1/claims")
except ImportError:
    logger.exception("claims_router_registration_failed")
    raise
