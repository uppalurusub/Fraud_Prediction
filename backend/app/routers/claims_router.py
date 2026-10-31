"""
FastAPI router for Claims Fraud Analytics with Loguru logging.

Expected service module:
    app/services/claims_service.py

Expected service methods:
    predict_claim(claim: dict, request_id: str | None = None)
    predict_batch(claims: list[dict], request_id: str | None = None)
    get_model_health()
    get_model_evaluation()

Adjust method names or result contracts if your existing service differs.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.config.settings import settings
logger = settings.get_logger()

from pydantic import ValidationError

from app.schemas.claims_schema import (
    BatchClaimPredictionRequest,
    BatchClaimPredictionResponse,
    ClaimPredictionRequest,
    ClaimPredictionResponse,
    HealthResponse,
    ModelEvaluationResponse,
)

router = APIRouter(
    prefix="/claims",
    tags=["Claims Fraud Analytics"],
)


def get_claims_service() -> Any:
    """Construct the claims service through FastAPI dependency injection."""
    try:
        from app.services.claims_service import ClaimsService
    except ImportError:
        logger.exception("claims_service_import_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims service is not configured.",
        )

    try:
        return ClaimsService()
    except Exception:
        logger.exception("claims_service_initialization_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Claims service is unavailable.",
        )


def _resolve_request_id(request_id: str | None) -> str:
    return request_id or str(uuid4())


def _model_to_dict(value: Any) -> dict[str, Any]:
    """Convert a Pydantic v2 model or dict into a JSON-compatible dictionary."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return value
    raise TypeError(
        f"Expected a Pydantic model or dict, received {type(value).__name__}"
    )


def _call_service(service: Any, method_name: str, **kwargs: Any) -> Any:
    """Invoke a service method and map expected failures to HTTP responses."""
    method = getattr(service, method_name, None)
    if not callable(method):
        logger.error(
            "claims_service_method_missing method={}",
            method_name,
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Claims service method '{method_name}' is not implemented.",
        )

    try:
        return method(**kwargs)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "claims_service_validation_error method={} error={}",
            method_name,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception(
            "claims_service_execution_failed method={}",
            method_name,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims analytics request failed.",
        ) from exc


@router.post(
    "/predict",
    response_model=ClaimPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud risk for one claim",
)
def predict_claim(
    payload: ClaimPredictionRequest,
    request: Request,
    service: Any = Depends(get_claims_service),
) -> ClaimPredictionResponse:
    """Predict fraud probability for a single claim."""
    request_id = _resolve_request_id(payload.request_id)
    logger.info(
        "claim_prediction_started request_id={} claim_id={}",
        request_id,
        payload.claim.claim_id,
    )

    try:
        result = _call_service(
            service,
            "predict_claim",
            claim=_model_to_dict(payload.claim),
            request_id=request_id,
        )

        if isinstance(result, ClaimPredictionResponse):
            response = result
        elif isinstance(result, dict) and "prediction" in result:
            response = ClaimPredictionResponse.model_validate(
                {**result, "request_id": result.get("request_id", request_id)}
            )
        else:
            response = ClaimPredictionResponse.model_validate(
                {
                    "request_id": request_id,
                    "status": "SUCCESS",
                    "prediction": result,
                    "message": "Prediction completed successfully",
                }
            )

        logger.info(
            "claim_prediction_completed request_id={} claim_id={}",
            request_id,
            payload.claim.claim_id,
        )
        return response

    except HTTPException:
        raise
    except ValidationError as exc:
        logger.exception(
            "claim_prediction_response_validation_failed request_id={}",
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims service returned an invalid prediction response.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "claim_prediction_failed request_id={} claim_id={}",
            request_id,
            payload.claim.claim_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete claim prediction.",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchClaimPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict fraud risk for multiple claims",
)
def predict_claims_batch(
    payload: BatchClaimPredictionRequest,
    request: Request,
    service: Any = Depends(get_claims_service),
) -> BatchClaimPredictionResponse:
    """Predict fraud risk for a batch of claims."""
    request_id = _resolve_request_id(payload.request_id)
    claim_count = len(payload.claims)

    logger.info(
        "batch_claim_prediction_started request_id={} claim_count={}",
        request_id,
        claim_count,
    )

    try:
        result = _call_service(
            service,
            "predict_batch",
            claims=[_model_to_dict(claim) for claim in payload.claims],
            request_id=request_id,
        )

        if isinstance(result, BatchClaimPredictionResponse):
            response = result
        else:
            if isinstance(result, dict):
                response_data = {
                    **result,
                    "request_id": result.get("request_id", request_id),
                    "total_claims": result.get("total_claims", claim_count),
                }
            else:
                response_data = {
                    "request_id": request_id,
                    "status": "SUCCESS",
                    "total_claims": claim_count,
                    "predictions": result,
                }
            response = BatchClaimPredictionResponse.model_validate(response_data)

        logger.info(
            "batch_claim_prediction_completed request_id={} claim_count={}",
            request_id,
            response.total_claims,
        )
        return response

    except HTTPException:
        raise
    except ValidationError as exc:
        logger.exception(
            "batch_claim_prediction_response_validation_failed request_id={}",
            request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims service returned an invalid batch response.",
        ) from exc
    except Exception as exc:
        logger.exception(
            "batch_claim_prediction_failed request_id={} claim_count={}",
            request_id,
            claim_count,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to complete batch claim prediction.",
        ) from exc


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check claims model readiness",
)
def claims_health(
    service: Any = Depends(get_claims_service),
) -> HealthResponse:
    """Return model/API readiness information from the service."""
    logger.debug("claims_health_check_started")

    result = _call_service(service, "get_model_health")

    try:
        response = (
            result
            if isinstance(result, HealthResponse)
            else HealthResponse.model_validate(result)
        )
    except ValidationError as exc:
        logger.exception("claims_health_response_validation_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims service returned an invalid health response.",
        ) from exc

    logger.info(
        "claims_health_check_completed status={} model_loaded={}",
        response.status,
        response.model_loaded,
    )
    return response


@router.get(
    "/model/evaluation",
    response_model=ModelEvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get recorded model evaluation metrics",
)
def get_model_evaluation(
    service: Any = Depends(get_claims_service),
) -> ModelEvaluationResponse:
    """
    Return recorded model metrics from the service.

    Prefer persisted evaluation results; do not re-evaluate on the test set
    on every API request.
    """
    logger.info("claims_model_evaluation_requested")
    result = _call_service(service, "get_model_evaluation")

    try:
        response = (
            result
            if isinstance(result, ModelEvaluationResponse)
            else ModelEvaluationResponse.model_validate(result)
        )
    except ValidationError as exc:
        logger.exception("claims_model_evaluation_response_validation_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Claims service returned invalid evaluation metrics.",
        ) from exc

    logger.info(
        "claims_model_evaluation_completed model_name={}",
        response.model_name,
    )
    return response