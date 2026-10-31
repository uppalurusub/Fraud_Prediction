"""
Service layer for Claims Fraud Analytics.

This layer coordinates request validation, inference implementation calls,
risk-band assignment, and response construction. Model preprocessing and
inference should remain in app/implementations/claims_impl.py.

Expected implementation contract:
    from app.implementations.claims_impl import ClaimsImplementation

    implementation.predict(claim_data: dict) -> dict

The predict result may contain:
    fraud_probability (preferred) or probability
    predicted_fraud (optional)
    model_name (optional)
    model_version (optional)

Optional implementation methods:
    get_model_health() -> dict
    get_model_evaluation() -> dict
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from app.config.settings import settings
logger = settings.get_logger()

from app.schemas.claims_schema import (
    BatchClaimPredictionResponse,
    ClaimPrediction,
    ClaimPredictionResponse,
    HealthResponse,
    ModelEvaluationResponse,
    ModelMetrics,
    RiskLevel,
)
from app.implementations.claims_implementation import ClaimsImplementation
import mlflow

# =====================================================
# MLFLOW
# =====================================================

mlflow.set_tracking_uri(
    #"http://127.0.0.1/5000"
    "http://localhost:5000"
)

EXPERIMENT_NAME = "Fraud_Prediction"

mlflow.set_experiment(
    EXPERIMENT_NAME
)


class ClaimsServiceError(RuntimeError):
    """Raised when the claims service cannot complete an operation."""


class ClaimsService:
    """Orchestrates claims fraud prediction operations."""

    def __init__(self, implementation: Any | None = None) -> None:
        self._implementation = implementation
        self._implementation_initialized = implementation is not None
        self.low_risk_threshold = self._read_threshold(
            "CLAIMS_LOW_RISK_THRESHOLD", 0.50
        )
        self.high_risk_threshold = self._read_threshold(
            "CLAIMS_HIGH_RISK_THRESHOLD", 0.80
        )

        if self.low_risk_threshold >= self.high_risk_threshold:
            raise ValueError(
                "CLAIMS_LOW_RISK_THRESHOLD must be lower than "
                "CLAIMS_HIGH_RISK_THRESHOLD"
            )

    @staticmethod
    def _read_threshold(name: str, default: float) -> float:
        raw_value = os.getenv(name, str(default))
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a numeric value") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return value

    def _get_implementation(self) -> Any:
        """Lazy-load the implementation so API imports do not load the model."""
        if self._implementation_initialized:
            return self._implementation

        try:
            

            self._implementation = ClaimsImplementation()
            self._implementation_initialized = True
            logger.info("claims_implementation_initialized")
            return self._implementation
        except Exception as exc:
            logger.exception("claims_implementation_initialization_failed")
            raise ClaimsServiceError(
                "Claims inference implementation could not be initialized."
            ) from exc

    @staticmethod
    def _flatten_claim(claim: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten additional_features into the model input.

        The implementation must still select the exact training feature columns
        and reject/handle unknown columns consistently with the training pipeline.
        """
        features = dict(claim)
        additional = features.pop("additional_features", None) or {}
        for key, value in additional.items():
            if key in features and features[key] != value:
                raise ValueError(
                    f"Feature '{key}' was provided both directly and in "
                    "additional_features with different values."
                )
            features.setdefault(key, value)
        return features

    @staticmethod
    def _probability(result: Any) -> float:
        if not isinstance(result, dict):
            raise ClaimsServiceError(
                "Claims implementation must return a dictionary."
            )

        raw_probability = result.get(
            "fraud_probability",
            result.get("probability", result.get("fraud_score")),
        )
        if raw_probability is None:
            raise ClaimsServiceError(
                "Implementation result is missing fraud_probability."
            )

        try:
            probability = float(raw_probability)
        except (TypeError, ValueError) as exc:
            raise ClaimsServiceError(
                "Implementation returned a non-numeric fraud probability."
            ) from exc

        if not 0.0 <= probability <= 1.0:
            raise ClaimsServiceError(
                "Implementation fraud probability must be between 0 and 1."
            )
        return probability

    def _risk_level(self, probability: float) -> RiskLevel:
        if probability >= self.high_risk_threshold:
            return RiskLevel.HIGH
        if probability >= self.low_risk_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def predict_claim(
        self,
        claim: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a normalized prediction payload for one claim."""
        started = time.perf_counter()
        claim_id = str(claim.get("claim_id", "UNKNOWN"))
        logger.info(
            "claims_service_prediction_started request_id={} claim_id={}",
            request_id or "-",
            claim_id,
        )

        try:
            model_input = self._flatten_claim(claim)
            implementation = self._get_implementation()
            predict_method = getattr(implementation, "predict", None)

            if not callable(predict_method):
                raise ClaimsServiceError(
                    "ClaimsImplementation must expose predict(claim_data)."
                )

            
            with mlflow.start_run(
                run_name=
                f"predict_claim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):

                result = predict_method(model_input)
                probability = self._probability(result)

                # Use the configured service threshold for consistent API behavior.
                predicted_fraud = probability >= self.high_risk_threshold
                prediction = ClaimPrediction(
                    claim_id=claim_id,
                    predicted_fraud=predicted_fraud,
                    fraud_probability=probability,
                    risk_level=self._risk_level(probability),
                    model_name=(
                        result.get("model_name")
                        or os.getenv("CLAIMS_MODEL_NAME")
                        or type(getattr(implementation, "model", implementation)).__name__
                    ),
                    
                    prediction_timestamp=datetime.now(timezone.utc),
                )

                elapsed = time.perf_counter() - started

                mlflow.log_param(
                    "input_text",
                    model_input
                )

                mlflow.log_param(
                    "predict_claim",
                    prediction.model_dump(mode="json")
                )

                

                mlflow.log_metric(
                    "execution_time",
                    elapsed
                )



                mlflow.set_tag(
                    "source",
                    "FastAPI"
                )

                mlflow.set_tag(
                    "endpoint",
                    "/predict_claim"
                )

            logger.info(
                "claims_service_prediction_completed request_id={} claim_id={} "
                "predicted_fraud={} risk_level={} execution_time_seconds={:.4f}",
                request_id or "-",
                claim_id,
                predicted_fraud,
                prediction.risk_level.value,
                elapsed,
            )
            return prediction.model_dump(mode="json")

        except (ValueError, ClaimsServiceError):
            logger.warning(
                "claims_service_prediction_rejected request_id={} claim_id={}",
                request_id or "-",
                claim_id,
            )
            raise
        except Exception as exc:
            logger.exception(
                "claims_service_prediction_failed request_id={} claim_id={}",
                request_id or "-",
                claim_id,
            )
            raise ClaimsServiceError(
                "The fraud prediction implementation failed."
            ) from exc

    def predict_batch(
        self,
        claims: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Predict each claim and return a normalized batch response."""
        if not claims:
            raise ValueError("claims must contain at least one claim")

        started = time.perf_counter()
        logger.info(
            "claims_service_batch_started request_id={} claim_count={}",
            request_id or "-",
            len(claims),
        )

        predictions: list[dict[str, Any]] = []
        for claim in claims:
            predictions.append(
                self.predict_claim(claim=claim, request_id=request_id)
            )

        with mlflow.start_run(
                run_name=
                f"predict_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):

            response = BatchClaimPredictionResponse(
                request_id=request_id,
                status="SUCCESS",
                total_claims=len(claims),
                predictions=[
                    ClaimPrediction.model_validate(item) for item in predictions
                ],
                message="Batch predictions completed successfully",
            )

            elapsed = time.perf_counter() - started


            mlflow.log_param(
                "input_text",
                claims
            )
            
            mlflow.log_param(
                "predict_batch",
                response
            )
            
            mlflow.log_metric(
                "execution_time",
                elapsed
            )
            
            mlflow.set_tag(
                "source",
                "FastAPI"
            )
            
            mlflow.set_tag(
                "endpoint",
                "/predict_batch"
            )



        logger.info(
            "claims_service_batch_completed request_id={} claim_count={} "
            "execution_time_seconds={:.4f}",
            request_id or "-",
            len(claims),
            time.perf_counter() - started,
        )
        return response.model_dump(mode="json")

    def get_model_health(self) -> dict[str, Any]:
        """Report whether the inference implementation can be initialized."""
        try:
            implementation = self._get_implementation()
            health_method = getattr(implementation, "get_model_health", None)

            if callable(health_method):
                result = health_method()
                if isinstance(result, HealthResponse):
                    return result.model_dump(mode="json")
                if isinstance(result, dict):
                    return HealthResponse.model_validate(result).model_dump(
                        mode="json"
                    )

            model = getattr(implementation, "model", None)
            model_loaded = model is not None
            response = HealthResponse(
                status="READY" if model_loaded else "DEGRADED",
                model_loaded=model_loaded,
                model_name=(
                    os.getenv("CLAIMS_MODEL_NAME")
                    or (type(model).__name__ if model is not None else None)
                ),
                #model_version=os.getenv("CLAIMS_MODEL_VERSION"),
                timestamp=datetime.now(timezone.utc),
            )
            return response.model_dump(mode="json")

        except Exception:
            logger.exception("claims_model_health_check_failed")
            return HealthResponse(
                status="UNAVAILABLE",
                model_loaded=False,
                model_name=os.getenv("CLAIMS_MODEL_NAME"),
                model_version=os.getenv("CLAIMS_MODEL_VERSION"),
                timestamp=datetime.now(timezone.utc),
            ).model_dump(mode="json")

    def get_model_evaluation(self) -> dict[str, Any]:
        """
        Return previously recorded evaluation metrics.

        The implementation should load metrics from a trusted persisted report
        or MLflow evaluation run. This method intentionally does not evaluate
        against a test dataset during an API request.
        """
        implementation = self._get_implementation()
        evaluation_method = getattr(implementation, "get_model_evaluation", None)

        if not callable(evaluation_method):
            logger.warning("claims_model_evaluation_not_configured")
            raise NotImplementedError(
                "Configure ClaimsImplementation.get_model_evaluation() to "
                "return persisted model evaluation metrics."
            )

        result = evaluation_method()
        if isinstance(result, ModelEvaluationResponse):
            return result.model_dump(mode="json")
        if not isinstance(result, dict):
            raise ClaimsServiceError(
                "get_model_evaluation() must return a dictionary or "
                "ModelEvaluationResponse."
            )

        # Accept either a full response or a simple metrics dictionary.
        if "metrics" not in result:
            metric_fields = set(ModelMetrics.model_fields.keys())
            metrics_payload = {
                key: value for key, value in result.items() if key in metric_fields
            }
            result = {
                "status": result.get("status", "SUCCESS"),
                "model_name": (
                    result.get("model_name")
                    or os.getenv("CLAIMS_MODEL_NAME")
                    or "claims-fraud-risk-classifier"
                ),
                "model_version": result.get("model_version")
                or os.getenv("CLAIMS_MODEL_VERSION"),
                "dataset_name": result.get("dataset_name"),
                "evaluation_timestamp": result.get(
                    "evaluation_timestamp", datetime.now(timezone.utc)
                ),
                "metrics": metrics_payload,
                "notes": result.get("notes", []),
            }

        validated = ModelEvaluationResponse.model_validate(result)
        logger.info(
            "claims_model_evaluation_loaded model_name={}",
            validated.model_name,
        )
        return validated.model_dump(mode="json")