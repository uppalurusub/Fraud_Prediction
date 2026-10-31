"""
Pydantic schemas for the Claims Fraud Analytics API.

Compatible with Pydantic v2 and FastAPI.

Responsibilities:
- Validate individual claim prediction requests.
- Validate batch prediction requests.
- Standardize fraud prediction responses.
- Represent model evaluation metrics.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class ClaimStatus(str, Enum):
    """Business status of a claim."""

    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


class RiskLevel(str, Enum):
    """Risk band assigned by the prediction implementation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class APIBaseSchema(BaseModel):
    """Common Pydantic configuration for API schemas."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ClaimFeatures(APIBaseSchema):
    """
    Claim fields used as model features.

    Keep this schema aligned with the feature columns expected by the
    preprocessing pipeline and trained model. Do not include the target
    label (fraud_label) or post-outcome fields that would cause leakage.
    """

    claim_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique claim identifier.",
        examples=["CLM-000001"],
    )
    patient_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Pseudonymous patient identifier, if available.",
    )
    provider_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Provider identifier, if available.",
    )
    claim_type: str = Field(..., min_length=1, max_length=100)
    diagnosis_code: str | None = Field(default=None, max_length=50)
    procedure_code: str | None = Field(default=None, max_length=50)
    claim_amount: float = Field(..., ge=0, le=100_000_000)
    service_date: date | None = None
    submission_date: date | None = None
    provider_type: str | None = Field(default=None, max_length=100)
    place_of_service: str | None = Field(default=None, max_length=100)
    patient_age: int | None = Field(default=None, ge=0, le=120)
    patient_gender: str | None = Field(default=None, max_length=50)
    procedure_count: int | None = Field(default=None, ge=0)
    diagnosis_count: int | None = Field(default=None, ge=0)
    prior_claim_count: int | None = Field(default=None, ge=0)
    prior_fraud_flag: int | None = Field(default=None, ge=0, le=1)
    claim_status: ClaimStatus | None = None
    additional_features: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional extra feature values. Only use keys explicitly supported "
            "by the configured preprocessing/model pipeline."
        ),
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "ClaimFeatures":
        if (
            self.service_date is not None
            and self.submission_date is not None
            and self.submission_date < self.service_date
        ):
            raise ValueError("submission_date cannot be earlier than service_date")
        return self


class ClaimPredictionRequest(APIBaseSchema):
    """Request body for predicting fraud risk for one claim."""

    claim: ClaimFeatures
    request_id: str | None = Field(default=None, min_length=1, max_length=100)


class BatchClaimPredictionRequest(APIBaseSchema):
    """Request body for predicting fraud risk for multiple claims."""

    claims: list[ClaimFeatures] = Field(..., min_length=1, max_length=1000)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_claim_ids(self) -> "BatchClaimPredictionRequest":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique within a batch")
        return self


class ClaimPrediction(APIBaseSchema):
    """Prediction result for one claim."""

    claim_id: str
    predicted_fraud: bool = Field(
        ...,
        description="True when the configured classification threshold predicts fraud.",
    )
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    model_name: str | None = None
    #model_version: str | None = None
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ClaimPredictionResponse(APIBaseSchema):
    """Standard API response for a single claim prediction."""

    request_id: str | None = None
    status: str = Field(default="SUCCESS")
    prediction: ClaimPrediction
    message: str | None = None


class BatchClaimPredictionResponse(APIBaseSchema):
    """Standard API response for batch claim predictions."""

    request_id: str | None = None
    status: str = Field(default="SUCCESS")
    total_claims: PositiveInt
    predictions: list[ClaimPrediction]
    message: str | None = None

    @model_validator(mode="after")
    def validate_prediction_count(self) -> "BatchClaimPredictionResponse":
        if self.total_claims != len(self.predictions):
            raise ValueError("total_claims must equal len(predictions)")
        return self


class ModelMetrics(APIBaseSchema):
    """Evaluation metrics for a binary fraud classifier."""

    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    pr_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    true_positives: int | None = Field(default=None, ge=0)
    false_positives: int | None = Field(default=None, ge=0)
    true_negatives: int | None = Field(default=None, ge=0)
    false_negatives: int | None = Field(default=None, ge=0)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelEvaluationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "SUCCESS"
    model_name: str | None = None
    run_id: str | None = None
    metrics: dict[str, float | int] = Field(default_factory=dict)
    source: str | None = None
    message: str | None = None


class HealthResponse(APIBaseSchema):
    """Response schema for a lightweight model/API health endpoint."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
