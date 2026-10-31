"""
Inference implementation for Claims Fraud Analytics.

Responsibilities:
- Load a serialized scikit-learn estimator or Pipeline.
- Optionally load a fitted preprocessing transformer.
- Map API fields to the common claims.csv feature names.
- Align inference data with the fitted model's expected columns.
- Return fraud probability and model metadata.
- Use Loguru for application logs.

Environment variables:
    CLAIMS_MODEL_PATH=models/claims_fraud_model.joblib
    CLAIMS_PREPROCESSOR_PATH=models/claims_preprocessor.joblib  # optional
    CLAIMS_EVALUATION_PATH=models/claims_model_evaluation.json  # optional
    CLAIMS_MODEL_NAME=claims-fraud-risk-classifier
    CLAIMS_MODEL_VERSION=1

IMPORTANT:
The model and preprocessing artifacts must come from the same training run.
If your training script saves a single sklearn Pipeline, configure only
CLAIMS_MODEL_PATH and leave CLAIMS_PREPROCESSOR_PATH unset.

This implementation intentionally does not invent missing feature values or
fit preprocessing on live requests. Align the field aliases and feature
engineering with the exact training pipeline before production use.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from app.config.settings import settings
logger = settings.get_logger()

import mlflow
from mlflow.tracking import MlflowClient

ARTIFACT_DIR = settings.artifact_dir



class ClaimsImplementationError(RuntimeError):
    """Raised when model loading or prediction cannot be completed."""


class ClaimsImplementation:
    """Load the trained model and generate fraud probabilities."""

    # API schema field -> canonical training column.
    FIELD_ALIASES = {
        "patient_age": "age",
        "patient_gender": "gender",
        "service_date": "claim_date",
        "submission_date": "submission_date",
        "procedure_count": "num_procedures",
        "fraud_label": "is_fraud",
    }

    # Identifiers and target/outcome columns must not be model features unless
    # the training pipeline explicitly used them (normally it should not).
    TARGET_COLUMNS = {
        "fraud_label",
        "is_fraud",
        "fraud_amount",
        "fraud_probability",
        "fraud_score",
        "predicted_fraud",
    }
    IDENTIFIER_COLUMNS = {"claim_id"}

    def __init__(
        self,
        model_path: str | Path | None = None,
        preprocessor_path: str | Path | None = None,
        evaluation_path: str | Path | None = None,
        model: Any | None = None,
        preprocessor: Any | None = None,
    ) -> None:
        self.model_path = Path(
            f"{ARTIFACT_DIR}/best_model.pkl"
        )
        raw_preprocessor_path = (
            f"{ARTIFACT_DIR}/preprocessor.pkl"
        )
        self.preprocessor_path = (
            Path(raw_preprocessor_path) if raw_preprocessor_path else None
        )
        raw_evaluation_path = (
            evaluation_path or os.getenv("CLAIMS_EVALUATION_PATH")
        )
        self.evaluation_path = (
            Path(raw_evaluation_path) if raw_evaluation_path else None
        )

        self.model = model
        self.preprocessor = preprocessor
        self._model_loaded = model is not None
        self._preprocessor_loaded = preprocessor is not None
        self.model_name = os.getenv(
            "CLAIMS_MODEL_NAME", "claims-fraud-risk-classifier"
        )
        self.model_version = os.getenv("CLAIMS_MODEL_VERSION")

        if not self._model_loaded:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized model and optional fitted preprocessing artifact."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Claims model artifact not found: {self.model_path}. "
                "Set CLAIMS_MODEL_PATH to the correct .joblib artifact."
            )

        try:
            self.model = joblib.load(self.model_path)
            self._model_loaded = True
            logger.info(
                "claims_model_loaded path={} estimator_type={}",
                str(self.model_path),
                type(self.model).__name__,
            )

            # If the model is already an sklearn Pipeline, do not apply a
            # separate preprocessor to avoid double transformation.
            is_pipeline = hasattr(self.model, "steps") and hasattr(
                self.model, "named_steps"
            )
            if self.preprocessor_path and not is_pipeline:
                if not self.preprocessor_path.is_file():
                    raise FileNotFoundError(
                        "Claims preprocessor artifact not found: "
                        f"{self.preprocessor_path}"
                    )
                self.preprocessor = joblib.load(self.preprocessor_path)
                self._preprocessor_loaded = True
                logger.info(
                    "claims_preprocessor_loaded path={} transformer_type={}",
                    str(self.preprocessor_path),
                    type(self.preprocessor).__name__,
                )
        except Exception as exc:
            logger.exception("claims_artifact_loading_failed")
            raise ClaimsImplementationError(
                "Could not load the claims model/preprocessor artifacts."
            ) from exc

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Normalize date/time and NumPy scalar values for pandas inputs."""
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _canonicalize_fields(cls, claim_data: dict[str, Any]) -> dict[str, Any]:
        """Map schema aliases to the common claims dataset column names."""
        row = {
            key: cls._json_safe(value)
            for key, value in claim_data.items()
            if value is not None
        }

        additional = row.pop("additional_features", None) or {}
        if not isinstance(additional, dict):
            raise ValueError("additional_features must be an object")

        for key, value in additional.items():
            if key in row and row[key] != value:
                raise ValueError(
                    f"Feature '{key}' was supplied both directly and in "
                    "additional_features with conflicting values."
                )
            row.setdefault(key, cls._json_safe(value))

        for source, destination in cls.FIELD_ALIASES.items():
            if source in row:
                if destination in row and row[destination] != row[source]:
                    raise ValueError(
                        f"Conflicting values were supplied for '{source}' "
                        f"and '{destination}'."
                    )
                row.setdefault(destination, row[source])
                if source != destination:
                    row.pop(source, None)

        # Map claim_amount aliases only when the API supplied a supported
        # amount field. Do not derive financial amounts or target labels.
        if "claim_amount" not in row and "billed_amount" in row:
            row["claim_amount"] = row["billed_amount"]

        return row

    @staticmethod
    def _expected_columns(transformer_or_model: Any) -> list[str] | None:
        columns = getattr(transformer_or_model, "feature_names_in_", None)
        if columns is not None:
            return [str(column) for column in columns]
        return None

    @staticmethod
    def _is_sklearn_pipeline(model: Any) -> bool:
        return hasattr(model, "steps") and hasattr(model, "named_steps")

    def _prepare_features(self, claim_data: dict[str, Any]) -> pd.DataFrame:
        """Build one-row input and align it to the fitted artifact's columns."""
        row = self._canonicalize_fields(claim_data)

        prohibited = self.TARGET_COLUMNS.intersection(row)
        if prohibited:
            raise ValueError(
                "Prediction input must not contain target/outcome fields: "
                f"{sorted(prohibited)}"
            )

        # Do not pass claim_id to a model unless the fitted preprocessor
        # explicitly requires it. IDs are identifiers, not predictive signals.
        row.pop("claim_id", None)
        frame = pd.DataFrame([row])

        # When a separate fitted preprocessor exists, align raw inputs to its
        # original training columns. This avoids sending arbitrary extra keys.
        if self.preprocessor is not None:
            expected = self._expected_columns(self.preprocessor)
            if expected:
                missing = [column for column in expected if column not in frame.columns]
                if missing:
                    raise ValueError(
                        "Missing required raw model features: "
                        f"{missing}. Confirm the API schema and training columns."
                    )
                frame = frame.loc[:, expected]
            return frame

        # A model/pipeline that records input columns can be aligned safely.
        expected = self._expected_columns(self.model)
        if expected:
            missing = [column for column in expected if column not in frame.columns]
            if missing:
                raise ValueError(
                    "Missing required model features: "
                    f"{missing}. Supply them in the request or additional_features."
                )
            frame = frame.loc[:, expected]

        return frame

    def _transform(self, frame: pd.DataFrame) -> Any:
        """Apply the fitted preprocessing transformer, if configured."""
        if self.preprocessor is None:
            return frame
        try:
            return self.preprocessor.transform(frame)
        except Exception as exc:
            logger.exception("claims_preprocessing_failed")
            raise ClaimsImplementationError(
                "Claims preprocessing failed. Check the fitted transformer "
                "and inference feature columns."
            ) from exc

    @staticmethod
    def _positive_class_index(model: Any) -> int:
        classes = getattr(model, "classes_", None)
        if classes is None:
            raise ClaimsImplementationError(
                "Loaded estimator does not expose classes_; cannot identify "
                "the positive fraud class."
            )

        for index, label in enumerate(classes):
            if str(label).strip().lower() in {"1", "true", "fraud"}:
                return index

        raise ClaimsImplementationError(
            f"Positive fraud class 1 was not found in model classes_: {list(classes)}"
        )

    def _fraud_probability(self, model_input: Any) -> float:
        """Get class-1 probability; do not treat hard labels as probabilities."""
        predict_proba = getattr(self.model, "predict_proba", None)
        if not callable(predict_proba):
            raise ClaimsImplementationError(
                "The loaded estimator does not implement predict_proba(). "
                "Use a probability-capable classifier or add a calibrated "
                "probability adapter in the training pipeline."
            )

        try:
            probabilities = np.asarray(predict_proba(model_input), dtype=float)
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ClaimsImplementationError(
                    "Model predict_proba returned an unexpected output shape."
                )
            positive_index = self._positive_class_index(self.model)
            if positive_index >= probabilities.shape[1]:
                raise ClaimsImplementationError(
                    "Positive class index is outside the probability output."
                )
            probability = float(probabilities[0, positive_index])
        except ClaimsImplementationError:
            raise
        except Exception as exc:
            logger.exception("claims_probability_prediction_failed")
            raise ClaimsImplementationError(
                "The fraud probability could not be calculated."
            ) from exc

        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ClaimsImplementationError(
                f"Model returned an invalid fraud probability: {probability}"
            )
        return probability

    def predict(self, claim_data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict one claim.

        Returns the keys expected by ClaimsService:
            fraud_probability, predicted_fraud, model_name, model_version,
            execution_time_seconds
        """
        started = time.perf_counter()
        claim_id = str(claim_data.get("claim_id", "UNKNOWN"))
        logger.info("claims_inference_started claim_id={}", claim_id)

        try:
            if not isinstance(claim_data, dict):
                raise ValueError("claim_data must be a dictionary")

            frame = self._prepare_features(claim_data)
            model_input = frame if self._is_sklearn_pipeline(self.model) else self._transform(frame)

            # If a separate preprocessor returns a matrix but the fitted model
            # expects named columns, preserve those names where available.
            model_columns = self._expected_columns(self.model)
            if (
                self.preprocessor is not None
                and model_columns
                and not isinstance(model_input, pd.DataFrame)
            ):
                if len(model_columns) != model_input.shape[1]:
                    raise ClaimsImplementationError(
                        "Transformed feature count does not match the model's "
                        "expected feature count."
                    )
                model_input = pd.DataFrame(model_input, columns=model_columns)

            probability = self._fraud_probability(model_input)
            elapsed = time.perf_counter() - started

            response = {
                "fraud_probability": probability,
                "predicted_fraud": bool(probability >= 0.5),
                "model_name": self.model_name,
                #"model_version": self.model_version,
                "execution_time_seconds": round(elapsed, 6),
            }

            logger.info(
                "claims_inference_completed claim_id={} fraud_probability={:.6f} "
                "execution_time_seconds={:.4f}",
                claim_id,
                probability,
                elapsed,
            )
            return response

        except (ValueError, ClaimsImplementationError):
            logger.warning("claims_inference_rejected claim_id={}", claim_id)
            raise
        except Exception as exc:
            logger.exception("claims_inference_failed claim_id={}", claim_id)
            raise ClaimsImplementationError(
                "Unexpected error during claims inference."
            ) from exc

    def get_model_health(self) -> dict[str, Any]:
        """Return a health summary without exposing local filesystem details."""
        loaded = self.model is not None and self._model_loaded
        return {
            "status": "READY" if loaded else "UNAVAILABLE",
            "model_loaded": loaded,
            "model_name": self.model_name,
            #"model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }




    def get_model_evaluation(self) -> dict:
        """Retrieve persisted model evaluation metrics without inventing values."""

        project_root = Path(__file__).resolve().parents[3]

        evaluation_path = Path(
            os.getenv(
                "CLAIMS_EVALUATION_PATH",
                str(project_root / "artifacts" / "claims_model_evaluation.json"),
            )
        )

        csv_path = project_root / "artifacts" / "validation_model_comparison.csv"

        # 1. Read persisted JSON evaluation metrics, when available.
        if evaluation_path.is_file():
            logger.info(
                "claims_evaluation_json_loading path={}",
                str(evaluation_path),
            )
            with evaluation_path.open("r", encoding="utf-8") as file:
                result = json.load(file)

            if not isinstance(result, dict):
                raise ValueError("Evaluation JSON must contain a JSON object.")

            return result

        # 2. Use the model-comparison CSV generated by model training.
        if csv_path.is_file():
            logger.info(
                "claims_evaluation_csv_loading path={}",
                str(csv_path),
            )

            df = pd.read_csv(csv_path)

            if df.empty:
                logger.warning("claims_evaluation_csv_empty path={}", str(csv_path))
            else:
                metric_columns = [
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall_sensitivity",
                    "specificity",
                    "f1",
                    "f2",
                    "roc_auc",
                    "pr_auc",
                    "brier_score",
                    "log_loss",
                    "threshold",
                    "true_positive",
                    "true_negative",
                    "false_positive",
                    "false_negative",
                ]

                available_metrics = [
                    column for column in metric_columns if column in df.columns
                ]

                if not available_metrics:
                    raise ValueError(
                        "The evaluation CSV contains no recognized metric columns."
                    )

                # Select the highest PR-AUC model when that metric is available.
                if "pr_auc" in df.columns:
                    df = df.sort_values("pr_auc", ascending=False, na_position="last")

                row = df.iloc[0]
                metrics = {}

                for column in available_metrics:
                    value = row[column]
                    if pd.notna(value):
                        metrics[column] = value.item() if hasattr(value, "item") else value

                model_name = (
                    str(row["Model"])
                    if "Model" in df.columns and pd.notna(row["Model"])
                    else "unknown"
                )

                logger.info(
                    "claims_evaluation_csv_loaded model_name={} metric_count={}",
                    model_name,
                    len(metrics),
                )

                return {
                    "status": "SUCCESS",
                    "model_name": model_name,
                    "metrics": metrics,
                    "source": str(csv_path),
                }

        # 3. Try retrieving the best available MLflow run.
        try:
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://localhost:5000",
            )
            experiment_name = os.getenv(
                "MLFLOW_EXPERIMENT_NAME",
                "Claims_Frad_Analysis",
            )

            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient(tracking_uri=tracking_uri)

            experiment = client.get_experiment_by_name(experiment_name)

            if experiment is not None:
                runs = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.pr_auc DESC"],
                    max_results=1,
                )

                if runs:
                    run = runs[0]
                    logger.info(
                        "claims_evaluation_mlflow_loaded run_id={} experiment={}",
                        run.info.run_id,
                        experiment_name,
                    )

                    return {
                        "status": "SUCCESS",
                        "model_name": run.data.tags.get("mlflow.runName", "unknown"),
                        "run_id": run.info.run_id,
                        "metrics": dict(run.data.metrics),
                        "source": "mlflow",
                    }

        except Exception:
            logger.exception("claims_evaluation_mlflow_lookup_failed")

        logger.warning("claims_evaluation_unavailable")

        return {
            "status": "NOT_AVAILABLE",
            "model_name": None,
            "metrics": {},
            "message": (
                "No persisted evaluation results or MLflow evaluation run was found. "
                "Run model training and evaluation first."
            ),
        }