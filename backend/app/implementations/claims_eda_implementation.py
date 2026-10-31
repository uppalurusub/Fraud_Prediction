"""
Claims Exploratory Data Analysis (EDA) implementation.

Responsibilities:
- Load the configured claims dataset.
- Produce dataset/profile summaries and data-quality checks.
- Compute numeric and categorical summaries.
- Inspect target-class distribution when a target column is present.
- Return JSON-serializable dictionaries suitable for a FastAPI service/router.

This module does not train a model or modify the source dataset.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from app.config.settings import settings
logger = settings.get_logger()


class ClaimsEDAImplementation:
    """Run read-only EDA against a claims dataset."""

    DEFAULT_TARGET_COLUMNS = ("is_fraud", "fraud", "fraud_flag", "is_fraudulent")
    DEFAULT_DATE_COLUMNS = (
        "claim_date",
        "service_date",
        "submission_date",
        "created_at",
        "updated_at",
    )

    def __init__(
        self,
        data_path: str | Path | None = None,
        target_column: str | None = None,
        sample_size: int = 5,
    ) -> None:
        """
        Args:
            data_path: CSV path. If omitted, CLAIMS_DATA_PATH is checked, then
                artifacts/claims.csv and data/claims.csv relative to the project root.
            target_column: Optional explicit fraud/target column.
            sample_size: Number of sample rows returned by preview().
        """
        self.data_path = self._resolve_data_path(data_path)
        self.target_column = target_column
        self.sample_size = max(0, int(sample_size))
        self._df: pd.DataFrame | None = None

        logger.info(
            "claims_eda_initialized data_path={} target_column={} sample_size={}",
            str(self.data_path) if self.data_path else None,
            self.target_column,
            self.sample_size,
        )

    @staticmethod
    def _resolve_data_path(data_path: str | Path | None) -> Path | None:
        if data_path:
            return Path(data_path).expanduser()

        env_path = os.getenv("CLAIMS_DATA_PATH")
        if env_path:
            return Path(env_path).expanduser()

        # Expected structure: project_root/app/implementations/this_file.py
        project_root = Path(__file__).resolve().parents[2]
        candidates = (
            project_root / "artifacts" / "claims.csv",
            project_root / "data" / "claims.csv",
            project_root / "datasets" / "claims.csv",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Keep a useful default for diagnostics; load_data() raises a clear error.
        return candidates[0]

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert pandas/numpy values to JSON-safe values."""
        if value is None:
            return None

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, float) and (
            math.isnan(value) or math.isinf(value)
        ):
            return None

        if isinstance(value, dict):
            return {
                str(k): ClaimsEDAImplementation._json_safe(v)
                for k, v in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                ClaimsEDAImplementation._json_safe(v)
                for v in value
            ]

        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            try:
                return ClaimsEDAImplementation._json_safe(value.tolist())
            except (ValueError, TypeError):
                pass

        try:
            missing = pd.isna(value)

            if isinstance(missing, bool):
                if missing:
                    return None
            elif getattr(missing, "ndim", 0) == 0 and bool(missing):
                return None

        except (TypeError, ValueError):
            pass

        return value

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """Load and cache the configured CSV dataset."""
        if self._df is not None and not force_reload:
            logger.debug("claims_eda_using_cached_dataset rows={} columns={}", *self._df.shape)
            return self._df

        if self.data_path is None:
            raise ValueError(
                "Claims dataset path is not configured. Set CLAIMS_DATA_PATH or pass data_path."
            )
        if not self.data_path.exists():
            logger.error("claims_eda_dataset_not_found path={}", str(self.data_path))
            raise FileNotFoundError(
                f"Claims dataset not found at '{self.data_path}'. "
                "Set CLAIMS_DATA_PATH or pass data_path explicitly."
            )

        started = time.perf_counter()
        logger.info("claims_eda_dataset_loading path={}", str(self.data_path))
        try:
            df = pd.read_csv(self.data_path)
        except Exception:
            logger.exception("claims_eda_dataset_load_failed path={}", str(self.data_path))
            raise

        if df.empty:
            logger.warning("claims_eda_dataset_empty path={}", str(self.data_path))
        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            logger.warning("claims_eda_duplicate_columns columns={}", duplicated)
            raise ValueError(f"Dataset contains duplicate column names: {duplicated}")

        self._df = df
        logger.info(
            "claims_eda_dataset_loaded path={} rows={} columns={} duration_seconds={:.4f}",
            str(self.data_path),
            int(df.shape[0]),
            int(df.shape[1]),
            time.perf_counter() - started,
        )
        return self._df

    def _get_target_column(self, df: pd.DataFrame) -> str | None:
        if self.target_column:
            if self.target_column not in df.columns:
                raise ValueError(
                    f"Configured target_column '{self.target_column}' is not present. "
                    f"Available columns: {df.columns.tolist()}"
                )
            return self.target_column

        for column in self.DEFAULT_TARGET_COLUMNS:
            if column in df.columns:
                return column
        return None

    def dataset_overview(self) -> dict[str, Any]:
        """Return row/column counts, memory usage, duplicate counts, and column names."""
        started = time.perf_counter()
        df = self.load_data()
        result = {
            "dataset_path": str(self.data_path),
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "columns": df.columns.tolist(),
            "duplicate_row_count": int(df.duplicated().sum()),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "target_column": self._get_target_column(df),
        }
        logger.info(
            "claims_eda_overview_completed rows={} columns={} duplicates={} duration_seconds={:.4f}",
            result["row_count"],
            result["column_count"],
            result["duplicate_row_count"],
            time.perf_counter() - started,
        )
        return self._json_safe(result)

    def missing_values(self) -> dict[str, Any]:
        """Return missing counts and percentages per column."""
        df = self.load_data()
        total_rows = len(df)
        counts = df.isna().sum()
        details = []
        for column in df.columns:
            count = int(counts[column])
            details.append(
                {
                    "column": str(column),
                    "missing_count": count,
                    "missing_percentage": round((count / total_rows * 100.0), 4)
                    if total_rows
                    else 0.0,
                }
            )
        details.sort(key=lambda item: item["missing_count"], reverse=True)
        logger.info(
            "claims_eda_missing_values_completed columns={} missing_cells={}",
            len(df.columns),
            int(counts.sum()),
        )
        return {
            "row_count": int(total_rows),
            "total_missing_cells": int(counts.sum()),
            "columns": details,
        }

    def numeric_summary(self) -> dict[str, Any]:
        """Return descriptive statistics for numeric columns."""
        df = self.load_data()
        numeric = df.select_dtypes(include="number")
        summary = numeric.describe().transpose().reset_index(names="column")
        records = summary.to_dict(orient="records") if not summary.empty else []
        logger.info(
            "claims_eda_numeric_summary_completed numeric_columns={}",
            len(numeric.columns),
        )
        return {
            "numeric_column_count": int(len(numeric.columns)),
            "summary": self._json_safe(records),
        }

    def categorical_summary(self, top_n: int = 10) -> dict[str, Any]:
        """Return distinct counts and top values for non-numeric columns."""
        df = self.load_data()
        top_n = max(1, int(top_n))
        categorical = df.select_dtypes(include=["object", "category", "bool"])
        details = []
        for column in categorical.columns:
            counts = categorical[column].astype("string").fillna("<MISSING>").value_counts().head(top_n)
            details.append(
                {
                    "column": str(column),
                    "unique_count": int(df[column].nunique(dropna=True)),
                    "top_values": [
                        {"value": str(value), "count": int(count)}
                        for value, count in counts.items()
                    ],
                }
            )
        logger.info(
            "claims_eda_categorical_summary_completed categorical_columns={} top_n={}",
            len(categorical.columns),
            top_n,
        )
        return {
            "categorical_column_count": int(len(categorical.columns)),
            "top_n": top_n,
            "summary": details,
        }

    def target_distribution(self) -> dict[str, Any]:
        """Return target class counts and percentages if a target column is available."""
        df = self.load_data()
        target = self._get_target_column(df)
        if target is None:
            logger.warning("claims_eda_target_not_detected")
            return {
                "available": False,
                "target_column": None,
                "message": "No target column detected. Pass target_column explicitly if needed.",
                "classes": [],
            }

        counts = df[target].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        total = int(len(df))
        classes = [
            {
                "class": str(label),
                "count": int(count),
                "percentage": round((int(count) / total * 100.0), 4) if total else 0.0,
            }
            for label, count in counts.items()
        ]
        logger.info(
            "claims_eda_target_distribution_completed target_column={} class_count={}",
            target,
            len(classes),
        )
        return {
            "available": True,
            "target_column": target,
            "row_count": total,
            "classes": classes,
        }

    def date_column_summary(self) -> dict[str, Any]:
        """Summarize known date columns if present; does not alter the dataset."""
        df = self.load_data()
        results = []
        for column in self.DEFAULT_DATE_COLUMNS:
            if column not in df.columns:
                continue
            parsed = pd.to_datetime(df[column], errors="coerce")
            valid = parsed.dropna()
            results.append(
                {
                    "column": column,
                    "valid_date_count": int(len(valid)),
                    "invalid_or_missing_count": int(len(df) - len(valid)),
                    "min_date": valid.min().isoformat() if not valid.empty else None,
                    "max_date": valid.max().isoformat() if not valid.empty else None,
                }
            )
        logger.info("claims_eda_date_summary_completed date_columns={}", len(results))
        return {"date_columns": results}

    def preview(self, n: int | None = None) -> dict[str, Any]:
        """Return the first n rows as JSON-safe records."""
        df = self.load_data()
        row_limit = self.sample_size if n is None else max(0, int(n))
        records = df.head(row_limit).to_dict(orient="records")
        logger.info("claims_eda_preview_completed requested_rows={} returned_rows={}", row_limit, len(records))
        return {
            "returned_rows": len(records),
            "columns": df.columns.tolist(),
            "rows": self._json_safe(records),
        }

    def run_full_eda(self, top_n: int = 10, preview_rows: int | None = None) -> dict[str, Any]:
        """Run all EDA reports and return one combined result."""
        started = time.perf_counter()
        logger.info("claims_eda_full_report_started")
        try:
            result = {
                "status": "SUCCESS",
                "overview": self.dataset_overview(),
                "missing_values": self.missing_values(),
                "numeric_summary": self.numeric_summary(),
                "categorical_summary": self.categorical_summary(top_n=top_n),
                "target_distribution": self.target_distribution(),
                "date_summary": self.date_column_summary(),
                "preview": self.preview(n=preview_rows),
            }
            result["execution_time_seconds"] = round(time.perf_counter() - started, 4)
            logger.info(
                "claims_eda_full_report_completed duration_seconds={:.4f}",
                result["execution_time_seconds"],
            )
            return result
        except Exception:
            logger.exception("claims_eda_full_report_failed")
            raise
