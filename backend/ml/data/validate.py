"""
Main validation entry point for the Claims Fraud Analytics pilot.

Runs validate_dataset.py checks against claims.csv, prints a concise summary,
and exits with a non-zero status if validation fails.

Examples:
    python validate.py
    python validate.py --input data/raw/claims.csv
    python validate.py --input claims.csv --report artifacts/reports/claims_validation.json
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.validate_dataset import validate_dataset,ARTIFACT_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/claims.csv"),
        help="Path to claims CSV (default: data/claims.csv)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=12_000,
        help="Expected rows; set 0 to disable this check.",
    )
    parser.add_argument(
        "--expected-fraud-cases",
        type=int,
        default=358,
        help="Expected fraud labels; set 0 to disable this check.",
    )
    parser.add_argument(
        "--expected-fraud-rate",
        type=float,
        default=2.9833333333,
        help="Expected fraud percentage; set a negative value to disable.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(f"{ARTIFACT_DIR}/reports/dataset_validation_report.json"),
        help="Path for JSON report output.",
    )
    args = parser.parse_args()

    result = validate_dataset(
        input_path=args.input,
        expected_rows=None if args.expected_rows == 0 else args.expected_rows,
        expected_fraud_cases=(
            None if args.expected_fraud_cases == 0
            else args.expected_fraud_cases
        ),
        expected_fraud_rate=(
            None if args.expected_fraud_rate < 0
            else args.expected_fraud_rate
        ),
    )

    print("=" * 72)
    print("CLAIMS FRAUD ANALYTICS — DATASET VALIDATION")
    print("=" * 72)
    print(f"Input file : {result.get('input_file', args.input)}")
    print(f"Status     : {result.get('status', 'FAIL')}")
    print(f"Rows       : {result.get('rows', 0):,}")
    print(f"Columns    : {result.get('column_count', 0)}")

    fraud = result.get("fraud_summary", {})
    if fraud:
        print(f"Fraud cases: {fraud.get('fraud_cases', 0):,}")
        print(f"Fraud rate : {fraud.get('fraud_rate_percent', 0):.2f}%")

    checks = result.get("checks", [])
    passed_checks = sum(item.get("status") == "PASS" for item in checks)
    failed_checks = sum(item.get("status") == "FAIL" for item in checks)
    print(f"Checks     : {passed_checks} passed, {failed_checks} failed")

    errors = result.get("errors", [])
    warnings = result.get("warnings", [])

    if errors:
        print("\nERRORS:")
        for item in errors:
            print(f"  - {item}")

    if warnings:
        print("\nWARNINGS:")
        for item in warnings:
            print(f"  - {item}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON report: {args.report.resolve()}")

    if result.get("status") == "PASS":
        print("\nValidation passed. You can proceed to EDA and model training.")
        return 0

    print("\nValidation failed. Resolve the errors before proceeding.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
