"""
Data validation and quality checks.
Validates schema, data quality, and detects data drift.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DataValidator:
    """Validate data quality and detect drift."""

    def __init__(self, data_path: str, reference_path: str = None):
        """
        Initialize data validator.

        Args:
            data_path: Path to current data
            reference_path: Path to reference/baseline data (optional)
        """
        self.data_path = Path(data_path)
        self.reference_path = Path(reference_path) if reference_path else None

        # Load data
        self.current_data = self._load_data(self.data_path)
        self.reference_data = None

        if self.reference_path and self.reference_path.exists():
            self.reference_data = self._load_data(self.reference_path)

    def _load_data(self, path: Path) -> pd.DataFrame:
        """Load data from file."""
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def validate_schema(self) -> Tuple[bool, Dict]:
        """
        Validate data schema.

        Returns:
            (is_valid, report)
        """
        print("\n" + "=" * 70)
        print("SCHEMA VALIDATION")
        print("=" * 70)

        report = {
            "total_rows": len(self.current_data),
            "total_columns": len(self.current_data.columns),
            "columns": list(self.current_data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.current_data.dtypes.items()},
            "missing_values": self.current_data.isnull().sum().to_dict(),
            "issues": [],
        }

        # Check for empty dataset
        if len(self.current_data) == 0:
            report["issues"].append("Dataset is empty")
            print("❌ Dataset is empty")
            return False, report

        print(f"✓ Total rows: {report['total_rows']:,}")
        print(f"✓ Total columns: {report['total_columns']}")

        # Check for missing values
        missing = self.current_data.isnull().sum()
        if missing.any():
            print("\n⚠️  Missing values detected:")
            for col, count in missing[missing > 0].items():
                pct = (count / len(self.current_data)) * 100
                print(f"   {col}: {count} ({pct:.1f}%)")
                if pct > 50:
                    report["issues"].append(f"Column '{col}' has >50% missing values")
        else:
            print("✓ No missing values")

        # Check for duplicate rows
        duplicates = self.current_data.duplicated().sum()
        if duplicates > 0:
            pct = (duplicates / len(self.current_data)) * 100
            print(f"\n⚠️  Duplicate rows: {duplicates} ({pct:.1f}%)")
            report["duplicate_rows"] = int(duplicates)
        else:
            print("✓ No duplicate rows")

        # Schema comparison with reference
        if self.reference_data is not None:
            print("\n→ Comparing with reference data...")

            ref_cols = set(self.reference_data.columns)
            curr_cols = set(self.current_data.columns)

            missing_cols = ref_cols - curr_cols
            new_cols = curr_cols - ref_cols

            if missing_cols:
                print(f"❌ Missing columns: {missing_cols}")
                report["issues"].append(f"Missing columns: {list(missing_cols)}")

            if new_cols:
                print(f"⚠️  New columns: {new_cols}")
                report["new_columns"] = list(new_cols)

            # Check dtype consistency
            for col in ref_cols & curr_cols:
                if self.reference_data[col].dtype != self.current_data[col].dtype:
                    msg = f"Column '{col}' dtype mismatch: {self.reference_data[col].dtype} → {self.current_data[col].dtype}"
                    print(f"❌ {msg}")
                    report["issues"].append(msg)

        is_valid = len(report["issues"]) == 0

        if is_valid:
            print("\n✅ Schema validation passed")
        else:
            print(f"\n❌ Schema validation failed: {len(report['issues'])} issues")

        return is_valid, report

    def check_data_quality(self) -> Tuple[bool, Dict]:
        """
        Check data quality metrics.

        Returns:
            (is_valid, report)
        """
        print("\n" + "=" * 70)
        print("DATA QUALITY CHECKS")
        print("=" * 70)

        report = {"numeric_stats": {}, "categorical_stats": {}, "issues": []}

        # Numeric columns
        numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            print(f"\nNumeric columns: {len(numeric_cols)}")

            for col in numeric_cols:
                stats_dict = {
                    "min": float(self.current_data[col].min()),
                    "max": float(self.current_data[col].max()),
                    "mean": float(self.current_data[col].mean()),
                    "std": float(self.current_data[col].std()),
                    "median": float(self.current_data[col].median()),
                }

                report["numeric_stats"][col] = stats_dict

                # Check for outliers (simple IQR method)
                Q1 = self.current_data[col].quantile(0.25)
                Q3 = self.current_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (self.current_data[col] < (Q1 - 1.5 * IQR))
                    | (self.current_data[col] > (Q3 + 1.5 * IQR))
                ).sum()

                if outliers > 0:
                    pct = (outliers / len(self.current_data)) * 100
                    print(f"  {col}: {outliers} outliers ({pct:.1f}%)")
                    stats_dict["outliers"] = int(outliers)

        # Categorical columns
        categorical_cols = self.current_data.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) > 0:
            print(f"\nCategorical columns: {len(categorical_cols)}")

            for col in categorical_cols:
                unique_vals = self.current_data[col].nunique()
                most_common = self.current_data[col].value_counts().head(5).to_dict()

                report["categorical_stats"][col] = {
                    "unique_values": int(unique_vals),
                    "most_common": most_common,
                }

                print(f"  {col}: {unique_vals} unique values")

                # Check for high cardinality
                cardinality_ratio = unique_vals / len(self.current_data)
                if cardinality_ratio > 0.9:
                    msg = f"Column '{col}' has very high cardinality ({cardinality_ratio:.1%})"
                    print(f"    ⚠️  {msg}")
                    report["issues"].append(msg)

        print("\n✅ Data quality checks completed")

        return True, report

    def detect_drift(self) -> Tuple[bool, Dict]:
        """
        Detect data drift using Kolmogorov-Smirnov test.

        Returns:
            (drift_detected, report)
        """
        if self.reference_data is None:
            print("\n⚠️  No reference data - skipping drift detection")
            return False, {"skipped": "No reference data"}

        print("\n" + "=" * 70)
        print("DATA DRIFT DETECTION")
        print("=" * 70)

        report = {"drift_detected": False, "drifted_columns": [], "column_details": {}}

        # Only check common numeric columns
        common_cols = set(self.current_data.columns) & set(self.reference_data.columns)
        numeric_cols = [
            col for col in common_cols if pd.api.types.is_numeric_dtype(self.current_data[col])
        ]

        print(f"\nChecking {len(numeric_cols)} numeric columns for drift...")

        drift_threshold = 0.05  # p-value threshold

        for col in numeric_cols:
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(), self.current_data[col].dropna()
            )

            has_drift = p_value < drift_threshold

            report["column_details"][col] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": has_drift,
            }

            if has_drift:
                report["drift_detected"] = True
                report["drifted_columns"].append(col)
                print(f"  ❌ {col}: DRIFT DETECTED (p={p_value:.4f})")
            else:
                print(f"  ✓ {col}: No drift (p={p_value:.4f})")

        if report["drift_detected"]:
            print(f"\n❌ Drift detected in {len(report['drifted_columns'])} columns")
        else:
            print("\n✅ No significant drift detected")

        return report["drift_detected"], report

    def run_all_validations(self) -> Dict:
        """
        Run all validation checks.

        Returns:
            Complete validation report
        """
        full_report = {
            "data_path": str(self.data_path),
            "reference_path": str(self.reference_path) if self.reference_path else None,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Schema validation
        schema_valid, schema_report = self.validate_schema()
        full_report["schema_validation"] = {"passed": schema_valid, **schema_report}

        # Data quality
        quality_valid, quality_report = self.check_data_quality()
        full_report["data_quality"] = {"passed": quality_valid, **quality_report}

        # Drift detection
        drift_detected, drift_report = self.detect_drift()
        full_report["drift_detection"] = drift_report

        # Overall status
        full_report["overall_status"] = "PASSED" if schema_valid and quality_valid else "FAILED"

        print("\n" + "=" * 70)
        print(f"OVERALL STATUS: {full_report['overall_status']}")
        print("=" * 70 + "\n")

        return full_report


def main():
    parser = argparse.ArgumentParser(description="Validate data quality and detect drift")
    parser.add_argument("--data-path", type=str, required=True, help="Path to current data")
    parser.add_argument(
        "--reference-path", type=str, help="Path to reference/baseline data (for drift detection)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_validation_report.json",
        help="Output file for validation report",
    )

    args = parser.parse_args()

    # Run validation
    validator = DataValidator(data_path=args.data_path, reference_path=args.reference_path)

    report = validator.run_all_validations()

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"✓ Validation report saved to {args.output}")

    # Exit with appropriate code
    exit(0 if report["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()
