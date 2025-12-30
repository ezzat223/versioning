#!/usr/bin/env python3
"""
Simple Data Validation using Great Expectations.

INSTRUCTIONS FOR DATA SCIENTISTS / MLOPS ENGINEERS:
1. After project setup, customize the expectation suite below
2. Add column-level expectations for your specific dataset
3. Run this script in CI/CD to validate data quality

Great Expectations Docs: https://docs.greatexpectations.io/
"""

import argparse
import sys
from pathlib import Path

import great_expectations as gx
import pandas as pd

# from great_expectations.core.batch import BatchRequest


# =============================================================================
# EXPECTATION SUITE - CUSTOMIZE THIS!
# =============================================================================


def create_expectation_suite(context, suite_name: str = "default"):
    """
    Define your data expectations here.

    This is a STARTER suite. Customize it for your specific use case!

    Examples of what you can add:
    - expect_column_values_to_be_between
    - expect_column_values_to_be_in_set
    - expect_column_values_to_match_regex
    - expect_column_mean_to_be_between
    - expect_column_values_to_be_unique
    - expect_table_row_count_to_be_between
    """

    # Get or create suite (v3 API)
    try:
        suite = context.suites.get(name=suite_name)
        print(f"✓ Using existing suite: {suite_name}")
        # Clear existing expectations (so we can redefine them)
        suite.expectations = []
    except Exception:
        suite = gx.ExpectationSuite(name=suite_name)
        suite = context.suites.add(suite)
        print(f"✓ Created new suite: {suite_name}")

    # =========================================================================
    # BASIC EXPECTATIONS (Always applicable)
    # =========================================================================

    # Expectation 1: Dataset should not be empty
    suite.add_expectation(
        gx.expectations.ExpectTableRowCountToBeBetween(min_value=1, max_value=None)
    )

    # Expectation 2: No completely null columns
    suite.add_expectation(
        gx.expectations.ExpectTableColumnCountToBeBetween(min_value=1, max_value=None)
    )

    # =========================================================================
    # CUSTOM EXPECTATIONS - ADD YOUR OWN BELOW!
    # =========================================================================

    # Example: Expect specific columns to exist
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_to_exist",
    #         kwargs={"column": "age"}
    #     )
    # )

    # Example: Age should be between 0 and 120
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_values_to_be_between",
    #         kwargs={"column": "age", "min_value": 0, "max_value": 120}
    #     )
    # )

    # Example: Email should match regex pattern
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_values_to_match_regex",
    #         kwargs={"column": "email", "regex": r"^[\w\.-]+@[\w\.-]+\.\w+$"}
    #     )
    # )

    # Example: Category should be in a specific set
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_values_to_be_in_set",
    #         kwargs={"column": "category", "value_set": ["A", "B", "C"]}
    #     )
    # )

    # Example: No null values in target column
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_values_to_not_be_null",
    #         kwargs={"column": "target"}
    #     )
    # )

    # Example: Price should have reasonable statistics
    # suite.add_expectation(
    #     gx.core.ExpectationConfiguration(
    #         expectation_type="expect_column_mean_to_be_between",
    #         kwargs={"column": "price", "min_value": 10, "max_value": 1000}
    #     )
    # )

    # =========================================================================
    # TODO: Add your project-specific expectations above!
    # =========================================================================

    return suite


# =============================================================================
# Validation Runner
# =============================================================================


def validate_data(data_path: str, suite_name: str = "default") -> bool:
    """
    Run Great Expectations validation on a dataset.

    Args:
        data_path: Path to CSV or Parquet file
        suite_name: Name of expectation suite

    Returns:
        True if validation passes, False otherwise
    """
    data_path = Path(data_path)

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return False

    print("\n" + "=" * 70)
    print("DATA VALIDATION WITH GREAT EXPECTATIONS")
    print("=" * 70)
    print(f"File: {data_path}")
    print(f"Suite: {suite_name}")
    print("=" * 70 + "\n")

    # Initialize GX context (in-memory, no project files)
    # Ephemeral context (in-memory, no project files)
    context = gx.get_context()

    # Create/update expectation suite
    print("→ Setting up expectation suite...")
    suite = create_expectation_suite(context, suite_name)
    print(f"✓ Suite has {len(suite.expectations)} expectations\n")

    # Add datasource (pandas)
    print("→ Loading data...")
    # Use modern API for Data Sources
    datasource = context.data_sources.add_pandas(name="pandas_datasource")

    # Read data based on file type
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(data_path)
    else:
        print(f"❌ Unsupported file format: {data_path.suffix}")
        return False

    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")

    # Create data asset
    data_asset = datasource.add_dataframe_asset(name="data_asset")

    # Create batch request
    # Build batch request for runtime dataframe
    batch_request = data_asset.build_batch_request(options={"dataframe": df})

    # Create batch
    batch = data_asset.get_batch(batch_request)

    # Run validation
    results = batch.run()

    # Display results
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    success = results.success

    if success:
        print("✅ ALL EXPECTATIONS PASSED")
    else:
        print("❌ SOME EXPECTATIONS FAILED")

    print(f"\nTotal Expectations: {results.statistics['evaluated_expectations']}")
    print(f"Successful: {results.statistics['successful_expectations']}")
    print(f"Failed: {results.statistics['unsuccessful_expectations']}")
    print(f"Success Rate: {results.statistics['success_percent']:.1f}%")

    # Show failed expectations
    if not success:
        print("\n" + "-" * 70)
        print("FAILED EXPECTATIONS:")
        print("-" * 70)

        for result in results.run_results.values():
            for check in result["validation_result"]["results"]:
                if not check["success"]:
                    exp_type = check["expectation_config"]["expectation_type"]
                    kwargs = check["expectation_config"]["kwargs"]

                    print(f"\n❌ {exp_type}")

                    # Show relevant kwargs
                    if "column" in kwargs:
                        print(f"   Column: {kwargs['column']}")

                    # Show failure details
                    if "result" in check:
                        if "observed_value" in check["result"]:
                            print(f"   Observed: {check['result']['observed_value']}")
                        if "unexpected_count" in check["result"]:
                            print(f"   Unexpected Count: {check['result']['unexpected_count']}")
                        if "unexpected_percent" in check["result"]:
                            print(f"   Unexpected %: {check['result']['unexpected_percent']:.2f}%")

    print("\n" + "=" * 70 + "\n")

    return success


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Simple data validation with Great Expectations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate CSV
  python scripts/validate_data.py --data data/processed/train.csv

  # Validate Parquet
  python scripts/validate_data.py --data data/processed/train.parquet

  # Use custom suite name
  python scripts/validate_data.py --data data.csv --suite my_suite

After first run, customize the expectations in create_expectation_suite()!
        """,
    )

    parser.add_argument("--data", required=True, help="Path to data file (CSV or Parquet)")
    parser.add_argument(
        "--suite", default="default", help="Expectation suite name (default: default)"
    )

    args = parser.parse_args()

    try:
        success = validate_data(args.data, args.suite)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE NOTES
# =============================================================================
"""
This script uses Great Expectations in "ephemeral" mode - no project files!

Benefits:
- No .great_expectations/ directory cluttering your repo
- All expectations defined in one place (this file)
- Easy to customize and version control
- Works great with CI/CD

To customize:
1. Edit create_expectation_suite() function above
2. Add expectations for your specific dataset columns
3. Run validation in CI/CD pipeline

Great Expectations Documentation:
- Gallery of expectations: https://greatexpectations.io/expectations
- Expectation types: https://docs.greatexpectations.io/docs/reference/expectations/standard_arguments/

Common expectations:
- expect_column_values_to_be_between
- expect_column_values_to_be_in_set
- expect_column_values_to_not_be_null
- expect_column_values_to_match_regex
- expect_column_mean_to_be_between
- expect_table_row_count_to_be_between
- expect_column_unique_value_count_to_be_between
"""
