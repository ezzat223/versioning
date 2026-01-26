#!/usr/bin/env python3
"""
Generate CML markdown report from comparison JSON.
Optimized for GitLab merge request comments with rich formatting.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def generate_markdown_report(comparison: Dict[str, Any]) -> str:
    """
    Generate markdown report for CML comments.

    Args:
        comparison: Comparison dictionary from compare_models.py

    Returns:
        Formatted markdown string
    """
    promote = comparison.get("promote_challenger", False)
    comp = comparison.get("comparison", {})

    # Header with decision badge
    if promote:
        badge = "🏆 **PROMOTE TO CHAMPION**"
        emoji = "✅"
        color = "success"
    else:
        badge = "⚠️ **KEEP CURRENT CHAMPION**"
        emoji = "⛔"
        color = "yellow"

    lines = [
        f"## {emoji} Model Comparison Report",
        "",
        badge,
        "",
        "---",
        "",
    ]

    # Decision reason with quote formatting
    reason = comp.get("reason", "No reason provided")
    lines.extend(
        [
            "### 📋 Decision",
            "",
            f"> {reason}",
            "",
        ]
    )

    # Metrics comparison table
    metric = comp.get("metric", "primary_metric")
    champ_score = comp.get("champion_score")
    chall_score = comp.get("challenger_score")
    improvement = comp.get("pct_improvement")
    higher_is_better = comp.get("higher_is_better", True)

    lines.extend(
        [
            "### 📊 Performance Comparison",
            "",
            f"**Primary Metric:** `{metric}`",
            "",
            "| Model | Score | Change |",
            "|:------|------:|:------:|",
        ]
    )

    # Champion row
    if champ_score is not None:
        lines.append(f"| Champion | `{champ_score:.4f}` | baseline |")
    else:
        lines.append("| Champion | N/A | _(first model)_ |")

    # Challenger row
    if chall_score is not None:
        if improvement is not None:
            # Determine trend emoji based on improvement direction
            if higher_is_better:
                trend = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            else:
                # For metrics where lower is better
                trend = "📉" if improvement > 0 else "📈" if improvement < 0 else "➡️"

            change_text = f"`{improvement*100:+.2f}%` {trend}"
        else:
            # First model case
            change_text = "🆕 _(new)_"

        lines.append(f"| **Challenger** | `{chall_score:.4f}` | {change_text} |")

    lines.append("")

    # All metrics in collapsible section
    all_metrics = comparison.get("all_challenger_metrics", {})
    if all_metrics and len(all_metrics) > 1:
        lines.extend(
            [
                "<details>",
                "<summary>📋 <b>All Metrics</b> (click to expand)</summary>",
                "",
                "| Metric | Value |",
                "|:-------|------:|",
            ]
        )

        for name in sorted(all_metrics.keys()):
            value = all_metrics[name]
            if name == metric:
                # Highlight primary metric
                lines.append(f"| **{name}** | **`{value:.4f}`** |")
            else:
                lines.append(f"| {name} | `{value:.4f}` |")

        lines.extend(["", "</details>", ""])

    # Detailed change summary
    if improvement is not None and champ_score is not None and chall_score is not None:
        abs_change = chall_score - champ_score

        lines.extend(
            [
                "### 📈 Change Summary",
                "",
                f"- **Absolute Change:** `{abs_change:+.4f}`",
                f"- **Relative Change:** `{improvement*100:+.2f}%`",
                "- **Required Threshold:** `1.00%`",
                f"- **Metric Direction:** {'Higher is better' if higher_is_better else 'Lower is better'}",
            ]
        )

        if promote:
            lines.append("- **Result:** ✅ Improvement exceeds threshold")
        else:
            if abs(improvement) < 0.01:
                lines.append("- **Result:** ⚠️ Improvement below threshold")
            else:
                lines.append("- **Result:** ❌ Performance degraded")

        lines.append("")

    # Run information
    lines.extend(
        [
            "---",
            "",
            "### 🔍 Run Information",
            "",
            f"- **Experiment:** `{comparison.get('experiment_name', 'N/A')}`",
            f"- **Champion Run:** `{comparison.get('champion_run_id', 'None')}`",
            f"- **Challenger Run:** `{comparison.get('challenger_run_id', 'Unknown')}`",
            "",
        ]
    )

    # Next steps based on decision
    if promote:
        lines.extend(
            [
                "### ✅ Next Steps",
                "",
                "The pipeline will automatically:",
                "",
                "1. ✅ Merge this MR to main",
                "2. 🏆 Promote challenger to champion",
                "3. 🐳 Build Docker image with new model",
                "4. 🚀 Deploy to production (if configured)",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### ⚠️ Next Steps",
                "",
                "- ❌ MR will **not** be merged automatically",
                "- 🛡️ Current champion remains in production",
                "- 📊 Review metrics and consider improvements",
                "- 🔧 Adjust hyperparameters or features",
                "",
            ]
        )

    # Footer
    lines.extend(
        [
            "---",
            "",
            "_Generated by MLOps CI/CD Pipeline_",
            "",
            f"![Status](https://img.shields.io/badge/Status-{color.upper()}-{color})",
        ]
    )

    return "\n".join(lines)


def validate_comparison_data(comparison: Dict[str, Any]) -> bool:
    """
    Validate comparison data structure.

    Args:
        comparison: Comparison dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["promote_challenger", "comparison", "challenger_run_id"]

    for key in required_keys:
        if key not in comparison:
            print(f"❌ Missing required key: {key}", file=sys.stderr)
            return False
    comp = comparison.get("comparison", {})
    if not isinstance(comp, dict):
        print("❌ 'comparison' must be a dictionary", file=sys.stderr)
        return False

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate CML markdown report from comparison JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            Basic usage
                >python generate_cml_report.py --comparison comparison.json
            With verbose output
                >python generate_cml_report.py --comparison comparison.json --verbose
            Custom output file
                >python generate_cml_report.py --comparison comparison.json --output custom_report.md
            """,
    )
    parser.add_argument(
        "--comparison", required=True, help="Path to comparison.json file from compare_models.py"
    )
    parser.add_argument(
        "--output", default="report.md", help="Output markdown file (default: report.md)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print report to stdout for debugging"
    )

    args = parser.parse_args()

    # Load comparison data
    comparison_path = Path(args.comparison)
    if not comparison_path.exists():
        print(f"❌ Error: Comparison file not found: {comparison_path}", file=sys.stderr)
        return 1

    try:
        with open(comparison_path, "r", encoding="utf-8") as f:
            comparison = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {comparison_path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error reading {comparison_path}: {e}", file=sys.stderr)
        return 1

    # Validate data
    if not validate_comparison_data(comparison):
        print("❌ Error: Invalid comparison data structure", file=sys.stderr)
        return 1

    # Generate report
    try:
        report = generate_markdown_report(comparison)
    except Exception as e:
        print(f"❌ Error generating report: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    # Save report
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ CML report saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error writing report to {output_path}: {e}", file=sys.stderr)
        return 1

    # Print preview if verbose
    if args.verbose:
        print("\n" + "=" * 70)
        print("REPORT PREVIEW:")
        print("=" * 70)
        print(report)
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
