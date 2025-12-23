"""
Generate CML (Continuous Machine Learning) Report.
Reads the comparison report and creates a markdown file for CML to post as a comment.
"""

import argparse
import json
import os
from pathlib import Path


def generate_report(comparison_path: str, output_path: str):
    """Generate Markdown report from comparison JSON."""
    
    if not os.path.exists(comparison_path):
        print(f"❌ Comparison report not found at {comparison_path}")
        return

    with open(comparison_path, "r") as f:
        data = json.load(f)

    experiment_name = data.get("experiment_name", "Unknown")
    promote = data.get("promote_challenger", False)
    comparison = data.get("comparison", {})
    
    # Icons
    status_icon = "✅" if promote else "⚠️"
    decision_text = "PROMOTE CHALLENGER" if promote else "KEEP CHAMPION"

    # Markdown Content
    md = f"""# 📊 Model Comparison Report
    
**Experiment:** `{experiment_name}`
**Decision:** {status_icon} **{decision_text}**

## 🏆 Decision Summary
{comparison.get("reason", "No details available")}

## 📈 Metrics Comparison

| Metric | Champion | Challenger | Improvement |
|--------|----------|------------|-------------|
"""

    # Add primary metric row
    primary_metric = comparison.get("primary_metric", "accuracy")
    champ_score = comparison.get("champion_score", 0)
    chall_score = comparison.get("challenger_score", 0)
    imp_pct = comparison.get("improvement_pct", 0)
    
    md += f"| **{primary_metric}** | {champ_score:.4f} | {chall_score:.4f} | {imp_pct:+.2f}% |\n"

    # Add other metrics if available
    champ_metrics = comparison.get("all_champion_metrics", {})
    chall_metrics = comparison.get("all_challenger_metrics", {})
    
    all_keys = set(champ_metrics.keys()) | set(chall_metrics.keys())
    for k in sorted(all_keys):
        if k == primary_metric:
            continue
        c_val = champ_metrics.get(k, 0)
        ch_val = chall_metrics.get(k, 0)
        # Calculate pct diff if possible
        if c_val != 0:
            diff = ((ch_val - c_val) / abs(c_val)) * 100
            diff_str = f"{diff:+.2f}%"
        else:
            diff_str = "N/A"
            
        md += f"| {k} | {c_val:.4f} | {ch_val:.4f} | {diff_str} |\n"

    md += """
## 🔗 Run Details
- **Champion Run ID:** `{}` 
- **Challenger Run ID:** `{}`
""".format(
        data.get("champion_run_id", "None"),
        data.get("challenger_run_id", "None")
    )

    # Write to file
    with open(output_path, "w") as f:
        f.write(md)
    
    print(f"✓ CML report generated at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CML Report")
    parser.add_argument("--comparison", required=True, help="Path to comparison JSON")
    parser.add_argument("--output", default="cml_report.md", help="Output markdown file")
    args = parser.parse_args()
    
    generate_report(args.comparison, args.output)
