from __future__ import annotations

import csv
from pathlib import Path

from .io import write_json


def write_report_bundle(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", report)
    (out / "summary.md").write_text(markdown_summary(report), encoding="utf-8")
    write_csv(report, out)


def markdown_summary(report: dict) -> str:
    kind = report.get("kind", "report")
    status = report.get("status", "unknown")
    lines = [f"# {kind.replace('_', ' ').title()} Report", "", f"Status: `{status}`", ""]
    if report.get("summary"):
        lines.extend(["## Summary", ""])
        for key, value in report["summary"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    if report.get("notes"):
        lines.extend(["## Notes", ""])
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    if kind == "logits":
        lines.extend(
            [
                "## Prompts",
                "",
                "| prompt | cosine | top1 match | top-k overlap | mean abs diff | max abs diff |",
                "|--------|--------|------------|---------------|---------------|--------------|",
            ]
        )
        for row in report.get("rows", []):
            lines.append(
                f"| {row['prompt_index']} | {row['cosine_similarity']:.6f} | "
                f"{row['top1_match']} | {row['topk_overlap_ratio']:.3f} | "
                f"{row['mean_abs_diff']:.6g} | {row['max_abs_diff']:.6g} |"
            )
    elif kind == "layers":
        lines.extend(
            [
                "## Layers",
                "",
                "| layer | cosine | candidate L2 | reference L2 | mean abs diff | max abs diff |",
                "|-------|--------|--------------|--------------|---------------|--------------|",
            ]
        )
        for row in report.get("layers", []):
            lines.append(
                f"| {row['layer']} | {row['cosine_similarity']:.6f} | "
                f"{row['candidate_l2']:.3f} | {row['reference_l2']:.3f} | "
                f"{row['mean_abs_diff']:.6g} | {row['max_abs_diff']:.6g} |"
            )
    elif kind == "token_audit":
        lines.extend(
            [
                "## Token Rows",
                "",
                "| index | prompt match | token IDs match |",
                "|-------|--------------|-----------------|",
            ]
        )
        for row in report.get("rows", []):
            lines.append(f"| {row['index']} | {row['prompt_match']} | {row['token_ids_match']} |")
    lines.append("")
    return "\n".join(lines)


def write_csv(report: dict, out_dir: Path) -> None:
    kind = report.get("kind")
    if kind == "logits":
        rows = report.get("rows", [])
        if not rows:
            return
        path = out_dir / "logit_rows.csv"
        fields = [
            "prompt_index",
            "cosine_similarity",
            "top1_match",
            "topk_overlap_ratio",
            "mean_abs_diff",
            "max_abs_diff",
        ]
    elif kind == "layers":
        rows = report.get("layers", [])
        if not rows:
            return
        path = out_dir / "layer_rows.csv"
        fields = [
            "layer",
            "cosine_similarity",
            "candidate_l2",
            "reference_l2",
            "mean_abs_diff",
            "max_abs_diff",
        ]
    else:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

