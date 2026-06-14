from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import layer_dims_from_metadata, load_layers, load_logits, read_json, write_json
from .metrics import LogitThresholds, compare_layers, compare_logits
from .reports import markdown_summary, write_report_bundle
from .token_audit import audit_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gguf-parity")
    sub = parser.add_subparsers(dest="command", required=True)

    logits = sub.add_parser("compare-logits", help="compare candidate logits against reference logits")
    logits.add_argument("--candidate", required=True, help="candidate .npy/.npz logits")
    logits.add_argument("--reference", required=True, help="reference .npy/.npz logits")
    logits.add_argument("--candidate-key", default="logits", help="candidate .npz key")
    logits.add_argument("--reference-key", default="logits", help="reference .npz key")
    logits.add_argument("--candidate-metadata", help="candidate metadata JSON")
    logits.add_argument("--reference-metadata", help="reference metadata JSON")
    logits.add_argument("--top-k", type=int, default=10)
    logits.add_argument("--max-abs", type=float)
    logits.add_argument("--mean-abs", type=float)
    logits.add_argument("--min-cosine", type=float)
    logits.add_argument("--min-topk-overlap", type=float, default=0.8)
    logits.add_argument("--require-top1", action="store_true")
    logits.add_argument("--out", help="output directory for report bundle")
    logits.set_defaults(func=cmd_compare_logits)

    layers = sub.add_parser("compare-layers", help="compare per-layer hidden-state dumps")
    layers.add_argument("--candidate", required=True, help="candidate flat f32 layer dump")
    layers.add_argument("--reference", required=True, help="reference flat f32 layer dump")
    layers.add_argument("--metadata", help="layer metadata JSON with layers/hidden_size")
    layers.add_argument("--layers", type=int)
    layers.add_argument("--hidden-size", type=int)
    layers.add_argument("--out", help="output directory for report bundle")
    layers.set_defaults(func=cmd_compare_layers)

    audit = sub.add_parser("token-audit", help="compare prompt/token ID metadata")
    audit.add_argument("--candidate-metadata", required=True)
    audit.add_argument("--reference-metadata", required=True)
    audit.add_argument("--out", help="optional JSON output path")
    audit.set_defaults(func=cmd_token_audit)
    return parser


def cmd_compare_logits(args: argparse.Namespace) -> int:
    candidate = load_logits(args.candidate, args.candidate_key)
    reference = load_logits(args.reference, args.reference_key)
    thresholds = LogitThresholds(
        max_abs=args.max_abs,
        mean_abs=args.mean_abs,
        min_cosine=args.min_cosine,
        min_topk_overlap=args.min_topk_overlap,
        require_top1=args.require_top1,
    )
    report = compare_logits(candidate, reference, args.top_k, thresholds)
    report["inputs"] = {
        "candidate": args.candidate,
        "reference": args.reference,
        "candidate_metadata_path": args.candidate_metadata,
        "reference_metadata_path": args.reference_metadata,
    }
    candidate_metadata = read_json(args.candidate_metadata)
    reference_metadata = read_json(args.reference_metadata)
    if candidate_metadata and reference_metadata:
        report["token_audit"] = audit_tokens(candidate_metadata, reference_metadata)
    elif candidate_metadata or reference_metadata:
        report.setdefault("notes", []).append("token audit skipped because only one metadata file was provided")
    if args.out:
        write_report_bundle(report, args.out)
    else:
        print(markdown_summary(report))
    return 1 if report["status"] == "fail" else 0


def cmd_compare_layers(args: argparse.Namespace) -> int:
    metadata = read_json(args.metadata)
    meta_layers, meta_hidden = layer_dims_from_metadata(metadata)
    layers = args.layers or meta_layers
    hidden_size = args.hidden_size or meta_hidden
    if layers is None or hidden_size is None:
        raise SystemExit("--layers and --hidden-size are required unless provided by --metadata")
    candidate = load_layers(args.candidate, layers, hidden_size)
    reference = load_layers(args.reference, layers, hidden_size)
    report = compare_layers(candidate, reference)
    report["inputs"] = {"candidate": args.candidate, "reference": args.reference, "metadata": args.metadata}
    if metadata:
        report["metadata"] = metadata
    if args.out:
        write_report_bundle(report, args.out)
    else:
        print(markdown_summary(report))
    return 1 if report["status"] == "fail" else 0


def cmd_token_audit(args: argparse.Namespace) -> int:
    report = audit_tokens(read_json(args.candidate_metadata), read_json(args.reference_metadata))
    if args.out:
        write_json(Path(args.out), report)
    else:
        print(markdown_summary(report))
    return 1 if report["status"] == "fail" else 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except BrokenPipeError:
        code = 1
    sys.exit(code)
