from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def top_k_ids(row: np.ndarray, k: int) -> list[int]:
    flat = row.reshape(-1)
    if k <= 0:
        return []
    k = min(k, flat.size)
    unordered = np.argpartition(-flat, np.arange(k))[:k]
    ordered = unordered[np.argsort(-flat[unordered])]
    return [int(i) for i in ordered]


@dataclass(frozen=True)
class LogitThresholds:
    max_abs: float | None = None
    mean_abs: float | None = None
    min_cosine: float | None = None
    min_topk_overlap: float = 0.8
    require_top1: bool = False


def compare_logits(
    candidate: np.ndarray,
    reference: np.ndarray,
    top_k: int = 10,
    thresholds: LogitThresholds | None = None,
) -> dict:
    thresholds = thresholds or LogitThresholds()
    if candidate.ndim != 2 or reference.ndim != 2:
        return {
            "kind": "logits",
            "status": "fail",
            "shape_check": {
                "matches": False,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "rows": [],
            "notes": ["logits must have shape [prompts, vocab]"],
        }
    shape_match = candidate.shape == reference.shape
    rows: list[dict] = []
    notes: list[str] = []
    if not shape_match:
        return {
            "kind": "logits",
            "status": "fail",
            "shape_check": {
                "matches": False,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "rows": rows,
            "notes": ["shape mismatch"],
        }
    if candidate.shape[0] == 0 or candidate.shape[1] == 0:
        return {
            "kind": "logits",
            "status": "fail",
            "shape_check": {
                "matches": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "summary": {
                "num_prompts": int(candidate.shape[0]),
                "vocab_size": int(candidate.shape[1]),
            },
            "rows": [],
            "notes": ["logits must contain at least one prompt and one vocabulary column"],
        }
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return {
            "kind": "logits",
            "status": "fail",
            "shape_check": {
                "matches": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "finite_check": {
                "candidate_finite": bool(np.isfinite(candidate).all()),
                "reference_finite": bool(np.isfinite(reference).all()),
            },
            "summary": {
                "num_prompts": int(candidate.shape[0]),
                "vocab_size": int(candidate.shape[1]),
            },
            "rows": [],
            "notes": ["logits contain NaN or infinite values"],
        }

    for i in range(candidate.shape[0]):
        c = candidate[i]
        r = reference[i]
        diff = np.abs(c - r)
        c_top = top_k_ids(c, top_k)
        r_top = top_k_ids(r, top_k)
        overlap = len(set(c_top) & set(r_top))
        overlap_ratio = overlap / max(len(r_top), 1)
        rows.append(
            {
                "prompt_index": i,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "median_abs_diff": float(np.median(diff)),
                "p95_abs_diff": float(np.percentile(diff, 95)),
                "cosine_similarity": cosine(c, r),
                "candidate_top1": c_top[0] if c_top else None,
                "reference_top1": r_top[0] if r_top else None,
                "top1_match": bool(c_top and r_top and c_top[0] == r_top[0]),
                "top_k": top_k,
                "topk_overlap_count": overlap,
                "topk_overlap_ratio": float(overlap_ratio),
                "candidate_topk": c_top,
                "reference_topk": r_top,
            }
        )

    status = "pass"
    if any(not row["top1_match"] for row in rows):
        notes.append("one or more top-1 tokens differ")
        if thresholds.require_top1:
            status = "fail"
        else:
            status = "warn"
    if min(row["topk_overlap_ratio"] for row in rows) < thresholds.min_topk_overlap:
        notes.append("top-k overlap below threshold")
        status = "fail"
    if thresholds.max_abs is not None and max(row["max_abs_diff"] for row in rows) > thresholds.max_abs:
        notes.append("max absolute diff above threshold")
        status = "fail"
    if thresholds.mean_abs is not None and max(row["mean_abs_diff"] for row in rows) > thresholds.mean_abs:
        notes.append("mean absolute diff above threshold")
        status = "fail"
    if thresholds.min_cosine is not None and min(row["cosine_similarity"] for row in rows) < thresholds.min_cosine:
        notes.append("cosine below threshold")
        status = "fail"

    return {
        "kind": "logits",
        "status": status,
        "shape_check": {
            "matches": True,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        },
        "summary": {
            "num_prompts": int(candidate.shape[0]),
            "vocab_size": int(candidate.shape[1]),
            "all_top1_match": all(row["top1_match"] for row in rows),
            "min_topk_overlap_ratio": min(row["topk_overlap_ratio"] for row in rows),
            "min_cosine_similarity": min(row["cosine_similarity"] for row in rows),
            "max_abs_diff_overall": max(row["max_abs_diff"] for row in rows),
            "mean_abs_diff_overall": float(np.mean([row["mean_abs_diff"] for row in rows])),
        },
        "rows": rows,
        "thresholds": thresholds.__dict__,
        "notes": notes,
    }


def compare_layers(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if candidate.ndim != 2 or reference.ndim != 2:
        return {
            "kind": "layers",
            "status": "fail",
            "shape_check": {
                "matches": False,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "layers": [],
            "notes": ["layer dumps must have shape [layers, hidden_size]"],
        }
    if candidate.shape != reference.shape:
        return {
            "kind": "layers",
            "status": "fail",
            "shape_check": {
                "matches": False,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "layers": [],
            "notes": ["shape mismatch"],
        }
    if candidate.shape[0] == 0 or candidate.shape[1] == 0:
        return {
            "kind": "layers",
            "status": "fail",
            "shape_check": {
                "matches": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "summary": {
                "layers": int(candidate.shape[0]),
                "hidden_size": int(candidate.shape[1]),
            },
            "layers": [],
            "notes": ["layer dumps must contain at least one layer and one hidden dimension"],
        }
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return {
            "kind": "layers",
            "status": "fail",
            "shape_check": {
                "matches": True,
                "candidate_shape": list(candidate.shape),
                "reference_shape": list(reference.shape),
            },
            "finite_check": {
                "candidate_finite": bool(np.isfinite(candidate).all()),
                "reference_finite": bool(np.isfinite(reference).all()),
            },
            "summary": {
                "layers": int(candidate.shape[0]),
                "hidden_size": int(candidate.shape[1]),
            },
            "layers": [],
            "notes": ["layer dumps contain NaN or infinite values"],
        }

    layers = []
    for i in range(candidate.shape[0]):
        c = candidate[i]
        r = reference[i]
        diff = np.abs(c - r)
        layers.append(
            {
                "layer": i,
                "cosine_similarity": cosine(c, r),
                "candidate_l2": float(np.linalg.norm(c)),
                "reference_l2": float(np.linalg.norm(r)),
                "mean_abs_diff": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )
    return {
        "kind": "layers",
        "status": "pass",
        "shape_check": {
            "matches": True,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        },
        "summary": {
            "layers": int(candidate.shape[0]),
            "hidden_size": int(candidate.shape[1]),
            "min_cosine_similarity": min(row["cosine_similarity"] for row in layers),
            "max_abs_diff_overall": max(row["max_abs_diff"] for row in layers),
        },
        "layers": layers,
        "notes": [],
    }
