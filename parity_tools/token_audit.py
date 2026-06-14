from __future__ import annotations

from typing import Any


def _prompt_rows(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not metadata:
        return []
    prompts = metadata.get("prompts")
    if isinstance(prompts, list):
        return prompts
    if "prompt" in metadata or "token_ids" in metadata:
        return [{"index": 0, "prompt": metadata.get("prompt"), "token_ids": metadata.get("token_ids", [])}]
    return []


def audit_tokens(candidate_metadata: dict[str, Any] | None, reference_metadata: dict[str, Any] | None) -> dict:
    candidate_rows = _prompt_rows(candidate_metadata)
    reference_rows = _prompt_rows(reference_metadata)
    rows = []
    count = max(len(candidate_rows), len(reference_rows))
    for i in range(count):
        c = candidate_rows[i] if i < len(candidate_rows) else {}
        r = reference_rows[i] if i < len(reference_rows) else {}
        c_tokens = c.get("token_ids", [])
        r_tokens = r.get("token_ids", [])
        rows.append(
            {
                "index": i,
                "candidate_prompt": c.get("prompt"),
                "reference_prompt": r.get("prompt"),
                "prompt_match": c.get("prompt") == r.get("prompt"),
                "candidate_token_ids": c_tokens,
                "reference_token_ids": r_tokens,
                "token_ids_match": c_tokens == r_tokens,
            }
        )
    all_match = bool(rows) and all(row["token_ids_match"] for row in rows)
    return {
        "kind": "token_audit",
        "status": "pass" if all_match else "fail",
        "summary": {
            "candidate_prompt_count": len(candidate_rows),
            "reference_prompt_count": len(reference_rows),
            "all_token_ids_match": all_match,
        },
        "rows": rows,
        "notes": [] if rows else ["no prompt/token rows found in metadata"],
    }

