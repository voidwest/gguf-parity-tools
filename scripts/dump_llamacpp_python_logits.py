#!/usr/bin/env python3
"""Dump final-position logits with llama-cpp-python.

This adapter is optional and requires:

    python -m pip install llama-cpp-python

It writes the standard gguf-parity logits contract:

    logits.npz      keys: logits, token_ids, token_lengths
    metadata.json   prompt text, token IDs, model path, run settings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from llama_cpp import Llama


def read_prompts(path: str | Path) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="dump llama-cpp-python final-position logits")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--no-bos", action="store_true")
    args = parser.parse_args()

    prompts = read_prompts(args.prompts)
    add_bos = not args.no_bos
    llm = Llama(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        logits_all=True,
        verbose=False,
    )

    logits_rows = []
    token_rows = []
    metadata_prompts = []

    for index, prompt in enumerate(prompts):
        token_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=add_bos)
        llm.reset()
        llm.eval(token_ids)
        # llama-cpp-python stores logits by evaluated token position. The last
        # allocated context row may be untouched zeros for short prompts.
        logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float32)
        logits_rows.append(logits)
        token_rows.append(np.asarray(token_ids, dtype=np.int32))
        metadata_prompts.append(
            {
                "index": index,
                "prompt": prompt,
                "token_ids": [int(t) for t in token_ids],
                "num_tokens": len(token_ids),
            }
        )

    max_len = max((len(row) for row in token_rows), default=0)
    token_matrix = np.full((len(token_rows), max_len), -1, dtype=np.int32)
    token_lengths = np.zeros((len(token_rows),), dtype=np.int32)
    for i, row in enumerate(token_rows):
        token_matrix[i, : len(row)] = row
        token_lengths[i] = len(row)

    logits_matrix = np.stack(logits_rows, axis=0).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, logits=logits_matrix, token_ids=token_matrix, token_lengths=token_lengths)

    metadata = {
        "engine": "llama-cpp-python",
        "model": args.model,
        "prompts_path": args.prompts,
        "n_ctx": args.n_ctx,
        "n_threads": args.n_threads,
        "add_bos": add_bos,
        "num_prompts": len(prompts),
        "vocab_size": int(logits_matrix.shape[-1]),
        "logits_shape": list(logits_matrix.shape),
        "prompts": metadata_prompts,
        "output_npz": str(out),
    }
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote logits: {out}")
    print(f"wrote metadata: {args.metadata}")


if __name__ == "__main__":
    main()
