from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_logits(path: str | Path, key: str = "logits") -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        archive = np.load(p)
        if key in archive:
            arr = archive[key]
        elif len(archive.files) == 1:
            arr = archive[archive.files[0]]
        else:
            raise ValueError(f"{p} has no '{key}' key; available keys: {archive.files}")
    else:
        arr = np.load(p)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"logits must have shape [prompts, vocab] or [vocab], got {arr.shape}")
    return arr


def layer_dims_from_metadata(metadata: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not metadata:
        return None, None
    layers = metadata.get("layers") or metadata.get("n_layers")
    hidden = metadata.get("hidden_size") or metadata.get("n_embd") or metadata.get("embedding_length")
    return (int(layers) if layers is not None else None, int(hidden) if hidden is not None else None)


def load_layers(path: str | Path, layers: int, hidden_size: int) -> np.ndarray:
    if layers <= 0 or hidden_size <= 0:
        raise ValueError(f"layers and hidden_size must be positive, got {layers} and {hidden_size}")
    arr = np.fromfile(path, dtype=np.float32)
    expected = layers * hidden_size
    if arr.size != expected:
        raise ValueError(
            f"expected {expected} float32 values in {path}, got {arr.size}; "
            f"check layer count and hidden size"
        )
    return arr.reshape(layers, hidden_size)
