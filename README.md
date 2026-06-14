# gguf-parity-tools

[![python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

small reference-comparison tools for gguf inference implementations. compares
candidate logits and hidden-state dumps against a trusted run from llama.cpp,
Hugging Face, or another audited engine.

the goal is boring parity debugging: make tokenizer mismatches, bos/eos policy
differences, logit drift, and first divergent layers visible before a model is
treated as numerically validated.

## what gguf-parity-tools is / is not

gguf-parity-tools is a file-based parity harness. any engine can participate if
it writes simple artifacts such as `logits.npz`, `metadata.json`, or
`layers.bin`.

gguf-parity-tools is not an inference engine, benchmark suite, or claim that a
model is production-ready. close logits are evidence for one exact model,
tokenizer, prompt set, quantization, and implementation path. they do not prove
generation quality, throughput, or broad architecture coverage.

the current reference path is llama.cpp via `llama-cpp-python` for final logits
and patched llama.cpp for internal layer dumps.

## validation ladder

Use these levels when interpreting a parity run:

1. **artifact smoke**: files load, shapes match, and a report is produced. this
   is not numerical validation.
2. **token audit**: prompt text, token ids, and bos/eos policy match between
   candidate and reference.
3. **golden logits**: final-position logits match a trusted reference closely
   enough for the configured thresholds.
4. **activation reference checks**: per-layer hidden states are compared at a
   documented tensor boundary.
5. **regression suite**: the same checks run repeatedly across prompts, models,
   quantization settings, and implementation commits.

## current evidence status

| path | status | notes |
|------|--------|-------|
| synthetic tests | passing | unit tests cover shape mismatch, empty arrays, non-finite values, top-k edge cases, npz key handling, and layer dimension validation |
| qwen3 0.6b smoke | passing locally | ember vs llama-cpp-python on `Hello world`: top-1 match, top-10 overlap 1.0, cosine ~0.999789 |
| llama-cpp-python logits | usable | optional adapter writes `logits.npz` and `metadata.json`; requires `llama-cpp-python` |
| patched llama.cpp layers | scaffolded | helper and docs exist; end-to-end patched llama.cpp build is still manual |
| packaging | local/dev | `pyproject.toml` and console script are present; not published to pypi |

## features

- **logit comparison**: compares `[prompts, vocab]` arrays from `.npy` or
  `.npz`, with top-1 match, top-k overlap, cosine, and absolute-difference
  metrics.
- **layer comparison**: compares flat native-endian f32 layer dumps reshaped as
  `[layers, hidden_size]`.
- **token audit**: compares prompt rows and token ids from metadata before
  trusting numeric comparisons.
- **report bundles**: writes `report.json`, `summary.md`, and CSV rows for
  downstream analysis.
- **adapter-first design**: core commands compare files; engine-specific dump
  scripts live outside the comparison logic.
- **llama.cpp reference path**: includes a `llama-cpp-python` logit dumper and a
  C++ helper for patched llama.cpp layer dumps.
- **edge-case handling**: reports shape mismatch, empty arrays, non-finite
  values, ambiguous npz keys, and invalid layer dimensions explicitly.

## usage

Install for local development:

```bash
python -m pip install -e .
```

Compare final-position logits:

```bash
gguf-parity compare-logits \
  --candidate ember_logits.npz \
  --reference llamacpp_logits.npz \
  --candidate-metadata ember_metadata.json \
  --reference-metadata llamacpp_metadata.json \
  --top-k 10 \
  --require-top1 \
  --min-topk-overlap 0.8 \
  --out report
```

Compare per-layer hidden-state dumps:

```bash
gguf-parity compare-layers \
  --candidate ember_layers.bin \
  --reference llama_layers.bin \
  --layers 35 \
  --hidden-size 1536 \
  --out report
```

Audit token ids only:

```bash
gguf-parity token-audit \
  --candidate-metadata ember_metadata.json \
  --reference-metadata llamacpp_metadata.json
```

### commands

| command | purpose |
|---------|---------|
| `compare-logits` | compare candidate and reference final-position logits |
| `compare-layers` | compare candidate and reference layer dumps |
| `token-audit` | compare prompt/token metadata without loading logits |

### compare-logits flags

| flag | default | description |
|------|---------|-------------|
| `--candidate` | required | candidate `.npy` or `.npz` logits |
| `--reference` | required | reference `.npy` or `.npz` logits |
| `--candidate-key` | `logits` | candidate `.npz` key |
| `--reference-key` | `logits` | reference `.npz` key |
| `--candidate-metadata` | none | candidate metadata json |
| `--reference-metadata` | none | reference metadata json |
| `--top-k` | `10` | number of top tokens to compare |
| `--max-abs` | none | fail if any row exceeds this max absolute diff |
| `--mean-abs` | none | fail if any row exceeds this mean absolute diff |
| `--min-cosine` | none | fail if any row is below this cosine |
| `--min-topk-overlap` | `0.8` | fail if top-k overlap ratio is lower |
| `--require-top1` | false | fail if any top-1 token differs |
| `--out` | stdout markdown | output directory for report bundle |

### compare-layers flags

| flag | default | description |
|------|---------|-------------|
| `--candidate` | required | candidate flat f32 layer dump |
| `--reference` | required | reference flat f32 layer dump |
| `--metadata` | none | optional json with `layers` and `hidden_size` |
| `--layers` | metadata | number of layer rows |
| `--hidden-size` | metadata | hidden size per layer |
| `--out` | stdout markdown | output directory for report bundle |

## artifact contracts

Any engine can participate if it emits the same simple file shapes.

### logits

`logits.npy` or `logits.npz`

- shape: `[prompts, vocab]` or `[vocab]`
- dtype: numeric dtype accepted by numpy, converted to `float32`
- default `.npz` key: `logits`

### metadata

`metadata.json`

```json
{
  "engine": "ember",
  "model": "model.gguf",
  "model_sha256": "...",
  "tokenizer": "tokenizer.json",
  "prompts": [
    {
      "index": 0,
      "prompt": "Hello",
      "token_ids": [128000, 9906]
    }
  ],
  "arch": "llama",
  "quantization": "Q8_0"
}
```

### layers

`layers.bin`

- dtype: native-endian `float32`
- shape: `[layers, hidden_size]`
- row order: layer 0 first
- capture point must be documented by metadata

Layer dimensions can be passed on the CLI or provided in metadata:

```json
{
  "layers": 35,
  "hidden_size": 1536,
  "prompt": "",
  "token_ids": [2],
  "capture_point": "block_output_after_residual"
}
```

## llama.cpp integration

Dump logits with `llama-cpp-python`:

```bash
python scripts/dump_llamacpp_python_logits.py \
  --model model.gguf \
  --prompts prompts.txt \
  --out runs/llama/logits.npz \
  --metadata runs/llama/metadata.json
```

Layer comparison requires a patched llama.cpp build that can expose internal
per-layer tensors. See:

- `tools/dump_llamacpp_layers.cpp`
- `docs/llamacpp-layer-dump.md`

## development

Run the test suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Run a minimal local example:

```bash
cd examples/minimal
python - <<'PY'
import numpy as np
np.save("candidate.npy", np.array([[0, 3, 2], [4, 1, 0]], dtype=np.float32))
np.save("reference.npy", np.array([[0, 3.1, 1.9], [4, 1, 0]], dtype=np.float32))
PY

python -m parity_tools compare-logits \
  --candidate candidate.npy \
  --reference reference.npy \
  --candidate-metadata metadata_candidate.json \
  --reference-metadata metadata_reference.json \
  --out report
```

## known limits

- no pypi package yet.
- no stock llama.cpp layer dump path; internal activations require patching.
- no plotting layer yet.
- no built-in engine runner beyond the optional llama-cpp-python adapter.
- token audit only runs when both candidate and reference metadata are present.
