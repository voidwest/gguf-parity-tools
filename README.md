# gguf-parity-tools

Small tools for comparing a GGUF inference implementation against a reference
run such as llama.cpp, Hugging Face, or another audited engine.

The first goal is practical debugging:

- verify that token IDs match before comparing numbers
- compare final-position logits across prompts
- compare per-layer hidden-state dumps
- write machine-readable JSON plus Markdown/CSV reports

This is not a benchmark harness and it does not claim model-quality parity.
Close logits are evidence for a specific model, prompt, tokenizer, quantization,
and implementation path.

## Install for development

```bash
python -m pip install -e .
```

## Artifact contracts

Any engine can participate if it can emit these files.

### Logits

`logits.npy` or `logits.npz`

- shape: `[prompts, vocab]` or `[vocab]`
- dtype: any numeric type accepted by NumPy, converted to `float32`
- default `.npz` key: `logits`

### Metadata

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

### Layers

`layers.bin`

- dtype: native-endian `float32`
- shape: `[layers, hidden_size]`
- row order: layer 0 first

Layer dimensions can be passed on the CLI or provided in a metadata JSON file:

```json
{
  "layers": 35,
  "hidden_size": 1536,
  "prompt": "",
  "token_ids": [2]
}
```

## Commands

Compare logits:

```bash
gguf-parity compare-logits \
  --candidate ember_logits.npz \
  --reference llamacpp_logits.npz \
  --candidate-metadata ember_metadata.json \
  --reference-metadata llamacpp_metadata.json \
  --out report
```

Compare layers:

```bash
gguf-parity compare-layers \
  --candidate ember_layers.bin \
  --reference llama_layers.bin \
  --layers 35 \
  --hidden-size 1536 \
  --out report
```

Audit token IDs only:

```bash
gguf-parity token-audit \
  --candidate-metadata ember_metadata.json \
  --reference-metadata llamacpp_metadata.json
```

## llama.cpp integration

Layer comparison requires a patched llama.cpp build that can dump internal
per-layer tensors. See `tools/dump_llamacpp_layers.cpp` and
`docs/llamacpp-layer-dump.md`.

