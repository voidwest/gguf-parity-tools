# Adapter Guide

Adapters are small scripts that turn an engine-specific command into standard
gguf-parity artifacts.

The core tools deliberately do not know how to run every engine. They compare
files. This keeps the comparison layer stable and makes each engine integration
replaceable.

## Required Logit Artifacts

Write:

```text
logits.npz
metadata.json
```

`logits.npz` should contain:

- `logits`: `float32`, shape `[prompts, vocab]`
- `token_ids`: optional `int32`, shape `[prompts, max_tokens]`, padded with `-1`
- `token_lengths`: optional `int32`, shape `[prompts]`

`metadata.json` should contain prompt/token rows:

```json
{
  "engine": "my-engine",
  "model": "model.gguf",
  "prompts": [
    {
      "index": 0,
      "prompt": "Hello",
      "token_ids": [1, 2, 3]
    }
  ]
}
```

## llama-cpp-python

```bash
python scripts/dump_llamacpp_python_logits.py \
  --model model.gguf \
  --prompts prompts.txt \
  --out runs/llama/logits.npz \
  --metadata runs/llama/metadata.json
```

## Candidate Engines

For another engine, the preferred integration is:

1. Read a prompt file.
2. Run the engine once per prompt with deterministic settings.
3. Save the final-position logits.
4. Save token IDs from the engine itself.
5. Record the exact command and engine commit in metadata.

If the candidate engine cannot emit token IDs yet, still write logits and mark
the metadata clearly. Token mismatch is one of the most common false parity
failures, so direct token ID emission is worth adding early.

