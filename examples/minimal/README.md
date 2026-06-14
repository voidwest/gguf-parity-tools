# Minimal Example

This directory contains prompt and metadata examples only. Generate toy logits
with NumPy when you want to exercise the CLI without a real model:

```bash
python - <<'PY'
import numpy as np
np.save("candidate.npy", np.array([[0, 3, 2], [4, 1, 0]], dtype=np.float32))
np.save("reference.npy", np.array([[0, 3.1, 1.9], [4, 1, 0]], dtype=np.float32))
PY

gguf-parity compare-logits \
  --candidate candidate.npy \
  --reference reference.npy \
  --candidate-metadata metadata_candidate.json \
  --reference-metadata metadata_reference.json \
  --out report
```

