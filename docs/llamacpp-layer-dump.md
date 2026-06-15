# llama.cpp Layer Dumps

`gguf-parity compare-layers` compares flat `float32` hidden-state dumps. It does
not depend on llama.cpp directly, but llama.cpp is a useful reference if patched
to expose internal layer tensors.

## Output Contract

The reference dump should be:

- native-endian `float32`
- flat row-major array
- shape `[layers, hidden_size]`
- one row per layer, layer 0 first
- vector for the final prompt token unless metadata says otherwise

Example metadata:

```json
{
  "engine": "llama.cpp",
  "layers": 35,
  "hidden_size": 1536,
  "prompt": "",
  "token_ids": [2],
  "capture_point": "block_output_after_residual"
}
```

## Helper

`tools/dump_llamacpp_layers.cpp` is a small helper intended to compile against a
patched llama.cpp checkout. The current helper documents the required llama.cpp
patch points in its source comments.

Build shape:

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF
cmake --build build --target llama -j"$(nproc)"
g++ -std=c++17 -I./include -I./ggml/include -I./src \
  /path/to/gguf-parity-tools/tools/dump_llamacpp_layers.cpp \
  ./build/src/libllama.a \
  ./build/ggml/src/libggml.a \
  ./build/ggml/src/libggml-base.a \
  ./build/ggml/src/libggml-cpu.a \
  -lpthread -ldl -lm -o dump_llamacpp_layers
```

Run shape:

```bash
./dump_llamacpp_layers model.gguf "" llama_layers.bin 16
```

The helper treats `llama_layers.bin` as the success contract. If the patched
decode path does not create a non-empty output file, the helper exits non-zero
instead of silently writing a single final hidden-state row. For debugging only,
pass `--allow-fallback` to write that single-row fallback:

```bash
./dump_llamacpp_layers model.gguf "" final_hidden.bin 16 --allow-fallback
```

Then compare:

```bash
gguf-parity compare-layers \
  --candidate engine_layers.bin \
  --reference llama_layers.bin \
  --layers 35 \
  --hidden-size 1536 \
  --out report
```

## Notes

Layer dumps only help if both engines capture the same tensor boundary. The
metadata should name the capture point precisely: pre-norm, post-attention,
post-residual, post-layer-scale, or final block output.
