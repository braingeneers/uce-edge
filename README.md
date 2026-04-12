# uce-edge

Exploring whether [UCE](https://github.com/snap-stanford/UCE) (Universal Cell Embedding), a single-cell RNA-seq foundation model, can run efficiently on researchers' machines — including in a web browser via ONNX Runtime Web + WebGPU.

## Repository structure

- **UCE/** — original UCE repo (submodule), 4-layer and 33-layer models
- **UCE-brain/** — newer [UCE-brain](https://github.com/snap-stanford/UCE-brain) repo (submodule), 8-layer model with smaller architecture (d_model=512 vs 1280)
- **model_files/** — pre-trained weights and supporting files (not checked in)
- **scripts/** — Python harnesses for baseline, ONNX export, quantization, and benchmarking
- **web/** — browser-based inference demo using ONNX Runtime Web

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone --recurse-submodules <repo-url> && cd uce-edge
uv sync
```

For browser benchmarks, install Playwright's Chromium:

```bash
.venv/bin/playwright install chromium
```

Model weights need to be placed in `model_files/`. The UCE-brain checkpoint downloads automatically from HuggingFace on first run:

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='KuanP/uce-brain-pilot-8l-512d', local_dir='model_files/uce-brain-pilot-8l-512d')
"
```

## Running

All experiments are available as Makefile targets:

```bash
# Original UCE (4-layer) — baseline, ONNX export, INT8 quantize, compare
make core-all

# UCE-brain (8-layer) — baseline, ONNX export, compare
make brain-all

# Automated browser benchmarks via Playwright (WebGPU + WASM)
make brain-web-bench

# Interactive browser demo (manual)
.venv/bin/python -m http.server 8765 -d web
# then open http://localhost:8765
```

Individual steps:

| Target | Description |
|--------|-------------|
| `brain-baseline` | Run UCE-brain on MPS, save reference outputs |
| `brain-onnx-export` | Export to ONNX (opset 17, dynamo) |
| `brain-compare` | Compare PyTorch vs ONNX FP32 vs INT8 on CPU |
| `brain-web-bench` | Playwright-driven WebGPU + WASM benchmarks |

## Findings

### Architecture comparison

| | UCE original (4L) | UCE-brain (8L) |
|---|---|---|
| d_model | 1280 | 512 |
| Non-embedding params | 106M | 30M |
| ONNX FP32 size | 373 MB | 117 MB |
| ONNX INT8 size | 100 MB | 33 MB |

### Benchmark results (MacBook Air M4, 32GB)

UCE-brain 8-layer, seq_len=128, batch=1:

```
variant                  size      time      cosine vs reference
Python MPS (baseline)    —         215 ms    1.000000
Python ONNX FP32 CPU     117 MB    204 ms    1.000000
Python ONNX INT8 CPU      33 MB    201 ms    0.999672
Browser FP32 WebGPU      117 MB     14 ms    1.000000
Browser FP32 WASM        117 MB    143 ms    1.000000
Browser INT8 WebGPU       33 MB    173 ms    0.998644
Browser INT8 WASM         33 MB    145 ms    0.998706
```

### Key takeaways

1. **Browser inference is viable.** FP32 on WebGPU runs at 14 ms — 10x faster than Python on the same hardware, with bit-perfect numerical fidelity.
2. **INT8 quantization is counterproductive on WebGPU.** WebGPU has no native INT8 kernels; the dequantization overhead makes it 12x slower than FP32. INT8 only helps if download size is the binding constraint.
3. **FP32 on WebGPU is the recommended path.** 117 MB is a reasonable one-time cached download for a web application.
4. **CoreML execution provider fails** for both UCE variants (known PyTorch TransformerEncoderLayer export issue). This doesn't block the browser path, which uses WebGPU.
5. **UCE-brain's smaller architecture** (512 vs 1280 d_model) is the better candidate for edge deployment — 3.5x smaller with equivalent model design.

### Full 33-layer UCE feasibility

The original 33-layer UCE model is not a viable candidate for browser deployment:

- **Size**: The core transformer (excluding the 2.8 GB protein embedding table) is ~870M params = **3.3 GB FP32** as an ONNX file. This exceeds practical WebGPU memory budgets on most machines and is not a reasonable browser download, even cached.
- **Architecture**: The original UCE uses `batch_first=False` (seq-first tensor layout), which produces ONNX graphs that fail on both CoreML and have not been validated on WebGPU. UCE-brain's `batch_first=True` layout produces a cleaner export that runs correctly.
- **Compute**: 33 layers at d_model=1280 is roughly **100x** the compute of UCE-brain's 8 layers at d_model=512 (33/8 layer ratio × (1280/512)² attention × (1280/512) FFN). Extrapolating from brain's 14 ms WebGPU time, the 33-layer model would take ~1.4 seconds per forward pass — usable but not interactive.
- **Design intent**: The 33-layer model was designed for server-side GPU inference. UCE-brain was explicitly designed to be smaller while retaining the same architecture pattern, making it the right candidate for edge deployment.

### What's not yet covered

- Real scRNA input (experiments use synthetic data to isolate the model inference path)
- Protein embedding lookup in the browser (the 145K x 5120 embedding table is ~2.8 GB and kept outside the ONNX graph)
- End-to-end preprocessing pipeline in JavaScript
- Scaling behavior at full seq_len=1024 or seq_len=2048
