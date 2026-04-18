# uce-edge

Exploring whether [UCE](https://github.com/snap-stanford/UCE) (Universal Cell Embedding), a single-cell RNA-seq foundation model, can run efficiently on researchers' machines — including in a web browser via ONNX Runtime Web + WebGPU.

## Status

The full UCE-brain pipeline (log1p normalize → weighted sampling → chromosome ordering → CLS/CHROM/PAD inserts → protein embedding gather → 8-layer transformer) now runs end-to-end in the browser on WebGPU, validated against a Python reference at every stage. **215 ms/cell** end-to-end at seq_len=2048, vs 327 ms/cell PyTorch CPU FP32 native. See [plan.md](plan.md) for the phased build-out and optimization roadmap.

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

End-to-end browser pipeline (Phases 0–6 — see [plan.md](plan.md)):

```bash
make web-install                 # one-time: npm deps + Playwright

# Phase 0-1: slice the embedding table, generate Python reference fixtures
make brain-extract-embeddings
make brain-reference-pipeline

# Each brain-phase* builds the web bundle and runs the Playwright validator
make brain-phase2                # transformer only (vs bit-exact reference)
make brain-phase3                # + browser gather
make brain-phase4                # + browser chrom-ordering / CLS-CHROM-PAD
make brain-phase5                # + browser weighted sampling
make brain-phase6                # + browser log1p + sum-to-1 normalize (full pipeline)

# Backend/options bench (WebGPU vs WASM, batching, int8, thread counts)
make brain-bench2
```

Individual steps:

| Target | Description |
|--------|-------------|
| `brain-baseline` | Run UCE-brain on MPS, save reference outputs |
| `brain-onnx-export` | Export to ONNX (opset 17, dynamo) |
| `brain-compare` | Compare PyTorch vs ONNX FP32 vs INT8 on CPU |
| `brain-web-bench` | Playwright-driven WebGPU + WASM benchmarks (synthetic) |
| `brain-phase{2..6}` | Phase-by-phase browser pipeline validation vs Python reference |
| `brain-bench2` | Backend/options benchmark (WebGPU batch size, WASM threads, int8) |

## Findings

### Architecture comparison

| | UCE original (4L) | UCE-brain (8L) |
|---|---|---|
| d_model | 1280 | 512 |
| Non-embedding params | 106M | 30M |
| ONNX FP32 size | 373 MB | 117 MB |
| ONNX INT8 size | 100 MB | 33 MB |

### Synthetic transformer-only benchmark (MacBook Air M4, 32GB)

UCE-brain 8-layer, seq_len=128, batch=1 (initial scoping benchmark, no preprocessing):

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

### Full pipeline benchmark at real inference shape (seq_len=2048)

End-to-end, raw counts → cell embedding (Phase 6, MacBook Air M4):

```
stage                          time
log1p + sum-to-1 normalize     0.3 ms
weighted sample + sentence     0.5 ms
gather + transformer (WebGPU)  214 ms
─────────────────────────────────────
total per cell                 215 ms
```

Backend comparison at seq_len=2048 (from `make brain-bench2`):

```
config                          ms/cell
WebGPU FP32 batch=1             215   ← default
WebGPU FP32 batch=2             452   (O(L²) attention)
WebGPU FP32 batch=4             349
WebGPU INT8                     949
WASM SIMD 1 thread              1341
WASM SIMD 4 threads             1354  (no gain from threading)
WASM SIMD 10 threads            1394
PyTorch CPU FP32 (reference)    327   ← browser is 1.5× faster
```

### Key takeaways

1. **Full pipeline in the browser works.** Not just the transformer — log1p/normalize, weighted sampling with an in-JS RNG, chromosome ordering, and gather all run in TypeScript with cosine similarity against Python within the intrinsic RNG noise floor (per-cell cos 0.92–0.97 on the allen-cortex h5ad; Python-vs-Python at different seeds sits in the same range).
2. **WebGPU batch=1 FP32 is the right default.** Batching hurts per-cell (O(L²) attention), INT8 regresses (no GPU-native int8 kernels), and WASM threading is flat. Graph-optimization levels make no difference — the exported model is already fused.
3. **FP32 is faster than a local Python reference.** 215 ms/cell WebGPU vs 327 ms/cell PyTorch CPU FP32 on the same hardware. A 100-cell h5ad processes in ~22 s in-browser.
4. **First-visit cost is the real UX issue.** ~400 MB protein embedding table + 117 MB model download, then HTTP-cached. Runtime per-tab GPU peak ~1 GB.
5. **WASM threading requires cross-origin isolation.** Without COOP/COEP headers, `ort.env.wasm.numThreads` is silently ignored. The dev server in `scripts/brain_web_bench2.py` sends the right headers; a deployed app must too. Even with threading properly enabled it didn't help this model.
6. **UCE-brain's smaller architecture** (8 layers, d_model=512) is the right candidate for edge deployment — 3.5× smaller than the original UCE with equivalent design.

### Optimization headroom (not yet implemented)

Ranked effort:payoff — see [plan.md](plan.md) for detail:

- **Dynamic seq_len** (skip padded tokens): real cells are ~1100 valid of 2048 tokens; attention is O(L²), so trimming gets ~3× on the dominant cost. Estimated **~100 ms/cell**.
- **FP16 WebGPU weights**: transformer is memory-bandwidth-bound; halving weights roughly halves kernel time. Combines with the above toward **~60 ms/cell**.
- **GPU-resident embedding table + persistent session**: do the 5120-wide gather on GPU instead of shipping src through CPU; pool tensors across runs. Modest latency win, much lower memory churn.
- **`enableGraphCapture: true`** on WebGPU once shapes are stable. 10–30% per ORT docs, untested here.

### Full 33-layer UCE feasibility

The original 33-layer UCE model is not a viable candidate for browser deployment:

- **Size**: The core transformer (excluding the 2.8 GB protein embedding table) is ~870M params = **3.3 GB FP32** as an ONNX file. This exceeds practical WebGPU memory budgets on most machines and is not a reasonable browser download, even cached.
- **Architecture**: The original UCE uses `batch_first=False` (seq-first tensor layout), which produces ONNX graphs that fail on both CoreML and have not been validated on WebGPU. UCE-brain's `batch_first=True` layout produces a cleaner export that runs correctly.
- **Compute**: 33 layers at d_model=1280 is roughly **100x** the compute of UCE-brain's 8 layers at d_model=512 (33/8 layer ratio × (1280/512)² attention × (1280/512) FFN). Extrapolating from brain's 14 ms WebGPU time, the 33-layer model would take ~1.4 seconds per forward pass — usable but not interactive.
- **Design intent**: The 33-layer model was designed for server-side GPU inference. UCE-brain was explicitly designed to be smaller while retaining the same architecture pattern, making it the right candidate for edge deployment.

### What's not yet covered

- Native h5ad parsing in the browser (the pipeline assumes the caller provides `(gene_symbols[], raw_counts[N,G])` in memory however they got there).
- Non-human species.
- The expression-prediction head (embedding extraction only).
- A polished demo UI — the phase*.html pages are validation harnesses, not product.
- The optimization phases listed above.
