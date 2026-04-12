"""Compare UCE-brain PyTorch baseline vs ONNX FP32 (CPU + CoreML) vs INT8."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from _brain import ARTIFACT_DIR


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def run_ort(model_path, src, mask, providers, warmup=2, iters=5):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    used = sess.get_providers()[0]
    feed = {"src": src, "mask": mask}
    for _ in range(warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        ge, ce = sess.run(None, feed)
    elapsed = (time.perf_counter() - t0) / iters
    return ge, ce, elapsed, used


def report_row(name, size_mb, ms, ref_cell, cell, ref_gene, gene, provider):
    cell_diff = float(np.abs(ref_cell - cell).max())
    gene_diff = float(np.abs(ref_gene - gene).max())
    cos = cosine(ref_cell, cell)
    print(f"  {name:<22} {size_mb:>7.1f} MB  {ms*1000:>8.1f} ms  "
          f"cell_maxdiff={cell_diff:.2e}  cos={cos:.6f}  "
          f"gene_maxdiff={gene_diff:.2e}  [{provider}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true",
                    help="(re)quantize the FP32 ONNX to INT8")
    args = ap.parse_args()

    baseline_path = ARTIFACT_DIR / "baseline.pt"
    fp32_path = ARTIFACT_DIR / "brain_8l_fp32.onnx"
    int8_path = ARTIFACT_DIR / "brain_8l_int8.onnx"

    if args.quantize or not int8_path.exists():
        print(f"quantizing {fp32_path.name} → {int8_path.name}")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

    ref = torch.load(baseline_path, map_location="cpu", weights_only=True)
    src = ref["src"].numpy()
    mask = ref["mask"].numpy()
    ref_gene = ref["gene"].numpy()
    ref_cell = ref["cell"].numpy()
    pt_ms = float(ref["per_iter_s"])
    pt_device = ref["device"]

    print()
    print(f"  {'variant':<22} {'size':>10}  {'time':>11}  diffs vs PyTorch baseline")
    print(f"  {'-'*22} {'-'*10}  {'-'*11}  {'-'*60}")
    print(f"  {'pytorch baseline':<22} {'  —':>10}  {pt_ms*1000:>8.1f} ms  "
          f"(reference)  [{pt_device}]")

    available = ort.get_available_providers()
    cpu = ["CPUExecutionProvider"]
    coreml = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
              if "CoreMLExecutionProvider" in available else None)

    for path, label in [(fp32_path, "onnx fp32 cpu"), (int8_path, "onnx int8 cpu")]:
        if not path.exists():
            print(f"  {label:<22}  (missing: {path.name})")
            continue
        ge, ce, ms, prov = run_ort(path, src, mask, cpu)
        report_row(label, path.stat().st_size / 1e6, ms, ref_cell, ce, ref_gene, ge, prov)

    if coreml and fp32_path.exists():
        try:
            ge, ce, ms, prov = run_ort(fp32_path, src, mask, coreml)
            report_row("onnx fp32 coreml", fp32_path.stat().st_size / 1e6, ms,
                       ref_cell, ce, ref_gene, ge, prov)
        except Exception as e:
            print(f"  {'onnx fp32 coreml':<22}  (failed: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
