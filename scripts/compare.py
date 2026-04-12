"""Compare PyTorch baseline vs ONNX FP32 vs ONNX INT8.

Loads the reference tensors saved by core_baseline.py and runs the same inputs
through each ONNX variant via onnxruntime. Reports per-variant wall clock,
on-disk size, max abs diff, and cosine similarity of the CLS embedding.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from _core import ARTIFACT_DIR


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    return float((num / den).mean())


def run_ort(
    model_path: Path,
    src: np.ndarray,
    mask: np.ndarray,
    providers: list[str],
    warmup: int = 2,
    iters: int = 5,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    used = sess.get_providers()[0]

    feed = {"src": src, "mask": mask}
    for _ in range(warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        gene_output, embedding = sess.run(None, feed)
    elapsed = (time.perf_counter() - t0) / iters
    return gene_output, embedding, elapsed, used


def report_row(name: str, size_mb: float, ms: float, ref_emb: np.ndarray, emb: np.ndarray,
               ref_go: np.ndarray, go: np.ndarray, provider: str) -> None:
    emb_diff = float(np.abs(ref_emb - emb).max())
    go_diff = float(np.abs(ref_go - go).max())
    cos = cosine(ref_emb, emb)
    print(f"  {name:<22} {size_mb:>7.1f} MB  {ms*1000:>8.1f} ms  "
          f"emb_maxdiff={emb_diff:.2e}  cos={cos:.6f}  "
          f"go_maxdiff={go_diff:.2e}  [{provider}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlayers", type=int, default=4)
    args = ap.parse_args()

    baseline_path = ARTIFACT_DIR / "baseline.pt"
    fp32_path = ARTIFACT_DIR / f"core_{args.nlayers}l_fp32.onnx"
    int8_path = ARTIFACT_DIR / f"core_{args.nlayers}l_int8.onnx"

    print(f"loading reference: {baseline_path}")
    ref = torch.load(baseline_path, map_location="cpu", weights_only=True)
    src = ref["src"].numpy()
    mask = ref["mask"].numpy()
    ref_go = ref["gene_output"].numpy()
    ref_emb = ref["embedding"].numpy()
    pt_ms = float(ref["per_iter_s"])
    pt_device = ref["device"]

    ckpt_size_mb = (Path(ref.get("ckpt", "")).stat().st_size / 1e6) if False else float("nan")
    print()
    print(f"  {'variant':<22} {'size':>10}  {'time':>11}  diffs vs PyTorch baseline")
    print(f"  {'-'*22} {'-'*10}  {'-'*11}  {'-'*60}")
    print(f"  {'pytorch baseline':<22} {'  —':>10}  {pt_ms*1000:>8.1f} ms  "
          f"(reference)  [{pt_device}]")

    available = ort.get_available_providers()
    cpu_providers = ["CPUExecutionProvider"]
    coreml_providers = (["CoreMLExecutionProvider", "CPUExecutionProvider"]
                        if "CoreMLExecutionProvider" in available else None)

    for path, label in [(fp32_path, "onnx fp32 cpu"), (int8_path, "onnx int8 cpu")]:
        if not path.exists():
            print(f"  {label:<22}  (missing: {path.name})")
            continue
        go, emb, ms, prov = run_ort(path, src, mask, cpu_providers)
        report_row(label, path.stat().st_size / 1e6, ms, ref_emb, emb, ref_go, go, prov)

    if coreml_providers and fp32_path.exists():
        try:
            go, emb, ms, prov = run_ort(fp32_path, src, mask, coreml_providers)
            report_row("onnx fp32 coreml", fp32_path.stat().st_size / 1e6, ms,
                       ref_emb, emb, ref_go, go, prov)
        except Exception as e:
            print(f"  {'onnx fp32 coreml':<22}  (failed: {type(e).__name__})")


if __name__ == "__main__":
    main()
