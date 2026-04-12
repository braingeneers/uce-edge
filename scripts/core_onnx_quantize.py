"""Dynamic INT8 quantization of the exported ONNX model."""
from __future__ import annotations

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from _core import ARTIFACT_DIR, ensure_artifact_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--in-model", default=None)
    ap.add_argument("--out-model", default=None)
    args = ap.parse_args()

    out_dir = ensure_artifact_dir()
    in_path = Path(args.in_model) if args.in_model else out_dir / f"core_{args.nlayers}l_fp32.onnx"
    out_path = Path(args.out_model) if args.out_model else out_dir / f"core_{args.nlayers}l_int8.onnx"

    print(f"quantizing {in_path} → {out_path} (dynamic INT8)")
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    fp32_mb = in_path.stat().st_size / 1e6
    int8_mb = out_path.stat().st_size / 1e6
    print(f"fp32: {fp32_mb:.1f} MB  int8: {int8_mb:.1f} MB  ratio: {fp32_mb/int8_mb:.2f}x")


if __name__ == "__main__":
    main()
