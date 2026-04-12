"""Export UCE core transformer to ONNX (FP32, opset 17).

The pe_embedding stays out of the graph: the exported model consumes
(seq_len, batch, token_dim) float src + (batch, seq_len) float mask, matching
TransformerModel.forward().
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _core import (
    ARTIFACT_DIR,
    CKPT_4L,
    DEFAULT_BATCH,
    DEFAULT_SEQ_LEN,
    ensure_artifact_dir,
    load_core_model,
    synthetic_inputs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_4L))
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = load_core_model(args.ckpt, args.nlayers)
    model = loaded.model.to("cpu").eval()

    src, mask = synthetic_inputs(
        seq_len=args.seq_len,
        batch=args.batch,
        pe_embedding=loaded.pe_embedding,
        seed=0,
    )

    out_dir = ensure_artifact_dir()
    out_path = Path(args.out) if args.out else out_dir / f"core_{args.nlayers}l_fp32.onnx"

    print(f"exporting → {out_path} (opset {args.opset})")
    torch.onnx.export(
        model,
        (src, mask),
        str(out_path),
        input_names=["src", "mask"],
        output_names=["gene_output", "embedding"],
        dynamic_axes={
            "src": {0: "seq_len", 1: "batch"},
            "mask": {0: "batch", 1: "seq_len"},
            "gene_output": {0: "seq_len", 1: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"done: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
