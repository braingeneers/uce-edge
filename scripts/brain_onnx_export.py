"""Export UCE-brain 8L core to ONNX (FP32, opset 17)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _brain import (
    DEFAULT_BATCH,
    DEFAULT_SEQ_LEN,
    ensure_artifact_dir,
    load_brain,
    synthetic_inputs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loaded = load_brain()
    core = loaded.core.to("cpu").eval()
    src, mask = synthetic_inputs(
        seq_len=args.seq_len, batch=args.batch, embedding=loaded.embedding, seed=0,
    )

    out_dir = ensure_artifact_dir()
    out_path = Path(args.out) if args.out else out_dir / "brain_8l_fp32.onnx"

    print(f"exporting → {out_path} (opset {args.opset})")
    torch.onnx.export(
        core,
        (src, mask),
        str(out_path),
        input_names=["src", "mask"],
        output_names=["gene_embeddings", "cell_embedding"],
        dynamic_axes={
            "src": {0: "batch", 1: "seq_len"},
            "mask": {0: "batch", 1: "seq_len"},
            "gene_embeddings": {0: "batch", 1: "seq_len"},
            "cell_embedding": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"done: {out_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
