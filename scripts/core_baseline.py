"""Baseline: run UCE core transformer on MPS with synthetic input.

Saves reference outputs + inputs to data/core_artifacts/ for later comparison.
"""
from __future__ import annotations

import argparse

import torch

from _core import (
    ARTIFACT_DIR,
    CKPT_4L,
    DEFAULT_BATCH,
    DEFAULT_SEQ_LEN,
    ensure_artifact_dir,
    load_core_model,
    pick_device,
    synthetic_inputs,
    time_forward,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_4L))
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default=None, help="cpu|mps (default: mps if available)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    print(f"loading checkpoint: {args.ckpt}")
    loaded = load_core_model(args.ckpt, args.nlayers)
    print(f"nlayers={loaded.nlayers}")

    print(f"building synthetic input: seq_len={args.seq_len} batch={args.batch}")
    src, mask = synthetic_inputs(
        seq_len=args.seq_len,
        batch=args.batch,
        pe_embedding=loaded.pe_embedding,
        seed=args.seed,
    )
    print(f"  src {tuple(src.shape)} {src.dtype}")
    print(f"  mask {tuple(mask.shape)} {mask.dtype}")

    gene_output, embedding, per_iter = time_forward(
        loaded.model, src, mask, device, warmup=2, iters=5
    )

    print(f"forward: {per_iter*1000:.1f} ms/iter ({5} iters averaged)")
    print(f"  gene_output {tuple(gene_output.shape)} "
          f"mean={gene_output.mean().item():+.4f} std={gene_output.std().item():.4f}")
    print(f"  embedding   {tuple(embedding.shape)} "
          f"norm={embedding.norm(dim=1).mean().item():.4f}")

    out_dir = ensure_artifact_dir()
    torch.save(
        {
            "src": src,
            "mask": mask,
            "gene_output": gene_output,
            "embedding": embedding,
            "device": str(device),
            "per_iter_s": per_iter,
            "seq_len": args.seq_len,
            "batch": args.batch,
            "nlayers": loaded.nlayers,
            "seed": args.seed,
        },
        out_dir / "baseline.pt",
    )
    print(f"saved reference → {out_dir/'baseline.pt'}")


if __name__ == "__main__":
    main()
