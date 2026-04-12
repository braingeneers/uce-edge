"""Baseline forward of UCE-brain 8L pilot on MPS, save reference."""
from __future__ import annotations

import argparse

import torch

from _brain import (
    DEFAULT_BATCH,
    DEFAULT_SEQ_LEN,
    ensure_artifact_dir,
    load_brain,
    pick_device,
    synthetic_inputs,
    time_forward,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    print("loading UCE-brain 8L pilot")
    loaded = load_brain()
    print(f"  d_model={loaded.config.d_model} nlayers={loaded.config.num_layers} "
          f"nhead={loaded.config.nhead} max_seq={loaded.config.max_sequence_length}")

    print(f"building synthetic input: batch={args.batch} seq_len={args.seq_len}")
    src, mask = synthetic_inputs(
        seq_len=args.seq_len, batch=args.batch,
        embedding=loaded.embedding, seed=args.seed,
    )
    print(f"  src {tuple(src.shape)} {src.dtype}")
    print(f"  mask {tuple(mask.shape)} {mask.dtype}")

    gene, cell, per_iter = time_forward(loaded.core, src, mask, device, warmup=2, iters=5)
    print(f"forward: {per_iter*1000:.1f} ms/iter")
    print(f"  gene_embeddings {tuple(gene.shape)} mean={gene.mean():.4f} std={gene.std():.4f}")
    print(f"  cell_embedding  {tuple(cell.shape)} norm={cell.norm(dim=1).mean():.4f}")

    out_dir = ensure_artifact_dir()
    torch.save(
        {
            "src": src, "mask": mask,
            "gene": gene, "cell": cell,
            "device": str(device),
            "per_iter_s": per_iter,
            "seq_len": args.seq_len, "batch": args.batch,
        },
        out_dir / "baseline.pt",
    )
    print(f"saved → {out_dir/'baseline.pt'}")


if __name__ == "__main__":
    main()
