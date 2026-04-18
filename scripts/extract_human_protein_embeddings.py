"""Phase 0: Extract the human-only protein embedding table for browser use.

Produces:
  - web/human_protein_embeddings.bin  (FP32, dense rows, post-LayerNorm)
  - web/human_gene_dict.json          (renumbered IDs for genes + special tokens)

Rationale:
- The full UCE-brain embedding table is (145469, 5120) -> ~2.8 GB. Too big for a
  browser asset.
- For human-only inference we need 19,656 gene rows plus a small set of special
  token rows (PAD, CLS, chromosome tokens, CHROM_END). Special tokens live at
  indices {0, 1, 1564..1613, 2000} in the original table.
- The exported ONNX graph (BrainCore) consumes `src` = post-LayerNorm embeddings
  (see UCE-brain ProteinEmbeddingLayer: embedding -> layer_norm). LayerNorm is
  position-wise over the feature axis, so pre-applying it to each row of the
  table is mathematically equivalent and saves a JS step.
- Renumber IDs densely so the browser table indexes directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "model_files" / "uce-brain-pilot-8l-512d"
GENE_DICT_PATH = REPO_ROOT / "UCE-brain" / "gene_data" / "human_gene_dict.json"
OUT_DIR = REPO_ROOT / "web"

# UCE-brain sampler constants (from UCE-brain/src/uce_brain/data/sampler.py)
PAD_TOKEN_IDX = 0
CLS_TOKEN_IDX = 1
CHROM_TOKEN_OFFSET = 1000
CHROM_TOKEN_RIGHT_IDX = 2000  # CHROM_END marker

EMBEDDING_DIM = 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR)
    ap.add_argument("--gene-dict", type=Path, default=GENE_DICT_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-layer-norm", action="store_true",
                    help="Ship raw embeddings; browser must apply LayerNorm.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load full embedding table + LayerNorm params from checkpoint.
    # ------------------------------------------------------------------
    print(f"loading checkpoint: {args.ckpt}")
    state = load_file(str(args.ckpt / "model.safetensors"))
    emb_full = state["uce.embedding_layer.embedding.weight"]          # (145469, 5120)
    ln_weight = state["uce.embedding_layer.layer_norm.weight"]        # (5120,)
    ln_bias = state["uce.embedding_layer.layer_norm.bias"]            # (5120,)
    assert emb_full.shape == (145469, EMBEDDING_DIM), emb_full.shape
    print(f"  full embedding: {tuple(emb_full.shape)} {emb_full.dtype}")

    # ------------------------------------------------------------------
    # 2. Load the human gene dict and derive the special-token set.
    # ------------------------------------------------------------------
    print(f"loading gene dict: {args.gene_dict}")
    with open(args.gene_dict) as f:
        gene_dict_raw = json.load(f)
    human = gene_dict_raw["human"]
    n_genes = len(human)
    print(f"  human genes: {n_genes}")

    chrom_ids = sorted({int(v["chromosome_id"]) for v in human.values()})
    chrom_tokens_old = [c + CHROM_TOKEN_OFFSET for c in chrom_ids]
    print(f"  chromosome tokens (old): {len(chrom_tokens_old)} in [{chrom_tokens_old[0]}, {chrom_tokens_old[-1]}]")

    # The full set of original row indices we need, in a stable order:
    #   PAD, CLS, chrom tokens (sorted), CHROM_END, then genes (sorted by old id).
    special_old_ids: list[int] = [PAD_TOKEN_IDX, CLS_TOKEN_IDX]
    special_old_ids += chrom_tokens_old
    special_old_ids += [CHROM_TOKEN_RIGHT_IDX]

    # Genes: keep them in old-id order for determinism.
    gene_symbols_by_old: list[tuple[int, str]] = sorted(
        ((int(v["protein_embedding_id"]), sym) for sym, v in human.items()),
        key=lambda p: p[0],
    )

    old_ids_in_order: list[int] = special_old_ids + [old_id for old_id, _ in gene_symbols_by_old]
    assert len(old_ids_in_order) == len(set(old_ids_in_order)), "duplicate old ids"
    print(f"  total rows: {len(old_ids_in_order)}  "
          f"(specials={len(special_old_ids)}, genes={n_genes})")

    # ------------------------------------------------------------------
    # 3. Slice + (optionally) pre-apply LayerNorm.
    # ------------------------------------------------------------------
    old_ids_t = torch.tensor(old_ids_in_order, dtype=torch.long)
    sliced = emb_full[old_ids_t].contiguous()  # (N_rows, 5120)

    if args.skip_layer_norm:
        print("skip-layer-norm: shipping raw embedding rows")
        shipped = sliced
    else:
        print("pre-applying LayerNorm to rows")
        with torch.no_grad():
            shipped = torch.nn.functional.layer_norm(
                sliced.float(),
                normalized_shape=(EMBEDDING_DIM,),
                weight=ln_weight.float(),
                bias=ln_bias.float(),
                eps=1e-5,
            )

    shipped_np = shipped.cpu().numpy().astype(np.float32)
    print(f"  shipped table: {shipped_np.shape} {shipped_np.dtype} "
          f"-> {shipped_np.nbytes / 1e6:.1f} MB")

    # ------------------------------------------------------------------
    # 4. Build renumbered dict + special token map.
    # ------------------------------------------------------------------
    old_to_new: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(old_ids_in_order)}

    # Renumbered gene dict: original shape (chromosome_id + location remain as-is
    # for browser-side sorting), but protein_embedding_id is the new dense index.
    new_gene_dict: dict[str, dict] = {}
    for sym, v in human.items():
        new_gene_dict[sym] = {
            "protein_embedding_id": old_to_new[int(v["protein_embedding_id"])],
            "chromosome_id": int(v["chromosome_id"]),
            "location": int(v["location"]),
        }

    # Special-token remapping so the browser constructs sentences with new IDs.
    specials: dict[str, int] = {
        "pad_token_idx": old_to_new[PAD_TOKEN_IDX],
        "cls_token_idx": old_to_new[CLS_TOKEN_IDX],
        "chrom_token_right_idx": old_to_new[CHROM_TOKEN_RIGHT_IDX],
        # chrom_token_offset in the old scheme was 1000; in the new scheme the
        # browser must look up each chromosome individually because the mapping
        # isn't a fixed offset anymore. We expose it as an explicit map.
    }
    chrom_old_to_new: dict[int, int] = {
        int(chrom_id): old_to_new[chrom_id + CHROM_TOKEN_OFFSET]
        for chrom_id in chrom_ids
    }

    output_dict = {
        "species": "human",
        "num_rows": int(shipped_np.shape[0]),
        "embedding_dim": int(shipped_np.shape[1]),
        "layer_norm_applied": not args.skip_layer_norm,
        "specials": specials,
        "chromosome_token_map": chrom_old_to_new,
        "genes": new_gene_dict,
    }

    # ------------------------------------------------------------------
    # 5. Write artifacts.
    # ------------------------------------------------------------------
    bin_path = args.out_dir / "human_protein_embeddings.bin"
    json_path = args.out_dir / "human_gene_dict.json"

    shipped_np.tofile(bin_path)
    with open(json_path, "w") as f:
        json.dump(output_dict, f, separators=(",", ":"))

    print(f"wrote {bin_path} ({bin_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {json_path} ({json_path.stat().st_size / 1e3:.1f} KB)")

    # ------------------------------------------------------------------
    # 6. Round-trip validation.
    # ------------------------------------------------------------------
    print("\n=== validation ===")
    sample_genes = ["SAMD11", "NOC2L", "KLHL17", "TP53", "EGFR"]
    sample_genes = [g for g in sample_genes if g in human]
    if len(sample_genes) < 3:
        sample_genes = list(human.keys())[:5]

    ok_all = True
    for sym in sample_genes:
        old_id = int(human[sym]["protein_embedding_id"])
        new_id = new_gene_dict[sym]["protein_embedding_id"]
        old_vec = emb_full[old_id].float().cpu().numpy()
        if args.skip_layer_norm:
            expected_vec = old_vec
        else:
            with torch.no_grad():
                expected_vec = torch.nn.functional.layer_norm(
                    torch.from_numpy(old_vec).unsqueeze(0),
                    (EMBEDDING_DIM,),
                    ln_weight.float(),
                    ln_bias.float(),
                    eps=1e-5,
                ).squeeze(0).numpy()
        new_vec = shipped_np[new_id]
        maxdiff = float(np.abs(expected_vec - new_vec).max())
        ok = maxdiff == 0.0
        ok_all = ok_all and ok
        print(f"  {sym:10s}  old_id={old_id:6d} new_id={new_id:6d}  maxdiff={maxdiff:.2e}  {'OK' if ok else 'FAIL'}")

    # Special-token round-trip
    for name, old_id in [("PAD", PAD_TOKEN_IDX), ("CLS", CLS_TOKEN_IDX),
                         ("CHROM_END", CHROM_TOKEN_RIGHT_IDX),
                         ("CHROM[0]", chrom_tokens_old[0])]:
        new_id = old_to_new[old_id]
        old_vec = emb_full[old_id].float().cpu().numpy()
        if args.skip_layer_norm:
            expected_vec = old_vec
        else:
            with torch.no_grad():
                expected_vec = torch.nn.functional.layer_norm(
                    torch.from_numpy(old_vec).unsqueeze(0),
                    (EMBEDDING_DIM,),
                    ln_weight.float(),
                    ln_bias.float(),
                    eps=1e-5,
                ).squeeze(0).numpy()
        new_vec = shipped_np[new_id]
        maxdiff = float(np.abs(expected_vec - new_vec).max())
        ok = maxdiff == 0.0
        ok_all = ok_all and ok
        print(f"  {name:10s}  old_id={old_id:6d} new_id={new_id:6d}  maxdiff={maxdiff:.2e}  {'OK' if ok else 'FAIL'}")

    # Shape checks
    assert shipped_np.shape[1] == EMBEDDING_DIM
    assert shipped_np.shape[0] == 2 + len(chrom_tokens_old) + 1 + n_genes

    if not ok_all:
        print("\nFAIL: round-trip mismatch"); sys.exit(1)
    print("\nOK: all round-trip checks passed")


if __name__ == "__main__":
    main()
