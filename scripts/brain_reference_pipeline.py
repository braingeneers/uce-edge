"""Phase 1: Run the reference UCE-brain pipeline on real h5ad and save per-stage fixtures.

For each test cell we save:
  - raw counts + gene symbols   (pre-stage-1)
  - normalized weights           (stage 2 output: log1p + sum-to-1)
  - sampled gene indices         (stage 3 output: which of the aligned genes got picked)
  - ordered token ids (old/new)  (stage 4 output: chromosome-ordered, with CLS/CHROM/PAD)
  - attention mask               (stage 6)
  - src embeddings (post LN)     (stage 7: embedding_layer(input_ids))
  - cell embeddings              (transformer output, L2-normalized CLS vector)

The fixture format is flat binary + JSON so the browser can load it with a
single fetch per file and compare element-wise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "UCE-brain" / "src"))

from _brain import load_brain, pick_device  # noqa: E402
from uce_brain.data.dataset import H5ADDataset, load_gene_mapping  # noqa: E402

DEFAULT_H5AD = REPO_ROOT / "data" / "allen-celltypes+human-cortex+m1-100.h5ad"
DEFAULT_GENE_DICT = REPO_ROOT / "UCE-brain" / "gene_data" / "human_gene_dict.json"
DEFAULT_RENUMBERED = REPO_ROOT / "web" / "human_gene_dict.json"
DEFAULT_OUT = REPO_ROOT / "data" / "brain_reference"

# UCE-brain sampler constants
PAD_LENGTH = 2048
SAMPLE_SIZE = 1024
CLS_TOKEN_IDX_OLD = 1
CHROM_TOKEN_OFFSET = 1000
CHROM_TOKEN_RIGHT_IDX_OLD = 2000
PAD_TOKEN_IDX_OLD = 0


def old_to_new_token_map(renumbered_json: dict) -> np.ndarray:
    """Build a dense int32 lookup from old token id -> new token id.

    Covers PAD, CLS, chromosome tokens, CHROM_END, and all gene protein IDs.
    Anything else maps to -1 (should never be hit for human inference).
    """
    specials = renumbered_json["specials"]
    chrom_map = renumbered_json["chromosome_token_map"]  # str(old_chrom_id) -> new_token_id

    # Max possible old id: 145468 (full table). Too big for a dense np array.
    # But for human inference we only need {0, 1, 1564..1613, 2000, gene_old_ids}.
    # gene_old_ids span [13466, 33255]; chromosome tokens span [1564, 1613].
    # Safe upper bound: 35000.
    max_old = 35000
    lut = np.full(max_old + 1, -1, dtype=np.int32)
    lut[PAD_TOKEN_IDX_OLD] = specials["pad_token_idx"]
    lut[CLS_TOKEN_IDX_OLD] = specials["cls_token_idx"]
    lut[CHROM_TOKEN_RIGHT_IDX_OLD] = specials["chrom_token_right_idx"]
    for old_chrom_str, new_tok in chrom_map.items():
        old_tok = int(old_chrom_str) + CHROM_TOKEN_OFFSET
        lut[old_tok] = int(new_tok)
    for _sym, g in renumbered_json["genes"].items():
        # genes dict uses new IDs, but we need old -> new. We don't store old
        # gene ids in the renumbered JSON, so we populate from the ORIGINAL
        # human_gene_dict.json outside this function.
        pass
    return lut


def build_old_to_new_lut(renumbered_json: dict, original_gene_dict_path: Path) -> np.ndarray:
    """Full lookup including genes, built by cross-referencing the renumbered
    dict's new IDs with the original dict's old IDs (keyed by gene symbol)."""
    lut = old_to_new_token_map(renumbered_json)
    with open(original_gene_dict_path) as f:
        orig = json.load(f)["human"]
    for sym, v in orig.items():
        old_id = int(v["protein_embedding_id"])
        new_id = renumbered_json["genes"][sym]["protein_embedding_id"]
        lut[old_id] = int(new_id)
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    ap.add_argument("--gene-dict", type=Path, default=DEFAULT_GENE_DICT)
    ap.add_argument("--renumbered", type=Path, default=DEFAULT_RENUMBERED,
                    help="Phase 0 output: web/human_gene_dict.json")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-cells", type=int, default=100,
                    help="Number of cells to process. src_embeddings is ~40 MB/cell at L=2048, "
                         "so for large N use --skip-src-embeddings (Phase 2/3 won't work but 4+ will).")
    ap.add_argument("--skip-src-embeddings", action="store_true",
                    help="Skip saving ref_src_embeddings.bin. Needed for n-cells > ~10 "
                         "to avoid multi-GB fixtures. Phases 2/3 will be unable to run.")
    ap.add_argument("--device", default="cpu",
                    help="Device for the reference forward. Default cpu for "
                         "bit-identity with the CPU-baked protein embedding table "
                         "(MPS LayerNorm differs at ~1e-6 from CPU).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"device: {device}")

    # ------------------------------------------------------------------
    # Load model + renumbered dict (used to translate old token ids -> new).
    # ------------------------------------------------------------------
    print("loading UCE-brain model")
    loaded = load_brain()
    loaded.core.to(device).eval()
    embedding_layer = loaded.embedding.to(device).eval()  # includes LayerNorm

    print(f"loading renumbered dict: {args.renumbered}")
    with open(args.renumbered) as f:
        renumbered = json.load(f)
    lut_old_to_new = build_old_to_new_lut(renumbered, args.gene_dict)

    # ------------------------------------------------------------------
    # Load h5ad + build Dataset.
    # ------------------------------------------------------------------
    print(f"loading h5ad: {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")

    gene_mapping = load_gene_mapping(str(args.gene_dict), species="human")

    ds = H5ADDataset(
        adata=adata,
        gene_mapping={"human": gene_mapping},
        pad_length=PAD_LENGTH,
        sample_size=SAMPLE_SIZE,
        cls_token_idx=CLS_TOKEN_IDX_OLD,
        chrom_token_offset=CHROM_TOKEN_OFFSET,
        chrom_token_right_idx=CHROM_TOKEN_RIGHT_IDX_OLD,
        pad_token_idx=PAD_TOKEN_IDX_OLD,
        mask_prop=0.0,
        max_cells=args.n_cells,
    )
    print(f"aligned genes: {len(ds.aligned_gene_names)}")

    # Save gene alignment once (same for all cells in this run).
    # gene_protein_ids_new = per-aligned-gene index into the shipped dense
    # protein embedding table (web/human_protein_embeddings.bin).
    gene_protein_ids_new = lut_old_to_new[ds.gene_protein_ids].astype(np.int32)
    assert (gene_protein_ids_new >= 0).all()

    with open(args.out_dir / "ref_gene_symbols.json", "w") as f:
        json.dump({
            "aligned_gene_names": ds.aligned_gene_names,
            "gene_protein_ids_old": ds.gene_protein_ids.tolist(),
            "gene_protein_ids_new": gene_protein_ids_new.tolist(),
            "gene_chroms": ds.gene_chroms.tolist(),
            "gene_starts": ds.gene_starts.tolist(),
            "n_cells": args.n_cells,
            "pad_length": PAD_LENGTH,
            "sample_size": SAMPLE_SIZE,
        }, f, separators=(",", ":"))

    # ------------------------------------------------------------------
    # Per-cell pipeline.
    # ------------------------------------------------------------------
    N = args.n_cells
    G = len(ds.aligned_gene_names)
    L = PAD_LENGTH
    S = SAMPLE_SIZE
    D = loaded.config.embedding_dim

    norm_weights_all = np.zeros((N, G), dtype=np.float32)
    ordered_ids_old_all = np.zeros((N, L), dtype=np.int32)
    ordered_ids_new_all = np.zeros((N, L), dtype=np.int32)
    attn_mask_all = np.zeros((N, L), dtype=np.float32)
    if not args.skip_src_embeddings:
        src_embeddings_all = np.zeros((N, L, D), dtype=np.float32)
    else:
        src_embeddings_all = None
    cell_embeddings_all = np.zeros((N, loaded.config.d_model), dtype=np.float32)

    # Per-cell valid-token count for later analysis (how much padding each cell
    # has). Drives the dynamic-seq-len optimization — real attention cost scales
    # as O(valid²), not O(pad_length²).
    valid_counts_all = np.zeros(N, dtype=np.int32)

    # Stage-3 fixtures for Phase 4+:
    #   - sample_indices: raw (sample_size,) indices into aligned gene array,
    #     in the order the sampler produced them (pre chrom-sort)
    #   - chrom_order: the post-shuffle unique chromosome order the sampler
    #     used to lay out the sentence. JS replays this deterministically
    #     without needing numpy's RNG.
    sample_indices_all = np.zeros((N, S), dtype=np.int32)
    # Max unique chromosomes per cell <= total chroms (39). Pad with -1.
    max_chroms = len(set(ds.gene_chroms.tolist()))
    chrom_order_all = -np.ones((N, max_chroms), dtype=np.int32)

    # Also save each cell's normalized counts path separately so browser can
    # reproduce stages 1-2 without re-running scanpy.
    raw_counts_all = np.zeros((N, G), dtype=np.float32)

    for i in range(N):
        # Stage 1-2: replicate Dataset.__getitem__ weight computation exactly.
        cell_expr = adata.X[i]
        if hasattr(cell_expr, "toarray"):
            cell_expr = cell_expr.toarray().flatten()
        else:
            cell_expr = np.array(cell_expr).flatten()
        valid_expr = cell_expr[ds.valid_h5ad_indices].astype(np.float32)
        raw_counts_all[i] = valid_expr

        counts_batch = torch.from_numpy(valid_expr).unsqueeze(0)
        log_expr = torch.log1p(counts_batch)
        expr_sum = torch.clamp(log_expr.sum(dim=1, keepdim=True), min=1e-8)
        weights_batch = (log_expr / expr_sum).squeeze(0).numpy()
        norm_weights_all[i] = weights_batch

        # Stage 3 (sampling) — replay independently to capture intermediates
        # the Dataset doesn't expose. Must match the sampler exactly.
        # (See UCE-brain/src/uce_brain/data/sampler.py, seed=idx branch.)
        rng_capture = np.random.default_rng(i)
        choice_idx = rng_capture.choice(
            np.arange(G), size=S, p=weights_batch, replace=True
        )
        sample_indices_all[i] = choice_idx.astype(np.int32)

        # Sampler does: uq_chroms = np.unique(new_chrom); rng.shuffle(uq_chroms)
        # where new_chrom = ds.gene_chroms[choice_idx[chrom_sort]] — but
        # np.unique on the pre-sort or post-sort is identical set-wise.
        chosen_chroms = ds.gene_chroms[choice_idx]
        uq_chroms_sorted = np.unique(chosen_chroms)  # sorted ascending
        uq_chroms_shuffled = uq_chroms_sorted.copy()
        rng_capture.shuffle(uq_chroms_shuffled)
        chrom_order_all[i, : len(uq_chroms_shuffled)] = uq_chroms_shuffled.astype(np.int32)

        # Stages 3-6 (full path): delegate to the Dataset (uses seed=idx internally).
        sample = ds[i]
        input_ids_old = sample["batch_sentences"][0].numpy().astype(np.int64)  # (L,)
        padding_mask = sample["mask"][0].numpy()  # True=pad
        attn_mask = (~padding_mask).astype(np.float32)

        # Translate old -> new token ids using the LUT.
        input_ids_new = lut_old_to_new[input_ids_old]
        assert (input_ids_new >= 0).all(), (
            f"cell {i}: some old ids have no new mapping: "
            f"{input_ids_old[input_ids_new < 0][:5]}"
        )

        ordered_ids_old_all[i] = input_ids_old.astype(np.int32)
        ordered_ids_new_all[i] = input_ids_new.astype(np.int32)
        attn_mask_all[i] = attn_mask
        valid_counts_all[i] = int(attn_mask.sum())

        # Stage 7: run embedding_layer on old ids to get (L, 5120) post-LN.
        with torch.no_grad():
            ids_t = torch.from_numpy(input_ids_old).unsqueeze(0).to(device)  # (1, L)
            src = embedding_layer(ids_t)  # (1, L, 5120)
        if src_embeddings_all is not None:
            src_np = src.squeeze(0).cpu().numpy().astype(np.float32)
            src_embeddings_all[i] = src_np

        # Transformer
        with torch.no_grad():
            mask_t = torch.from_numpy(attn_mask).unsqueeze(0).to(device)
            _, cell_emb = loaded.core(src, mask_t)
        cell_embeddings_all[i] = cell_emb.squeeze(0).cpu().numpy().astype(np.float32)

        if i == 0:
            n_valid = int(attn_mask.sum())
            n_cls = int((input_ids_old == 1).sum())
            n_pad = int((input_ids_old == 0).sum())
            print(f"cell 0: valid_positions={n_valid}, cls_count={n_cls}, pad_count={n_pad}")
            print(f"cell 0: cell_embedding norm={np.linalg.norm(cell_embeddings_all[0]):.6f}")

        # Cross-check: multiset of protein IDs from the replayed sample must
        # match the multiset embedded in the Dataset's ordered output (i.e.
        # sentence tokens above CHROM_END are exactly the genes we sampled).
        gene_tokens = input_ids_old[input_ids_old > CHROM_TOKEN_RIGHT_IDX_OLD]
        replay_gene_ids = ds.gene_protein_ids[choice_idx]
        from collections import Counter
        assert Counter(gene_tokens.tolist()) == Counter(replay_gene_ids.tolist()), (
            f"cell {i}: replayed sample differs from sampler output"
        )

    # ------------------------------------------------------------------
    # Write fixtures.
    # ------------------------------------------------------------------
    def _save_bin(name: str, arr: np.ndarray):
        path = args.out_dir / name
        arr.tofile(path)
        print(f"  {name:40s} {tuple(arr.shape)} {arr.dtype} -> {path.stat().st_size/1e6:.2f} MB")

    # Valid-token stats — drive the dynamic-seq-len optimization.
    vc = valid_counts_all
    print(f"\nvalid tokens per cell: min={vc.min()}  max={vc.max()}  "
          f"mean={vc.mean():.1f}  median={int(np.median(vc))}  "
          f"p90={int(np.percentile(vc, 90))}  "
          f"(pad_length={L}, so avg utilization {vc.mean()/L*100:.1f}%)")

    print("\nwriting fixtures")
    _save_bin("ref_raw_counts.bin", raw_counts_all)
    _save_bin("ref_normalized_weights.bin", norm_weights_all)
    _save_bin("ref_sample_indices.bin", sample_indices_all)
    _save_bin("ref_chrom_order.bin", chrom_order_all)
    _save_bin("ref_ordered_token_ids_old.bin", ordered_ids_old_all)
    _save_bin("ref_ordered_token_ids_new.bin", ordered_ids_new_all)
    _save_bin("ref_attention_mask.bin", attn_mask_all)
    _save_bin("ref_valid_counts.bin", valid_counts_all)
    if src_embeddings_all is not None:
        _save_bin("ref_src_embeddings.bin", src_embeddings_all)
    _save_bin("ref_cell_embedding.bin", cell_embeddings_all)

    # Manifest so the browser knows shapes/dtypes without guessing.
    files = {
        "ref_raw_counts.bin": {"shape": [N, G], "dtype": "float32"},
        "ref_normalized_weights.bin": {"shape": [N, G], "dtype": "float32"},
        "ref_sample_indices.bin": {"shape": [N, S], "dtype": "int32"},
        "ref_chrom_order.bin": {"shape": [N, int(max_chroms)], "dtype": "int32"},
        "ref_ordered_token_ids_old.bin": {"shape": [N, L], "dtype": "int32"},
        "ref_ordered_token_ids_new.bin": {"shape": [N, L], "dtype": "int32"},
        "ref_attention_mask.bin": {"shape": [N, L], "dtype": "float32"},
        "ref_valid_counts.bin": {"shape": [N], "dtype": "int32"},
        "ref_cell_embedding.bin": {"shape": [N, loaded.config.d_model], "dtype": "float32"},
    }
    if src_embeddings_all is not None:
        files["ref_src_embeddings.bin"] = {"shape": [N, L, D], "dtype": "float32"}
    manifest = {
        "n_cells": N,
        "n_aligned_genes": G,
        "pad_length": L,
        "sample_size": S,
        "embedding_dim": D,
        "d_model": loaded.config.d_model,
        "max_chroms": int(max_chroms),
        "valid_tokens": {
            "min": int(vc.min()),
            "max": int(vc.max()),
            "mean": float(vc.mean()),
            "median": int(np.median(vc)),
            "p90": int(np.percentile(vc, 90)),
        },
        "files": files,
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {args.out_dir / 'manifest.json'}")

    # ------------------------------------------------------------------
    # Sanity checks.
    # ------------------------------------------------------------------
    print("\n=== sanity ===")
    assert not np.isnan(cell_embeddings_all).any(), "cell embeddings contain NaN"
    norms = np.linalg.norm(cell_embeddings_all, axis=1)
    print(f"cell embedding norms: min={norms.min():.6f} max={norms.max():.6f} (expect ~1.0)")
    assert np.allclose(norms, 1.0, atol=1e-4), "cell embeddings not L2-normalized"

    # Cell embedding diversity: cosine between cell 0 and cell 1 should be <1 but correlated.
    if N >= 2:
        c0, c1 = cell_embeddings_all[0], cell_embeddings_all[1]
        cos01 = float(np.dot(c0, c1))
        print(f"cos(cell0, cell1): {cos01:.4f}")

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
