"""Shared helpers for the core-model harness.

Loads UCE's TransformerModel standalone (no scRNA/protein-embedding pipeline)
and builds synthetic inputs for the transformer forward path. The model's
forward() consumes already-embedded tokens of shape (seq_len, batch, token_dim),
so we deliberately keep pe_embedding out of the exported graph and do the
lookup in Python on the caller side.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
UCE_DIR = REPO_ROOT / "UCE"
if str(UCE_DIR) not in sys.path:
    sys.path.insert(0, str(UCE_DIR))

from model import TransformerModel  # noqa: E402


VOCAB_SIZE = 145469
TOKEN_DIM = 5120
D_MODEL = 1280
NHEAD = 20
D_HID = 5120
OUTPUT_DIM = 1280
DEFAULT_SEQ_LEN = 1536
DEFAULT_BATCH = 2

CKPT_4L = REPO_ROOT / "model_files" / "4layer_model.torch"
CKPT_33L = REPO_ROOT / "model_files" / "33l_8ep_1024t_1280.torch"

ARTIFACT_DIR = REPO_ROOT / "data" / "core_artifacts"


@dataclass
class LoadedModel:
    model: TransformerModel
    pe_embedding: nn.Embedding
    nlayers: int


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_core_model(ckpt_path: Path, nlayers: int) -> LoadedModel:
    model = TransformerModel(
        token_dim=TOKEN_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=nlayers,
        output_dim=OUTPUT_DIM,
        dropout=0.05,
    )
    empty_pe = torch.zeros(VOCAB_SIZE, TOKEN_DIM)
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    pe = model.pe_embedding
    model.pe_embedding = None  # keep it out of the exported graph
    return LoadedModel(model=model, pe_embedding=pe, nlayers=nlayers)


def synthetic_inputs(
    seq_len: int = DEFAULT_SEQ_LEN,
    batch: int = DEFAULT_BATCH,
    pe_embedding: nn.Embedding | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (src, mask).

    src:  (seq_len, batch, TOKEN_DIM) float32 — already embedded
    mask: (batch, seq_len) float32 — 1 valid, 0 pad
    """
    g = torch.Generator().manual_seed(seed)
    if pe_embedding is None:
        src = torch.randn(seq_len, batch, TOKEN_DIM, generator=g)
    else:
        token_ids = torch.randint(0, VOCAB_SIZE, (seq_len, batch), generator=g)
        with torch.no_grad():
            src = pe_embedding(token_ids)
    mask = torch.ones(batch, seq_len, dtype=torch.float32)
    return src, mask


def time_forward(
    model: TransformerModel,
    src: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    warmup: int = 2,
    iters: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model = model.to(device)
    src = src.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(src, mask)
        if device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            gene_output, embedding = model(src, mask)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) / iters

    return gene_output.detach().cpu(), embedding.detach().cpu(), elapsed


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR
