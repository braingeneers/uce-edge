"""Shared helpers for the UCE-brain spike.

Loads the inner UCEModel (no expression decoder), strips the input embedding
out of the exported graph (matching how we treat the original UCE), and
provides synthetic inputs.

The exported forward consumes:
  src:  (batch, seq_len, embedding_dim=5120) float32  — already-embedded tokens
  mask: (batch, seq_len) float32                       — 1 valid, 0 pad

This mirrors the convention used in scripts/_core.py for the original UCE.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
BRAIN_DIR = REPO_ROOT / "UCE-brain" / "src"
if str(BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_DIR))

from uce_brain.model.modeling import UCEModel  # noqa: E402
from uce_brain.model.config import UCEConfig  # noqa: E402

CKPT_DIR = REPO_ROOT / "model_files" / "uce-brain-pilot-8l-512d"
ARTIFACT_DIR = REPO_ROOT / "data" / "brain_artifacts"

VOCAB_SIZE = 145469
EMBEDDING_DIM = 5120
DEFAULT_SEQ_LEN = 1024
DEFAULT_BATCH = 2


class BrainCore(nn.Module):
    """Wraps UCEModel so forward() takes pre-embedded (B, L, 5120) input.

    The protein embedding lookup is moved out of the graph (caller-side).
    Everything from input_gene_embedding_projector onward is exported.
    """

    def __init__(self, uce: UCEModel):
        super().__init__()
        self.projector = uce.input_gene_embedding_projector
        self.pos_encoder = uce.pos_encoder
        self.transformer_encoder = uce.transformer_encoder
        # Disable the nested-tensor fast path: it uses MPS-unsupported ops
        # and produces a graph that doesn't trace cleanly to ONNX.
        self.transformer_encoder.enable_nested_tensor = False
        self.transformer_encoder.use_nested_tensor = False
        self.transformer_encoder.mask_check = False
        self.output_projector = uce.output_embedding_projector
        # We deliberately inline the CLS-select + normalize rather than reuse
        # EmbeddingAggregator to keep the graph as flat as possible.

    def forward(self, src: torch.Tensor, mask: torch.Tensor):
        """
        src: (batch, seq_len, embedding_dim) float32
        mask: (batch, seq_len) float32 — 1 valid, 0 pad
        returns: (gene_embeddings, cell_embedding)
          gene_embeddings: (batch, seq_len, output_dim)
          cell_embedding:  (batch, output_dim) L2-normalized
        """
        x = self.projector(src)
        x = self.pos_encoder(x)
        key_padding_mask = (mask == 0)
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.output_projector(x)
        cell = x[:, 0, :]
        cell = nn.functional.normalize(cell, dim=-1)
        return x, cell


@dataclass
class LoadedBrain:
    core: BrainCore
    embedding: nn.Embedding
    config: UCEConfig


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_brain(ckpt_dir: Path = CKPT_DIR) -> LoadedBrain:
    """Load just the inner UCEModel from a saved UCEForExpressionPrediction.

    The HF checkpoint has all weights prefixed with `uce.`. We construct a
    bare UCEModel and load only those keys.
    """
    from safetensors.torch import load_file

    config = UCEConfig.from_pretrained(str(ckpt_dir))
    config.dropout = 0.0  # eval; matters for pos_encoder Dropout
    uce = UCEModel(config)
    uce.eval()

    full_state = load_file(str(ckpt_dir / "model.safetensors"))
    inner = {}
    for k, v in full_state.items():
        if k.startswith("uce."):
            inner[k[len("uce."):]] = v
    missing, unexpected = uce.load_state_dict(inner, strict=False)
    # `expression_decoder.*` keys are intentionally excluded; nothing in `uce`
    # should be missing.
    assert not missing, f"missing UCE keys: {missing[:5]}..."

    embedding = uce.embedding_layer  # keeps the LayerNorm too
    core = BrainCore(uce).eval()
    return LoadedBrain(core=core, embedding=embedding, config=config)


def synthetic_inputs(
    seq_len: int = DEFAULT_SEQ_LEN,
    batch: int = DEFAULT_BATCH,
    embedding: nn.Module | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    if embedding is None:
        src = torch.randn(batch, seq_len, EMBEDDING_DIM, generator=g)
    else:
        ids = torch.randint(1, VOCAB_SIZE, (batch, seq_len), generator=g)
        with torch.no_grad():
            src = embedding(ids)
    mask = torch.ones(batch, seq_len, dtype=torch.float32)
    return src, mask


def time_forward(model, src, mask, device, warmup=2, iters=5):
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
            ge, ce = model(src, mask)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) / iters
    return ge.detach().cpu(), ce.detach().cpu(), elapsed


def ensure_artifact_dir() -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR
