"""00_transformer_model.py — TransformerIntentPredictor.

Single source of truth for the transformer architecture (transformer/PLAN.md #3).
The Kaggle notebooks (phase2_kaggle_search/03_search_kaggle.ipynb,
phase4_kaggle_final/04_final_loso_kaggle.ipynb) embed this class verbatim under a
"KEEP IN SYNC WITH 00_transformer_model.py" banner — edit here first, then copy down.

Forward contract matches pipeline/03_bilstm_model.py::BiLSTMIntentPredictor:
    input  (B, 16, 5) float32, already z-scored by the caller
    output (B, 1) float32 raw logits (sigmoid applied by the caller)
No required constructor args (defaults = transformer_default, PLAN.md #3).
"""
import math

import torch
import torch.nn as nn


def _sinusoidal_pe(length, d_model):
    pe = torch.zeros(length, d_model)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                         * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class TransformerIntentPredictor(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=4, num_layers=2, dim_ff=None,
                 dropout=0.1, pool="cls", pos="learned", seq_len=16, norm_first=True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        if pool not in ("cls", "mean", "last"):
            raise ValueError(f"pool must be one of cls/mean/last, got {pool!r}")
        if pos not in ("learned", "sin"):
            raise ValueError(f"pos must be one of learned/sin, got {pos!r}")

        dim_ff = dim_ff or 2 * d_model
        self.pool = pool
        self.use_cls = pool == "cls"
        self.seq_len = seq_len

        self.input_proj = nn.Linear(input_dim, d_model)

        pe_len = seq_len + 1 if self.use_cls else seq_len
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if pos == "learned":
            self.pos_embed = nn.Parameter(torch.randn(1, pe_len, d_model) * 0.02)
        else:
            self.register_buffer("pos_embed", _sinusoidal_pe(pe_len, d_model), persistent=False)

        self.embed_drop = nn.Dropout(dropout)

        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=norm_first,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
        else:
            # Gate-1 linear-probe floor: skip the encoder entirely (PLAN.md Phase T1 Gate 1).
            self.encoder = None

        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        h = self.input_proj(x)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embed[:, : h.shape[1], :]
        h = self.embed_drop(h)
        if self.encoder is not None:
            h = self.encoder(h)

        if self.pool == "cls":
            pooled = h[:, 0, :]
        elif self.pool == "last":
            pooled = h[:, -1, :]
        else:  # mean — exclude the CLS token if present
            seq = h[:, 1:, :] if self.use_cls else h
            pooled = seq.mean(dim=1)

        pooled = self.head_drop(pooled)
        return self.head(pooled)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 16, 5)

    print("Shape self-test (all pool x pos combos):")
    for pool in ("cls", "mean", "last"):
        for pos in ("learned", "sin"):
            m = TransformerIntentPredictor(pool=pool, pos=pos)
            out = m(x)
            assert out.shape == (4, 1), f"bad shape {out.shape} for pool={pool} pos={pos}"
    print("  OK — every combo produces (B,1) logits")

    print("\nParameter ladder (Stage-A sizes; brackets the 594,561-param BiLSTM):")
    print(f"{'(d_model, ff)':>16} {'L':>3} {'params':>10}")
    for d_model, ff in [(64, 128), (128, 256), (128, 512)]:
        for L in (2, 4):
            n = count_params(TransformerIntentPredictor(d_model=d_model, dim_ff=ff, num_layers=L))
            print(f"{f'({d_model}, {ff})':>16} {L:>3} {n:>10,}")

    default = TransformerIntentPredictor(d_model=128, num_layers=2, dim_ff=256, dropout=0.1,
                                         pool="cls", pos="learned")
    print(f"\ntransformer_default params: {count_params(default):,}  (BiLSTM baseline: 594,561)")

    floor = TransformerIntentPredictor(num_layers=0, pool="mean")
    assert floor(x).shape == (4, 1)
    print(f"L=0 linear-probe floor params: {count_params(floor):,}")
