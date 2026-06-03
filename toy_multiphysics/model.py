"""
Tiny ICON-LM-style transformer for in-context 2D operator learning.

Sequence layout (J examples + optional final query "example"):
    [C_1, U_1, Q_1, C_2, U_2, Q_2, ..., C_J, U_J, Q_J]
where each block C_j, U_j, Q_j contains n_cond / n_qoi / n_query tokens.

A token is t = (type_id in {0=C, 1=U, 2=Q}, x, y, time, value). For the
multi-physics toy, Q tokens carry the query input value q(0,x,y), and the
target is the future value q(t,x,y) at the same location.

The attention mask implements the ICON-LM invariants
(Yang, Liu, Osher 2024, Fig. 3) adapted to the no-caption setting:

  - C tokens of example j see: all C/U of earlier examples + own C.
  - U tokens of example j see: all C/U of earlier examples + own C/U.
  - Q tokens of example j see: all C/U of earlier examples + own C, plus
                                themselves only inside the own-Q block (diagonal).
  - Q tokens are never attended to from later examples (causal in J).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------ mask --------------------------------------

def build_mask(types: torch.Tensor, examples: torch.Tensor) -> torch.Tensor:
    """Build the ICON-LM attention mask for a single sequence.

    Args:
        types:    LongTensor [L] of token types (0=C, 1=U, 2=Q).
        examples: LongTensor [L] of example indices (0-indexed).

    Returns:
        BoolTensor [L, L]: True where attention is ALLOWED, False where masked.
    """
    L = types.shape[0]
    ti = types[:, None].expand(L, L)            # row type
    tj = types[None, :].expand(L, L)            # col type
    ei = examples[:, None].expand(L, L)         # row example
    ej = examples[None, :].expand(L, L)         # col example

    same_example = (ei == ej)
    past_example = (ej < ei)

    # Past-example tokens visible iff they are C or U.
    past_ok = past_example & ((tj == 0) | (tj == 1))

    # Same-example rules.
    same_C = same_example & (ti == 0) & (tj == 0)          # C sees C in same ex.
    same_U = same_example & (ti == 1) & ((tj == 0) | (tj == 1))  # U sees C/U
    same_Q_to_C = same_example & (ti == 2) & (tj == 0)     # Q sees own C
    # Q sees own Q only on the diagonal.
    diag = torch.eye(L, dtype=torch.bool, device=types.device)
    same_Q_to_Q = same_example & (ti == 2) & (tj == 2) & diag

    mask = past_ok | same_C | same_U | same_Q_to_C | same_Q_to_Q
    return mask


# ------------------------------ model -------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [L, L] bool (True = attend)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                # [B, h, L, dk]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)            # [B, h, L, L]
        attn = attn.masked_fill(~mask[None, None], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                                                   # [B, h, L, dk]
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ICONLM(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 128, coord_dim: int = 3):
        super().__init__()
        self.type_embed = nn.Embedding(3, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, types, coords, values, mask):
        # types: [B, L] long; coords: [B, L, C]; values: [B, L, 1]; mask: [L, L]
        h = (self.type_embed(types)
             + self.coord_proj(coords)
             + self.value_proj(values))
        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h)
        return self.head(h).squeeze(-1)  # [B, L]


# ------------------------------ smoke test --------------------------------

if __name__ == "__main__":
    J, nC, nU, nQ = 3, 24, 24, 24
    L = J * (nC + nU + nQ)
    types = torch.zeros(L, dtype=torch.long)
    examples = torch.zeros(L, dtype=torch.long)
    pos = 0
    for j in range(J):
        types[pos:pos + nC] = 0; examples[pos:pos + nC] = j; pos += nC
        types[pos:pos + nU] = 1; examples[pos:pos + nU] = j; pos += nU
        types[pos:pos + nQ] = 2; examples[pos:pos + nQ] = j; pos += nQ

    mask = build_mask(types, examples)
    print("mask shape:", mask.shape, "fraction open:", mask.float().mean().item())

    # Sanity: Q row should never attend to its own U block.
    # First Q in example 0 is at index nC + nU = 48.
    q0 = nC + nU
    u0_start, u0_end = nC, nC + nU
    assert not mask[q0, u0_start:u0_end].any().item(), "leakage!"
    # First Q should attend only to itself in own Q block.
    q0_start = nC + nU
    q0_end = nC + nU + nQ
    q0_self = mask[q0, q0_start:q0_end]
    assert q0_self.sum().item() == 1 and q0_self[0].item(), "Q diagonal broken"
    print("invariants OK")

    model = ICONLM(d_model=64, n_layers=4, n_heads=4, d_ff=128, coord_dim=3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e3:.1f}K")

    B = 2
    coords = torch.rand(B, L, 3)
    values = torch.zeros(B, L, 1)
    out = model(types[None].expand(B, L), coords, values, mask)
    print("output:", out.shape)
