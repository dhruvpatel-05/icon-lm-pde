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

def build_mask(types: torch.Tensor, examples: torch.Tensor,
               caption_len: int = 0) -> torch.Tensor:
    """Build the ICON-LM attention mask, optionally with a caption prefix.

    Args:
        types:       LongTensor [L] of token types (0=C, 1=U, 2=Q).
        examples:    LongTensor [L] of example indices (0-indexed).
        caption_len: Number of caption tokens prepended to the sequence.
                     When > 0 the returned mask has shape
                     [caption_len+L, caption_len+L] and follows the paper's
                     prefix-attention rule (models_lm.py lines 54-56):
                       - caption tokens attend to each other (full),
                       - caption tokens do NOT attend to sequence tokens,
                       - all sequence tokens attend to all caption tokens.

    Returns:
        BoolTensor [caption_len+L, caption_len+L]: True where attention is
        ALLOWED, False where masked.
    """
    L = types.shape[0]
    ti = types[:, None].expand(L, L)
    tj = types[None, :].expand(L, L)
    ei = examples[:, None].expand(L, L)
    ej = examples[None, :].expand(L, L)

    same_example = (ei == ej)
    past_example = (ej < ei)

    past_ok      = past_example & ((tj == 0) | (tj == 1))
    same_C       = same_example & (ti == 0) & (tj == 0)
    same_U       = same_example & (ti == 1) & ((tj == 0) | (tj == 1))
    same_Q_to_C  = same_example & (ti == 2) & (tj == 0)
    diag         = torch.eye(L, dtype=torch.bool, device=types.device)
    same_Q_to_Q  = same_example & (ti == 2) & (tj == 2) & diag

    seq_mask = past_ok | same_C | same_U | same_Q_to_C | same_Q_to_Q

    if caption_len == 0:
        return seq_mask

    # Prefix-attention block following the paper's mask construction.
    # Full mask shape: [caption_len + L, caption_len + L]
    #   [0:cap, 0:cap] = True   caption attends to itself
    #   [0:cap, cap:]  = False  caption does NOT attend to sequence
    #   [cap:,  0:cap] = True   sequence attends to caption
    #   [cap:,  cap:]  = seq_mask
    total = caption_len + L
    full = torch.zeros(total, total, dtype=torch.bool, device=types.device)
    full[:caption_len, :caption_len] = True
    full[caption_len:, :caption_len] = True
    full[caption_len:, caption_len:] = seq_mask
    return full


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
    """ICON-LM transformer with optional caption prefix.

    When caption_vocab_size > 0 the model supports a language caption
    prepended to each prompt.  The caption is encoded as a sequence of
    character-level token IDs; each ID is embedded and given a learned
    positional encoding, then prepended to the physics token sequence.
    The attention mask (built by build_mask with caption_len > 0) gives
    caption tokens full self-attention but prevents them from attending
    to physics tokens, while every physics token attends to all caption
    tokens (prefix-attention, as in the paper).
    """

    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_heads: int = 4, d_ff: int = 128, coord_dim: int = 3,
                 embed_dim: int = 0, max_demo_num: int = 8):
        super().__init__()
        self.type_embed = nn.Embedding(3, d_model)
        self.coord_proj = nn.Linear(coord_dim, d_model)
        self.value_proj = nn.Linear(1, d_model)
        # Demo-index positional embedding: each token gets index 3*j + type_id
        # (0=C, 1=U, 2=Q), matching the paper's func_pos_embedding scheme
        # (models_utils.py:build_index_integer, models_lm.py:func_pos_embedding).
        self.demo_pos_embed = nn.Embedding(max_demo_num * 3, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        # Sentence-embedding caption: a frozen encoder produces a [1, embed_dim]
        # vector per caption; a single learned linear layer projects it to d_model.
        # caption_len is always 1 (one token per caption).
        self.embed_dim = embed_dim
        if embed_dim > 0:
            self.caption_proj = nn.Linear(embed_dim, d_model)

    def forward(self, types, coords, values, mask, demo_idx=None,
                caption_emb: "torch.Tensor | None" = None):
        """
        types:       [B, L] long
        coords:      [B, L, coord_dim] float
        values:      [B, L, 1] float
        mask:        [L, L] or [1+L, 1+L] bool
        demo_idx:    [L] long  —  3*j + type_id per token (paper's
                     func_pos_embedding index).  If None, skipped.
        caption_emb: [B, 1, embed_dim] float  —  pre-computed sentence
                     embedding from a frozen encoder, projected to d_model
                     and prepended as a single caption token.  If None,
                     no caption is used and the mask must be [L, L].
        """
        h = (self.type_embed(types)
             + self.coord_proj(coords)
             + self.value_proj(values))
        if demo_idx is not None:
            h = h + self.demo_pos_embed(demo_idx).unsqueeze(0)

        if caption_emb is not None:
            cap_h = self.caption_proj(caption_emb)   # [B, 1, d_model]
            h = torch.cat([cap_h, h], dim=1)

        for blk in self.blocks:
            h = blk(h, mask)
        h = self.ln_f(h)
        out = self.head(h).squeeze(-1)  # [B, 1+L] or [B, L]

        if caption_emb is not None:
            out = out[:, 1:]  # drop the single caption output position
        return out  # [B, L]


# ------------------------------ smoke test --------------------------------

if __name__ == "__main__":
    EMBED_DIM = 384
    J, nC, nU, nQ = 3, 24, 24, 24
    L = J * (nC + nU + nQ)
    types = torch.zeros(L, dtype=torch.long)
    examples = torch.zeros(L, dtype=torch.long)
    pos = 0
    for j in range(J):
        types[pos:pos + nC] = 0; examples[pos:pos + nC] = j; pos += nC
        types[pos:pos + nU] = 1; examples[pos:pos + nU] = j; pos += nU
        types[pos:pos + nQ] = 2; examples[pos:pos + nQ] = j; pos += nQ

    # sequence-only mask
    mask = build_mask(types, examples)
    q0 = nC + nU
    assert not mask[q0, nC:nC+nU].any().item(), "leakage!"
    q0_self = mask[q0, nC+nU:nC+nU+nQ]
    assert q0_self.sum().item() == 1 and q0_self[0].item(), "Q diagonal broken"
    print("sequence-only invariants OK")

    # caption-prefix mask (caption_len=1)
    cmask = build_mask(types, examples, caption_len=1)
    assert cmask[:1, :1].all().item(),     "cap-cap broken"
    assert not cmask[:1, 1:].any().item(), "cap->seq leak"
    assert cmask[1:, :1].all().item(),     "seq->cap broken"
    print("caption-prefix (len=1) invariants OK")

    # model smoke test
    model = ICONLM(d_model=64, n_layers=2, n_heads=4, d_ff=128,
                   coord_dim=2, embed_dim=EMBED_DIM, max_demo_num=J+1)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")

    B = 2
    coords = torch.rand(B, L, 2)
    values = torch.zeros(B, L, 1)

    out = model(types[None].expand(B, L), coords, values, mask)
    assert out.shape == (B, L), out.shape
    print("no-caption output:", out.shape)

    cap_emb = torch.randn(B, 1, EMBED_DIM)
    out_cap = model(types[None].expand(B, L), coords, values, cmask,
                    caption_emb=cap_emb)
    assert out_cap.shape == (B, L), out_cap.shape
    print("caption output:", out_cap.shape)
    print("all smoke tests passed")
