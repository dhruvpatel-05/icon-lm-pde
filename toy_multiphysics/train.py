"""
Train ICON-LM on the 1D viscous Burgers equation.

The hidden operator is the viscosity ν.  Each training prompt samples one ν
value and J demonstration trajectories from that operator, then asks the model
to predict a new trajectory (query) for the same ν in context.

Captions are pre-computed once at init using a frozen sentence encoder
(all-MiniLM-L6-v2) and stored as [O, 2, 384] arrays — no encoder calls in
the hot loop.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from data import make_dataset, make_caption, NU_MIN, NU_MAX, NUM_FRAMES, N_GRID
from model import ICONLM, build_mask


# ── config ────────────────────────────────────────────────────────────────────

N              = N_GRID   # 64 spatial points
J              = 4        # demonstration trajectories per prompt
N_COND         = 32       # sparse initial-condition samples per demo
N_QOI          = 32       # sparse future-value samples per frame per demo (50% coverage)
BATCH          = 8
STEPS          = 5000
LR             = 3e-4

D_MODEL        = 256
N_LAYERS       = 4
N_HEADS        = 8
D_FF           = 512
COORD_DIM      = 2        # (x, t)
EMBED_DIM      = 384      # all-MiniLM-L6-v2 output dimension

CAPTION_PROB   = 0.6      # fraction of batches that use a caption
PRECISE_PROB   = 0.5      # among caption batches: fraction that are precise

NUM_TRAIN_OPS  = 200
NUM_TEST_OPS   = 40
SOURCES_PER_OP = 8

SEED           = 0


# ── sentence encoder (lazy-loaded once) ───────────────────────────────────────

_ENCODER = None

def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        print("  loading sentence encoder ...")
        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
    return _ENCODER


# ── token-layout helpers ──────────────────────────────────────────────────────

def make_layout(J: int, nC: int, nU: int, nQ: int, device):
    """Return (types, examples, demo_idx, L).

    demo_idx[i] = 3*j + type_id  (paper's func_pos_embedding index).
    """
    L        = J * (nC + nU + nQ)
    types    = torch.zeros(L, dtype=torch.long, device=device)
    examples = torch.zeros(L, dtype=torch.long, device=device)
    demo_idx = torch.zeros(L, dtype=torch.long, device=device)
    pos = 0
    for j in range(J):
        types[pos:pos+nC]=0; examples[pos:pos+nC]=j; demo_idx[pos:pos+nC]=3*j+0; pos+=nC
        types[pos:pos+nU]=1; examples[pos:pos+nU]=j; demo_idx[pos:pos+nU]=3*j+1; pos+=nU
        types[pos:pos+nQ]=2; examples[pos:pos+nQ]=j; demo_idx[pos:pos+nQ]=3*j+2; pos+=nQ
    return types, examples, demo_idx, L


def _make_types_examples(block_sizes: list[tuple[int, int]], device):
    L        = sum(sz for _, sz in block_sizes)
    types    = torch.zeros(L, dtype=torch.long, device=device)
    examples = torch.zeros(L, dtype=torch.long, device=device)
    demo_idx = torch.zeros(L, dtype=torch.long, device=device)
    pos, ex  = 0, 0
    for t, sz in block_sizes:
        types[pos:pos+sz]    = t
        examples[pos:pos+sz] = ex
        demo_idx[pos:pos+sz] = 3 * ex + t
        pos += sz
        if t == 2:
            ex += 1
    return types, examples, demo_idx, L


# ── prompt sampler ────────────────────────────────────────────────────────────

class PromptSampler:
    """Sample in-context prompts from the 1D Burgers dataset."""

    def __init__(self, data: dict, J: int, nC: int, nU_per_frame: int,
                 c_scale: float, u_scale: float,
                 c_mean: float = 0.0, u_mean: float = 0.0,
                 rng_seed: int = 0):
        self.c       = data["c"]        # [O, S, N]
        self.u       = data["u"]        # [O, S, K, N]
        self.grid    = data["grid"]     # [N]
        self.times   = data["times"]    # [K]
        self.nu      = data["nu"]       # [O]
        self.J           = J
        self.nC          = nC
        self.nU_per_frame = nU_per_frame
        self.n_frames    = self.u.shape[2]
        self.nU_total    = self.n_frames * self.nU_per_frame
        self.O, self.S, self.N = self.c.shape   # 1D: no trailing spatial dim
        self.c_scale = c_scale
        self.u_scale = u_scale
        self.c_mean  = c_mean
        self.u_mean  = u_mean
        self.rng     = np.random.default_rng(rng_seed)

        # Pre-compute sentence embeddings for all operators (both modes).
        # Shape: [O, 2, EMBED_DIM]  where dim-1 is [precise, vague].
        enc = _get_encoder()
        cap_rng = np.random.default_rng(rng_seed + 1000)
        texts_p = [make_caption(float(nu), precise=True,  rng=cap_rng)
                   for nu in self.nu]
        texts_v = [make_caption(float(nu), precise=False, rng=cap_rng)
                   for nu in self.nu]
        embs_p = enc.encode(texts_p, batch_size=64, show_progress_bar=False)
        embs_v = enc.encode(texts_v, batch_size=64, show_progress_bar=False)
        self.caption_embs = np.stack(
            [embs_p.astype(np.float32), embs_v.astype(np.float32)], axis=1
        )  # [O, 2, 384]

    def _coords_at_time(self, idx: np.ndarray, t: float) -> np.ndarray:
        """Return [len(idx), 2] coordinate array (x, t)."""
        coords = np.zeros((len(idx), 2), dtype=np.float32)
        coords[:, 0] = self.grid[idx]
        coords[:, 1] = np.float32(t)
        return coords

    def sample(self, batch: int, caption_mode: "str | None" = None):
        """Sample a batch of in-context prompts.

        caption_mode: 'precise', 'vague', or None.
        Returns (coords, values, targets, target_mask, caption_emb).
        caption_emb is [B, 1, 384] float32 or None.
        """
        nC, nU, nU_total, J, N = (
            self.nC, self.nU_per_frame, self.nU_total, self.J, self.N)
        L      = J * (nC + nU_total + nU_total)
        coords      = np.zeros((batch, L, 2), dtype=np.float32)
        values      = np.zeros((batch, L, 1), dtype=np.float32)
        targets     = np.zeros((batch, L),    dtype=np.float32)
        target_mask = np.zeros((batch, L),    dtype=np.float32)
        cap_embs    = (np.zeros((batch, EMBED_DIM), dtype=np.float32)
                       if caption_mode is not None else None)

        for b in range(batch):
            op      = self.rng.integers(self.O)
            src_ids = self.rng.choice(self.S, size=J, replace=False)
            pos     = 0
            for j in range(J):
                c_field = (self.c[op, src_ids[j]] - self.c_mean) / self.c_scale
                u_traj  = (self.u[op, src_ids[j]] - self.u_mean) / self.u_scale
                # u_traj: [K, N]

                idx_q      = self.rng.choice(N, size=nU, replace=False)
                extra_pool = np.setdiff1d(np.arange(N), idx_q, assume_unique=True)
                idx_extra  = self.rng.choice(extra_pool,
                                             size=max(0, nC - nU), replace=False)
                idx_c      = np.concatenate([idx_q, idx_extra])[:nC]

                coords[b, pos:pos+nC]    = self._coords_at_time(idx_c, 0.0)
                values[b, pos:pos+nC, 0] = c_field[idx_c]
                pos += nC

                for k, t in enumerate(self.times):
                    sl = slice(pos + k*nU, pos + (k+1)*nU)
                    coords[b, sl]    = self._coords_at_time(idx_q, float(t))
                    values[b, sl, 0] = u_traj[k, idx_q]
                pos += nU_total

                for k, t in enumerate(self.times):
                    sl = slice(pos + k*nU, pos + (k+1)*nU)
                    coords[b, sl]    = self._coords_at_time(idx_q, float(t))
                    values[b, sl, 0] = c_field[idx_q]
                    targets[b, sl]   = u_traj[k, idx_q]
                    # shot_num_min=0 with caption, 1 without (paper runner_jax.py:223,229)
                    if j >= (0 if caption_mode is not None else 1):
                        target_mask[b, sl] = 1.0
                pos += nU_total

            if caption_mode is not None:
                emb_idx       = 0 if caption_mode == "precise" else 1
                cap_embs[b]   = self.caption_embs[op, emb_idx]

        cap_tensor = (torch.from_numpy(cap_embs).unsqueeze(1)  # [B, 1, 384]
                      if cap_embs is not None else None)
        return (torch.from_numpy(coords), torch.from_numpy(values),
                torch.from_numpy(targets), torch.from_numpy(target_mask),
                cap_tensor)


# ── dense-query inference (for figure) ───────────────────────────────────────

def predict_full_grid_step(model, sampler: PromptSampler, op_idx: int,
                           query_src_idx: int, J: int, step_idx: int,
                           device, caption_mode: "str | None" = None) -> np.ndarray:
    """Predict u(x, t_{step_idx}) on the full grid for a held-out trajectory."""
    nC, nU, nU_total = sampler.nC, sampler.nU_per_frame, sampler.nU_total
    N = sampler.N

    block_sizes = []
    for _ in range(J):
        block_sizes.extend([(0, nC), (1, nU_total), (2, nU_total)])
    block_sizes.extend([(0, N), (2, N)])   # full-grid C + Q blocks
    types, examples, demo_idx, L = _make_types_examples(block_sizes, device)

    coords = np.zeros((1, L, 2), dtype=np.float32)
    values = np.zeros((1, L, 1), dtype=np.float32)

    rng      = np.random.default_rng(123 + op_idx)
    src_pool = [i for i in range(sampler.S) if i != query_src_idx]
    src_ids  = rng.choice(src_pool, size=J, replace=False)
    pos      = 0

    for j in range(J):
        c_field = (sampler.c[op_idx, src_ids[j]] - sampler.c_mean) / sampler.c_scale
        u_traj  = (sampler.u[op_idx, src_ids[j]] - sampler.u_mean) / sampler.u_scale
        idx_q   = rng.choice(N, size=nU, replace=False)
        extra   = np.setdiff1d(np.arange(N), idx_q, assume_unique=True)
        idx_c   = np.concatenate([idx_q,
                                  rng.choice(extra, size=max(0, nC-nU),
                                             replace=False)])[:nC]

        coords[0, pos:pos+nC]    = sampler._coords_at_time(idx_c, 0.0)
        values[0, pos:pos+nC, 0] = c_field[idx_c]
        pos += nC

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k*nU, pos + (k+1)*nU)
            coords[0, sl]    = sampler._coords_at_time(idx_q, float(t))
            values[0, sl, 0] = u_traj[k, idx_q]
        pos += nU_total

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k*nU, pos + (k+1)*nU)
            coords[0, sl]    = sampler._coords_at_time(idx_q, float(t))
            values[0, sl, 0] = c_field[idx_q]
        pos += nU_total

    # Full-grid query: all N spatial points.
    c_query = (sampler.c[op_idx, query_src_idx] - sampler.c_mean) / sampler.c_scale
    coords[0, pos:pos+N]    = sampler._coords_at_time(np.arange(N), 0.0)
    values[0, pos:pos+N, 0] = c_query
    pos += N

    t = float(sampler.times[step_idx])
    coords[0, pos:pos+N]    = sampler._coords_at_time(np.arange(N), t)
    values[0, pos:pos+N, 0] = c_query

    cap_emb = None
    if caption_mode is not None:
        emb_idx = 0 if caption_mode == "precise" else 1
        cap_np  = sampler.caption_embs[op_idx, emb_idx][None, None, :]  # [1,1,384]
        cap_emb = torch.from_numpy(cap_np).to(device)

    cap_len = 1 if cap_emb is not None else 0
    mask    = build_mask(types, examples, caption_len=cap_len)
    model.eval()
    with torch.no_grad():
        pred = model(types[None].to(device),
                     torch.from_numpy(coords).to(device),
                     torch.from_numpy(values).to(device),
                     mask, demo_idx=demo_idx, caption_emb=cap_emb)
    return pred[0, -N:].cpu().numpy() * sampler.u_scale + sampler.u_mean


# ── training driver ───────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device  = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)
    if device == "cuda":
        print("cuda device:", torch.cuda.get_device_name(0))

    # ── data ──────────────────────────────────────────────────────────────────
    print("generating data ...")
    t0 = time.time()
    nu_train = np.exp(np.linspace(np.log(NU_MIN), np.log(NU_MAX), NUM_TRAIN_OPS))
    nu_test  = np.exp(np.random.default_rng(SEED + 9999).uniform(
                   np.log(NU_MIN), np.log(NU_MAX), NUM_TEST_OPS))
    train_data = make_dataset(nu_train, SOURCES_PER_OP, N, seed=SEED,
                              num_frames=NUM_FRAMES)
    test_data  = make_dataset(nu_test,  SOURCES_PER_OP, N, seed=SEED + 9999,
                              num_frames=NUM_FRAMES)
    print(f"  train ops {NUM_TRAIN_OPS}, test ops {NUM_TEST_OPS}, "
          f"frames {NUM_FRAMES}, took {time.time()-t0:.1f}s")

    c_mean  = float(train_data["c"].mean())
    u_mean  = float(train_data["u"].mean())
    c_scale = float(train_data["c"].std())  + 1e-6
    u_scale = float(train_data["u"].std())  + 1e-6
    print(f"normalize: c_mean={c_mean:.4f}, c_scale={c_scale:.4f}, "
          f"u_mean={u_mean:.4f}, u_scale={u_scale:.4f}")

    # ── samplers (pre-compute sentence embeddings inside __init__) ─────────────
    print("pre-computing sentence embeddings ...")
    sampler      = PromptSampler(train_data, J, N_COND, N_QOI,
                                 c_scale, u_scale, c_mean, u_mean, rng_seed=1)
    test_sampler = PromptSampler(test_data,  J, N_COND, N_QOI,
                                 c_scale, u_scale, c_mean, u_mean, rng_seed=2)

    # ── layout + masks ─────────────────────────────────────────────────────────
    nU_total = sampler.nU_total
    types, examples, demo_idx, L = make_layout(J, N_COND, nU_total, nU_total, device)
    types_b       = types[None].expand(BATCH, L)
    mask_no_cap   = build_mask(types, examples, caption_len=0)
    mask_with_cap = build_mask(types, examples, caption_len=1)  # always 1 token

    # ── model ──────────────────────────────────────────────────────────────────
    model = ICONLM(D_MODEL, N_LAYERS, N_HEADS, D_FF,
                   coord_dim=COORD_DIM, embed_dim=EMBED_DIM,
                   max_demo_num=J + 1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e3:.1f}K  seq_len: {L}")

    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                        eta_min=LR * 0.1)
    cap_rng = np.random.default_rng(42)

    # ── training loop ──────────────────────────────────────────────────────────
    print("training ...")
    t0, losses = time.time(), []
    for step in range(STEPS):
        r        = float(cap_rng.random())
        cap_mode = (("precise" if cap_rng.random() < PRECISE_PROB else "vague")
                    if r < CAPTION_PROB else None)

        coords, values, targets, target_mask, cap_emb = sampler.sample(
            BATCH, caption_mode=cap_mode)
        coords      = coords.to(device)
        values      = values.to(device)
        targets     = targets.to(device)
        target_mask = target_mask.to(device)
        mask        = mask_with_cap if cap_emb is not None else mask_no_cap
        cap_emb     = cap_emb.to(device) if cap_emb is not None else None

        pred = model(types_b, coords, values, mask,
                     demo_idx=demo_idx, caption_emb=cap_emb)
        sq   = (pred - targets) ** 2 * target_mask
        loss = sq.sum() / target_mask.sum().clamp_min(1.0)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))

        if (step + 1) % 200 == 0 or step == 0:
            recent = sum(losses[-200:]) / max(1, len(losses[-200:]))
            print(f"  step {step+1:5d}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg200={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"cap={cap_mode or 'none'}")

    train_seconds = time.time() - t0
    print(f"training took {train_seconds:.1f}s")

    # ── quick test eval (no caption baseline) ─────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_losses, rel_num, rel_den = [], 0.0, 0.0
        for _ in range(20):
            coords, values, targets, target_mask, _ = test_sampler.sample(
                BATCH, caption_mode=None)
            coords      = coords.to(device)
            values      = values.to(device)
            targets     = targets.to(device)
            target_mask = target_mask.to(device)
            pred   = model(types_b, coords, values, mask_no_cap, demo_idx=demo_idx)
            sq     = (pred - targets) ** 2 * target_mask
            test_losses.append(float(
                (sq.sum() / target_mask.sum().clamp_min(1.0)).item()))
            pu = pred    * u_scale + u_mean
            tu = targets * u_scale + u_mean
            rel_num += float(((pu - tu)**2 * target_mask).sum())
            rel_den += float((tu**2       * target_mask).sum())

    rel = (rel_num / max(rel_den, 1e-8)) ** 0.5
    print(f"test MSE (scaled, no-cap): {sum(test_losses)/len(test_losses):.4f}")
    print(f"test relative L2 (no-cap): {rel*100:.2f}%")

    # ── save ──────────────────────────────────────────────────────────────────
    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       d_ff=D_FF, coord_dim=COORD_DIM,
                       embed_dim=EMBED_DIM, max_demo_num=J + 1),
        "c_scale": c_scale, "u_scale": u_scale,
        "c_mean":  c_mean,  "u_mean":  u_mean,
        "N": N, "J": J, "nC": N_COND, "nU": N_QOI,
        "n_frames": NUM_FRAMES, "times": train_data["times"],
    }, "checkpoint.pt")

    np.savez("test_data.npz",
             c=test_data["c"], u=test_data["u"],
             grid=test_data["grid"], times=test_data["times"],
             nu=test_data["nu"])

    with open("metrics.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated by toy_multiphysics/train.py\n")
        f.write(f"\\newcommand{{\\ToyGrid}}{{{N}}}\n")
        f.write(f"\\newcommand{{\\ToyFrames}}{{{NUM_FRAMES}}}\n")
        f.write(f"\\newcommand{{\\ToyShots}}{{{J}}}\n")
        f.write(f"\\newcommand{{\\ToyTrainDevice}}{{{device.upper()}}}\n")
        f.write(f"\\newcommand{{\\ToyTrainTime}}{{{train_seconds/60:.1f} min}}\n")
        f.write(f"\\newcommand{{\\ToyTestRel}}{{{rel*100:.1f}\\%}}\n")

    print("saved checkpoint.pt, test_data.npz, metrics.tex")


if __name__ == "__main__":
    main()
