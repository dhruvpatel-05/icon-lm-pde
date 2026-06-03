"""
Train a tiny ICON-LM on a toy multi-regime 2D fluid-tracer family.

Each prompt samples one hidden operator and J distinct trajectories from that
operator.  A trajectory is

    q(0, x, y) -> [q(t_1, x, y), ..., q(t_K, x, y)].

The transformer receives sparse demonstration tokens plus sparse query tokens.
It must infer the hidden regime/operator from the demos and forecast the query
trajectory in context, with no weight updates at test time.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from data import make_dataset
from model import ICONLM, build_mask


# ------------------------------ config ------------------------------------

N = 64                 # grid resolution
J = 4                  # demonstration trajectories per prompt
N_COND = 128           # sampled initial-tracer points per trajectory
N_QOI = 32             # sampled spatial points per future frame
BATCH = 8
STEPS = 1000
LR = 3e-4

D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 128
COORD_DIM = 3          # x, y, time

NUM_TRAIN_OPS = 240
NUM_TEST_OPS = 48
SOURCES_PER_OP = 10
NUM_FRAMES = 4
STEPS_PER_FRAME = 8
DT = 0.02

SEED = 0


# --------------------- token-layout helpers --------------------------------

def make_layout(J: int, nC: int, nU: int, nQ: int, device):
    """Return (types, examples) [L] describing the prompt structure."""
    L = J * (nC + nU + nQ)
    types = torch.zeros(L, dtype=torch.long, device=device)
    examples = torch.zeros(L, dtype=torch.long, device=device)
    pos = 0
    for j in range(J):
        types[pos:pos + nC] = 0
        examples[pos:pos + nC] = j
        pos += nC
        types[pos:pos + nU] = 1
        examples[pos:pos + nU] = j
        pos += nU
        types[pos:pos + nQ] = 2
        examples[pos:pos + nQ] = j
        pos += nQ
    return types, examples, L


# --------------------- batch sampler ---------------------------------------

class PromptSampler:
    """Sample in-context prompts from the swirly tracer trajectory dataset."""

    def __init__(self, data: dict, J: int, nC: int, nU_per_frame: int,
                 c_scale: float, u_scale: float, c_mean: float = 0.0,
                 u_mean: float = 0.0, rng_seed: int = 0):
        self.c = data["c"]                  # [O, S, N, N]
        self.u = data["u"]                  # [O, S, K, N, N]
        self.grid = data["grid"]            # [N]
        self.times = data["times"]          # [K]
        self.J = J
        self.nC = nC
        self.nU_per_frame = nU_per_frame
        self.n_frames = self.u.shape[2]
        self.nU_total = self.n_frames * self.nU_per_frame
        self.O, self.S, self.N, _ = self.c.shape
        self.c_scale = c_scale
        self.u_scale = u_scale
        self.c_mean = c_mean
        self.u_mean = u_mean
        self.rng = np.random.default_rng(rng_seed)

        gx, gy = np.meshgrid(self.grid, self.grid, indexing="ij")
        self.flat_xy = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)

    def _coords_at_time(self, idx: np.ndarray, t: float) -> np.ndarray:
        coords = np.zeros((len(idx), 3), dtype=np.float32)
        coords[:, :2] = self.flat_xy[idx]
        coords[:, 2] = np.float32(t)
        return coords

    def sample(self, batch: int):
        nC = self.nC
        nU = self.nU_per_frame
        nU_total = self.nU_total
        J = self.J
        N2 = self.N * self.N
        L = J * (nC + nU_total + nU_total)
        coords = np.zeros((batch, L, 3), dtype=np.float32)
        values = np.zeros((batch, L, 1), dtype=np.float32)
        targets = np.zeros((batch, L), dtype=np.float32)
        target_mask = np.zeros((batch, L), dtype=np.float32)

        for b in range(batch):
            op = self.rng.integers(self.O)
            src_ids = self.rng.choice(self.S, size=J, replace=False)
            pos = 0
            for j in range(J):
                c_field = ((self.c[op, src_ids[j]].reshape(-1) - self.c_mean)
                           / self.c_scale)
                u_traj = ((self.u[op, src_ids[j]].reshape(self.n_frames, -1)
                           - self.u_mean) / self.u_scale)

                idx_q = self.rng.choice(N2, size=nU, replace=False)
                extra_pool = np.setdiff1d(np.arange(N2), idx_q, assume_unique=True)
                n_extra = max(0, nC - nU)
                idx_extra = self.rng.choice(extra_pool, size=n_extra, replace=False)
                idx_c = np.concatenate([idx_q, idx_extra])[:nC]

                # C tokens: initial tracer field at t=0.
                coords[b, pos:pos + nC] = self._coords_at_time(idx_c, 0.0)
                values[b, pos:pos + nC, 0] = c_field[idx_c]
                pos += nC

                # U tokens: known future demo values at several future times.
                for k, t in enumerate(self.times):
                    sl = slice(pos + k * nU, pos + (k + 1) * nU)
                    coords[b, sl] = self._coords_at_time(idx_q, float(t))
                    values[b, sl, 0] = u_traj[k, idx_q]
                pos += nU_total

                # Q tokens: same future locations/times, carrying q(0) as
                # the query input value.  The target is the future tracer value.
                for k, t in enumerate(self.times):
                    sl = slice(pos + k * nU, pos + (k + 1) * nU)
                    coords[b, sl] = self._coords_at_time(idx_q, float(t))
                    values[b, sl, 0] = c_field[idx_q]
                    targets[b, sl] = u_traj[k, idx_q]
                    if j >= 1:
                        target_mask[b, sl] = 1.0
                pos += nU_total

        return (
            torch.from_numpy(coords),
            torch.from_numpy(values),
            torch.from_numpy(targets),
            torch.from_numpy(target_mask),
        )


# --------------------- dense-query inference for figure --------------------

def _make_types_examples(block_sizes: list[tuple[int, int]], device):
    L = sum(sz for _, sz in block_sizes)
    types = torch.zeros(L, dtype=torch.long, device=device)
    examples = torch.zeros(L, dtype=torch.long, device=device)
    pos = 0
    ex = 0
    for t, sz in block_sizes:
        types[pos:pos + sz] = t
        examples[pos:pos + sz] = ex
        pos += sz
        if t == 2:
            ex += 1
    return types, examples, L


def predict_full_grid_step(model, sampler: PromptSampler, op_idx: int,
                           query_src_idx: int, J: int, step_idx: int,
                           device) -> np.ndarray:
    """Predict one full future frame for a held-out query trajectory."""
    nC = sampler.nC
    nU = sampler.nU_per_frame
    nU_total = sampler.nU_total
    N = sampler.N
    N2 = N * N

    block_sizes = []
    for _ in range(J):
        block_sizes.extend([(0, nC), (1, nU_total), (2, nU_total)])
    block_sizes.extend([(0, N2), (2, N2)])
    types, examples, L = _make_types_examples(block_sizes, device)

    coords = np.zeros((1, L, 3), dtype=np.float32)
    values = np.zeros((1, L, 1), dtype=np.float32)

    rng = np.random.default_rng(123 + op_idx)
    src_pool = [i for i in range(sampler.S) if i != query_src_idx]
    src_ids = rng.choice(src_pool, size=J, replace=False)
    pos = 0

    for j in range(J):
        c_field = ((sampler.c[op_idx, src_ids[j]].reshape(-1) - sampler.c_mean)
                   / sampler.c_scale)
        u_traj = ((sampler.u[op_idx, src_ids[j]].reshape(sampler.n_frames, -1)
                   - sampler.u_mean) / sampler.u_scale)
        idx_q = rng.choice(N2, size=nU, replace=False)
        extra_pool = np.setdiff1d(np.arange(N2), idx_q, assume_unique=True)
        idx_extra = rng.choice(extra_pool, size=max(0, nC - nU), replace=False)
        idx_c = np.concatenate([idx_q, idx_extra])[:nC]

        coords[0, pos:pos + nC] = sampler._coords_at_time(idx_c, 0.0)
        values[0, pos:pos + nC, 0] = c_field[idx_c]
        pos += nC

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k * nU, pos + (k + 1) * nU)
            coords[0, sl] = sampler._coords_at_time(idx_q, float(t))
            values[0, sl, 0] = u_traj[k, idx_q]
        pos += nU_total

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k * nU, pos + (k + 1) * nU)
            coords[0, sl] = sampler._coords_at_time(idx_q, float(t))
            values[0, sl, 0] = c_field[idx_q]
        pos += nU_total

    # Query trajectory: full initial field and a dense Q block for one time.
    c_query = ((sampler.c[op_idx, query_src_idx].reshape(-1) - sampler.c_mean)
               / sampler.c_scale)
    coords[0, pos:pos + N2] = sampler._coords_at_time(np.arange(N2), 0.0)
    values[0, pos:pos + N2, 0] = c_query
    pos += N2

    t = float(sampler.times[step_idx])
    coords[0, pos:pos + N2] = sampler._coords_at_time(np.arange(N2), t)
    values[0, pos:pos + N2, 0] = c_query
    pos += N2

    mask = build_mask(types, examples)
    model.eval()
    with torch.no_grad():
        pred = model(types[None].to(device), torch.from_numpy(coords).to(device),
                     torch.from_numpy(values).to(device), mask)
    pred = pred[0, -N2:].cpu().numpy()
    return pred.reshape(N, N) * sampler.u_scale + sampler.u_mean


# --------------------- driver ---------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)
    if device == "cuda":
        print("cuda device:", torch.cuda.get_device_name(0))

    print("generating data ...")
    t0 = time.time()
    train_data = make_dataset(NUM_TRAIN_OPS, SOURCES_PER_OP, N, seed=SEED,
                              num_frames=NUM_FRAMES,
                              steps_per_frame=STEPS_PER_FRAME, dt=DT)
    test_data = make_dataset(NUM_TEST_OPS, SOURCES_PER_OP, N, seed=SEED + 9999,
                             num_frames=NUM_FRAMES,
                             steps_per_frame=STEPS_PER_FRAME, dt=DT)
    print(f"  train ops {NUM_TRAIN_OPS}, test ops {NUM_TEST_OPS}, "
          f"frames {NUM_FRAMES}, took {time.time() - t0:.1f}s")

    c_mean = float(train_data["c"].mean())
    u_mean = float(train_data["u"].mean())
    c_scale = float(train_data["c"].std()) + 1e-6
    u_scale = float(train_data["u"].std()) + 1e-6
    print(f"normalize: c_mean={c_mean:.4f}, c_scale={c_scale:.4f}, "
          f"u_mean={u_mean:.4f}, u_scale={u_scale:.4f}")

    sampler = PromptSampler(train_data, J, N_COND, N_QOI, c_scale, u_scale,
                            c_mean, u_mean, rng_seed=1)
    test_sampler = PromptSampler(test_data, J, N_COND, N_QOI, c_scale, u_scale,
                                 c_mean, u_mean,
                                 rng_seed=2)

    nU_total = sampler.nU_total
    types, examples, L = make_layout(J, N_COND, nU_total, nU_total, device)
    mask = build_mask(types, examples)
    types_b = types[None].expand(BATCH, L)

    model = ICONLM(D_MODEL, N_LAYERS, N_HEADS, D_FF, coord_dim=COORD_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e3:.1f}K, seq len: {L}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                       eta_min=LR * 0.1)

    print("training ...")
    t0 = time.time()
    losses = []
    for step in range(STEPS):
        coords, values, targets, target_mask = sampler.sample(BATCH)
        coords = coords.to(device)
        values = values.to(device)
        targets = targets.to(device)
        target_mask = target_mask.to(device)

        pred = model(types_b, coords, values, mask)
        sq = (pred - targets) ** 2 * target_mask
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
                  f"avg200={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}")

    train_seconds = time.time() - t0
    print(f"training took {train_seconds:.1f}s")

    model.eval()
    with torch.no_grad():
        test_losses = []
        rel_num = torch.tensor(0.0, device=device)
        rel_den = torch.tensor(0.0, device=device)
        for _ in range(20):
            coords, values, targets, target_mask = test_sampler.sample(BATCH)
            coords = coords.to(device)
            values = values.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)
            pred = model(types_b, coords, values, mask)
            sq = (pred - targets) ** 2 * target_mask
            test_losses.append(float((sq.sum() / target_mask.sum().clamp_min(1.0)).item()))

            target_unscaled = targets * u_scale + u_mean
            pred_unscaled = pred * u_scale + u_mean
            rel_num += ((pred_unscaled - target_unscaled) ** 2 * target_mask).sum()
            rel_den += (target_unscaled ** 2 * target_mask).sum()

        rel = float(torch.sqrt(rel_num / rel_den.clamp_min(1e-8)).item())

    print(f"test MSE (scaled): {sum(test_losses)/len(test_losses):.4f}")
    print(f"test relative L2 error: {rel*100:.2f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "config": dict(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                       d_ff=D_FF, coord_dim=COORD_DIM),
        "c_scale": c_scale,
        "u_scale": u_scale,
        "c_mean": c_mean,
        "u_mean": u_mean,
        "N": N,
        "J": J,
        "nC": N_COND,
        "nU": N_QOI,
        "n_frames": NUM_FRAMES,
        "times": train_data["times"],
    }, "checkpoint.pt")

    test_npz = dict(c=test_data["c"], u=test_data["u"], grid=test_data["grid"],
                    times=test_data["times"], regime=test_data["regime"],
                    nu=test_data["nu"], speed=test_data["speed"],
                    regime_names=test_data["regime_names"])
    np.savez("test_data.npz", **test_npz)

    with open("metrics.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated by toy_multiphysics/train.py\n")
        f.write(f"\\newcommand{{\\ToyGrid}}{{${N}\\times {N}$}}\n")
        f.write(f"\\newcommand{{\\ToyFrames}}{{{NUM_FRAMES}}}\n")
        f.write(f"\\newcommand{{\\ToyShots}}{{{J}}}\n")
        f.write(f"\\newcommand{{\\ToyTrainDevice}}{{{device.upper()}}}\n")
        f.write(f"\\newcommand{{\\ToyTrainTime}}{{{train_seconds / 60.0:.1f} min}}\n")
        f.write(f"\\newcommand{{\\ToyTestRel}}{{{rel * 100.0:.1f}\\%}}\n")

    print("saved checkpoint.pt and test_data.npz")
    print("wrote metrics.tex")


if __name__ == "__main__":
    main()
