"""
Render a qualitative prediction figure for the 1D Burgers experiment.

Layout  (4 time steps × 2 viscosity examples):
  Left column:  a low-ν operator  (advection-dominated, sharper gradients)
  Right column: a high-ν operator (diffusion-dominated, smoother)
  Rows: initial condition + 4 future frames, target vs prediction overlaid.
"""

from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), ".matplotlib"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import ICONLM
from train import PromptSampler, predict_full_grid_step


def _predict_trajectory(model, sampler, op, src, J, device):
    frames = []
    for step_idx in range(sampler.n_frames):
        pred = predict_full_grid_step(model, sampler, op, src, J, step_idx,
                                      device, caption_mode=None)
        frames.append(pred)
    return np.stack(frames, axis=0)   # [K, N]


def _pick_ops_by_error(model, sampler, data, device, J, n_candidates=8, rng=None):
    """Score a random sample of operators, return (low_nu, high_nu) medians."""
    if rng is None:
        rng = np.random.default_rng(2026)
    nu  = data["nu"]
    q33 = np.percentile(nu, 33)
    q67 = np.percentile(nu, 67)
    low_pool  = np.flatnonzero(nu <= q33)
    high_pool = np.flatnonzero(nu >= q67)

    def _score(pool):
        candidates = rng.choice(pool, size=min(n_candidates, len(pool)), replace=False)
        scored = []
        for op in candidates:
            pred  = _predict_trajectory(model, sampler, int(op), 0, J, device)
            true  = data["u"][op, 0]
            rel   = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8)
            scored.append((rel, int(op)))
        scored.sort()
        return scored[len(scored) // 2][1]   # median-error operator

    return _score(low_pool), _score(high_pool)


def main():
    ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device  = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)

    model = ICONLM(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    raw  = np.load("test_data.npz", allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    sampler = PromptSampler(data, ckpt["J"], ckpt["nC"], ckpt["nU"],
                            ckpt["c_scale"], ckpt["u_scale"],
                            ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
                            rng_seed=42)

    x     = data["grid"]           # [N]
    times = data["times"]          # [K]
    K     = len(times)

    op_low, op_high = _pick_ops_by_error(model, sampler, data, device, ckpt["J"])
    nu_low  = float(data["nu"][op_low])
    nu_high = float(data["nu"][op_high])

    q_src = 0
    pred_low  = _predict_trajectory(model, sampler, op_low,  q_src, ckpt["J"], device)
    pred_high = _predict_trajectory(model, sampler, op_high, q_src, ckpt["J"], device)
    true_low  = data["u"][op_low,  q_src]   # [K, N]
    true_high = data["u"][op_high, q_src]

    u0_low  = data["c"][op_low,  q_src]
    u0_high = data["c"][op_high, q_src]

    def rel_l2(pred, true):
        return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8)

    rl_low  = rel_l2(pred_low,  true_low)
    rl_high = rel_l2(pred_high, true_high)
    print(f"low-ν  op={op_low}  ν={nu_low:.5f}  rel L2={rl_low*100:.1f}%")
    print(f"high-ν op={op_high} ν={nu_high:.5f}  rel L2={rl_high*100:.1f}%")

    # ── plot ─────────────────────────────────────────────────────────────────
    n_cols = K + 1   # t=0 + K frames
    fig, axes = plt.subplots(2, n_cols, figsize=(2.8 * n_cols, 5.0),
                             sharex=True)

    def _plot_row(row_axes, u0, true, pred, nu, label):
        # t=0
        ax = row_axes[0]
        ax.plot(x, u0, color="steelblue", lw=1.5)
        ax.set_title("t = 0", fontsize=8)
        ax.set_ylabel(f"ν={nu:.4f}\n({label})", fontsize=8)
        for ki in range(K):
            ax = row_axes[ki + 1]
            step_rel = rel_l2(pred[ki], true[ki])
            ax.plot(x, true[ki], color="steelblue", lw=1.5, label="target")
            ax.plot(x, pred[ki], color="darkorange", lw=1.5,
                    linestyle="--", label="prediction")
            ax.set_title(f"t={times[ki]:.2f}\nerr={step_rel*100:.1f}%", fontsize=8)
            if ki == 0:
                ax.legend(fontsize=7, loc="upper right")

    _plot_row(axes[0], u0_low,  true_low,  pred_low,  nu_low,  "low ν")
    _plot_row(axes[1], u0_high, true_high, pred_high, nu_high, "high ν")

    for ax in axes.flat:
        ax.set_xlabel("x", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"ICON-LM · 1D Burgers  $u_t + u u_x = \\nu u_{{xx}}$  "
        f"({ckpt['J']}-shot, no caption)",
        fontsize=11, y=1.01)
    plt.tight_layout()

    per_step_low  = " ".join(
        f"{rel_l2(pred_low[k], true_low[k])*100:.1f}%" for k in range(K))
    per_step_high = " ".join(
        f"{rel_l2(pred_high[k], true_high[k])*100:.1f}%" for k in range(K))
    print(f"per-step rel L2 (low-ν):  {per_step_low}")
    print(f"per-step rel L2 (high-ν): {per_step_high}")

    fig.savefig("figure.pdf", bbox_inches="tight")
    fig.savefig("figure.png", dpi=200, bbox_inches="tight")
    print("wrote figure.pdf and figure.png")


if __name__ == "__main__":
    main()
