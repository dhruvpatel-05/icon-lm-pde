"""
Render a qualitative multi-step fluid-tracer forecast figure.

For one held-out hidden stirring operator, predict four future tracer frames
on the full grid:

    row 0: target q(t)
    row 1: ICON-LM prediction
    row 2: signed prediction error
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__),
                                                   ".matplotlib"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import ICONLM
from train import PromptSampler, predict_full_grid_step


def _as_str_list(arr) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in arr]


def _predict_trajectory(model, sampler: PromptSampler, op: int, src: int,
                        J: int, device) -> np.ndarray:
    frames = []
    for step_idx in range(sampler.n_frames):
        pred = predict_full_grid_step(model, sampler, op, src, J, step_idx, device)
        frames.append(pred)
    return np.stack(frames, axis=0)


def main():
    ckpt = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    c_scale = ckpt["c_scale"]
    u_scale = ckpt["u_scale"]
    c_mean = ckpt.get("c_mean", 0.0)
    u_mean = ckpt.get("u_mean", 0.0)
    J = ckpt["J"]
    nC = ckpt["nC"]
    nU = ckpt["nU"]

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)
    if device == "cuda":
        print("cuda device:", torch.cuda.get_device_name(0))

    model = ICONLM(**cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    test = np.load("test_data.npz", allow_pickle=True)
    data = {key: test[key] for key in test.files}
    names = _as_str_list(data.get("regime_names",
                                  np.array(["diffusion", "shear", "vortical"])))
    sampler = PromptSampler(data, J, nC, nU, c_scale, u_scale,
                            c_mean, u_mean, rng_seed=42)

    # Dense inference is expensive, so evaluate a small reproducible subset and
    # show the median-error trajectory rather than handpicking the prettiest one.
    q_src = 0
    rng = np.random.default_rng(2026)
    candidates = np.arange(data["c"].shape[0])
    if len(candidates) > 6:
        candidates = rng.choice(candidates, size=6, replace=False)

    scored = []
    for op in candidates:
        target = data["u"][op, q_src]
        pred = _predict_trajectory(model, sampler, int(op), q_src, J, device)
        rel = np.linalg.norm(target - pred) / max(np.linalg.norm(target), 1e-8)
        scored.append((float(rel), int(op), target, pred))
        regime = int(data["regime"][op])
        print(f"{names[regime]:10s} op={int(op):3d} "
              f"nu={data['nu'][op]:.5f} rel L2={rel*100:.2f}%")

    scored.sort(key=lambda x: x[0])
    rel, op, target, pred = scored[len(scored) // 2]
    diff = pred - target
    regime = int(data["regime"][op])
    print(f"selected op={op}, regime={names[regime]}, "
          f"nu={data['nu'][op]:.5f}, rel L2={rel*100:.2f}%")

    n_cols = target.shape[0]
    fig, axes = plt.subplots(3, n_cols, figsize=(2.35 * n_cols, 5.9))
    row_titles = ["Target  $q(t)$",
                  "Prediction  $\\hat q(t)$",
                  "$\\hat q-q$"]
    extent = [0, 1, 0, 1]

    field_vmin = min(float(target.min()), float(pred.min()), 0.10)
    field_vmax = max(float(target.max()), float(pred.max()), 1.80)
    diff_vmax = max(float(np.abs(diff).max()), 1e-6)
    field_kw = dict(cmap="RdBu_r", vmin=field_vmin, vmax=field_vmax,
                    origin="lower", extent=extent, interpolation="bicubic")
    diff_kw = dict(cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax,
                   origin="lower", extent=extent, interpolation="bicubic")

    step_rels = []
    for col in range(n_cols):
        step_rel = np.linalg.norm(target[col] - pred[col]) / max(np.linalg.norm(target[col]), 1e-8)
        step_rels.append(step_rel)
        im0 = axes[0, col].imshow(target[col], **field_kw)
        im1 = axes[1, col].imshow(pred[col], **field_kw)
        im2 = axes[2, col].imshow(diff[col], **diff_kw)
        axes[0, col].set_title(f"Step {col + 1}\n"
                               f"$t={data['times'][col]:.2f}$, "
                               f"err={step_rel*100:.1f}%",
                               fontsize=9)
        if col == n_cols - 1:
            cb_kw = dict(fraction=0.046, pad=0.04)
            fig.colorbar(im0, ax=axes[0, col], **cb_kw)
            fig.colorbar(im1, ax=axes[1, col], **cb_kw)
            fig.colorbar(im2, ax=axes[2, col], **cb_kw)

    for row in range(3):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=11)
        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Toy multi-regime 2D fluid tracer forecast: "
                 "$q_t + u\\cdot\\nabla q = \\kappa\\Delta q$, "
                 f"{J} prompt trajectories, "
                 f"{names[regime]}, $\\kappa$={data['nu'][op]:.5f}",
                 y=1.00, fontsize=11)
    fig.tight_layout()
    fig.savefig("figure.pdf", bbox_inches="tight")
    fig.savefig("figure.png", dpi=230, bbox_inches="tight")
    print("per-step rel L2:", " ".join(f"{100*x:.1f}%" for x in step_rels))
    print("wrote figure.pdf and figure.png")


if __name__ == "__main__":
    main()
