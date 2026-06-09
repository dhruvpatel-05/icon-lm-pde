"""
Evaluate the trained ICON-LM Burgers model and render the results figures:
  scaling_curve.png  — few-shot context scaling (relative L2 vs shots, 3 caption modes)
  zero_shot_bar.png  — zero-shot caption benefit (bar chart)
  scaling_results.csv — the mean ± std values plotted in scaling_curve.png

Run after train.py: expects checkpoint.pt and test_data.npz.
"""

from __future__ import annotations
import csv
import os
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), ".matplotlib"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from model    import ICONLM
from train    import PromptSampler
from evaluate import _eval_one_config

ROOT = Path(__file__).resolve().parent

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "precise": "#2166AC",   # strong blue
    "vague":   "#762A83",   # purple
    "none":    "#D6604D",   # red-orange
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "lines.linewidth":   2.0,
    "lines.markersize":  6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.dpi":        150,
})


def eval_scaling(model, sampler, device,
                 shot_counts=(0, 1, 2, 3, 4),
                 n_prompts=300, batch_size=8,
                 seeds=(2026, 2027, 2028)):
    """Return mean±std relative L2 per (caption_mode, shots)."""
    modes  = ("precise", "vague", None)
    result = {}                          # mode → {shots: (mean, std)}
    for mode in modes:
        by_shot = {s: [] for s in shot_counts}
        for seed in seeds:
            rows = _eval_one_config(model, sampler, device, list(shot_counts),
                                    n_prompts, batch_size, seed, mode)
            for r in rows:
                by_shot[r["shots"]].append(r["relative_l2_percent"])
        result[mode or "none"] = {
            s: (float(np.mean(v)), float(np.std(v)))
            for s, v in by_shot.items()
        }
    return result


def main():
    ckpt = torch.load(ROOT / "checkpoint.pt", map_location="cpu",
                      weights_only=False)
    raw  = np.load(ROOT / "test_data.npz", allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device  = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")

    model = ICONLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("loading sentence embeddings ...")
    sampler = PromptSampler(data, ckpt["J"], ckpt["nC"], ckpt["nU"],
                            ckpt["c_scale"], ckpt["u_scale"],
                            ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
                            rng_seed=2026)

    print("evaluating ...")
    shot_counts = (0, 1, 2, 3)
    scaling = eval_scaling(model, sampler, device,
                           shot_counts=shot_counts,
                           n_prompts=300, batch_size=8)

    print(f"\n{'Shots':>6}  {'No caption':>12}  {'Vague caption':>14}  {'Precise caption':>16}")
    print("-" * 56)
    for s in shot_counts:
        nm, ns = scaling["none"][s]
        vm, vs = scaling["vague"][s]
        pm, ps = scaling["precise"][s]
        print(f"{s:>6}  {nm:6.2f}±{ns:.2f}%  {vm:6.2f}±{vs:.2f}%  {pm:8.2f}±{ps:.2f}%")
    print()

    csv_path = ROOT / "scaling_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shots", "caption_mode", "relative_l2_percent_mean",
                         "relative_l2_percent_std"])
        for mode in ("none", "vague", "precise"):
            for s in shot_counts:
                m, sd = scaling[mode][s]
                writer.writerow([s, mode, f"{m:.4f}", f"{sd:.4f}"])
    print(f"wrote {csv_path.name}")

    labels  = {"precise": "Precise caption",
               "vague":   "Vague caption",
               "none":    "No caption"}
    markers = {"precise": "o", "vague": "s", "none": "^"}

    # ── Few-shot scaling curve ────────────────────────────────────────────────
    fig_a, ax_a = plt.subplots(figsize=(5.5, 4.0))

    for mode in ("none", "vague", "precise"):
        xs = list(shot_counts)
        ys = [scaling[mode][s][0] for s in xs]
        es = [scaling[mode][s][1] for s in xs]
        ax_a.plot(xs, ys, marker=markers[mode], color=C[mode],
                  label=labels[mode])
        ax_a.fill_between(xs,
                          [y - e for y, e in zip(ys, es)],
                          [y + e for y, e in zip(ys, es)],
                          color=C[mode], alpha=0.12)

    ax_a.set_xlabel("Examples in prompt $J$", fontsize=13)
    ax_a.set_ylabel("Relative $L_2$ error (%)", fontsize=13)
    ax_a.set_title("")
    ax_a.set_xticks(list(shot_counts))
    ax_a.legend(loc="upper right")
    ax_a.set_ylim(bottom=0)

    fig_a.tight_layout()
    fig_a.savefig(ROOT / "scaling_curve.png", dpi=200, bbox_inches="tight")
    print("wrote scaling_curve.png")

    # ── Zero-shot caption benefit ─────────────────────────────────────────────
    fig_b, ax_b = plt.subplots(figsize=(4.5, 4.0))

    modes  = ["none", "vague", "precise"]
    blabels = ["No caption", "Vague\ncaption", "Precise\ncaption"]
    vals   = [scaling[m][0][0] for m in modes]
    errs   = [scaling[m][0][1] for m in modes]
    cols   = [C[m] for m in modes]

    bars = ax_b.bar(blabels, vals, color=cols, width=0.5,
                    yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})

    for bar, val in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.4,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax_b.set_ylabel("Relative $L_2$ error (%)")
    ax_b.set_title("Zero-shot caption benefit")
    ax_b.set_ylim(bottom=0)

    fig_b.tight_layout()
    fig_b.savefig(ROOT / "zero_shot_bar.png", dpi=200, bbox_inches="tight")
    print("wrote zero_shot_bar.png")


if __name__ == "__main__":
    main()
