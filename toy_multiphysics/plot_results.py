"""
Publication-quality results figures for the ICON-LM 1D Burgers experiment.
Matches the evaluation and presentation style of Yang, Liu, Osher (2023).

Generates  results_figure.pdf / .png  with four panels:
  (a) Few-shot context scaling   — relative L2 vs shots for 3 caption modes
  (b) Zero-shot caption benefit  — bar chart with persistence baseline
  (c) Per-regime in-context gain — low / mid / high ν at 0- and 2-shot
  (d) Temporal error growth      — per-timestep error for low vs high ν
"""

from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR",
                      os.path.join(os.path.dirname(__file__), ".matplotlib"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from model  import ICONLM, build_mask
from train  import (PromptSampler, make_layout, predict_full_grid_step,
                    EMBED_DIM, J, N_COND, N_QOI, BATCH)
from evaluate import _eval_one_config

ROOT = Path(__file__).resolve().parent

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "precise": "#2166AC",   # strong blue
    "vague":   "#762A83",   # purple
    "none":    "#D6604D",   # red-orange
    "low":     "#1A9641",   # dark green  (sharp, low ν)
    "mid":     "#F4A800",   # amber       (mid ν)
    "high":    "#D7191C",   # red         (smooth, high ν)
    "persist": "#888888",   # grey        (baseline)
}

# ── global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     9.5,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "lines.linewidth":    1.8,
    "lines.markersize":   5.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "figure.dpi":         150,
})


# ── evaluation helpers ────────────────────────────────────────────────────────

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


def eval_regime(model, data, ckpt, device,
                n_prompts=300, batch_size=8,
                seeds=(2026, 2027, 2028)):
    """Return 0-shot and 2-shot relative L2 for three ν regimes."""
    nu   = data["nu"]
    q33, q67 = np.percentile(nu, 33), np.percentile(nu, 67)
    regimes  = {
        "low":  nu <= q33,
        "mid":  (nu > q33) & (nu < q67),
        "high": nu >= q67,
    }
    result = {}
    for regime_name, mask in regimes.items():
        sub = {k: (v[mask] if hasattr(v, "shape") and v.shape[0] == len(nu) else v)
               for k, v in data.items()}
        ss  = PromptSampler(sub, J, N_COND, N_QOI,
                            ckpt["c_scale"], ckpt["u_scale"],
                            ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
                            rng_seed=42)
        by_shot = {0: [], 2: []}
        for seed in seeds:
            for shots in (0, 2):
                rows = _eval_one_config(model, ss, device, [shots],
                                        n_prompts, batch_size, seed, None)
                by_shot[shots].append(rows[0]["relative_l2_percent"])
        result[regime_name] = {
            s: (float(np.mean(v)), float(np.std(v)))
            for s, v in by_shot.items()
        }
    return result


def eval_temporal(model, sampler, data, device,
                  n_ops=16, src_idx=0):
    """Per-timestep relative L2 for low-ν and high-ν operators."""
    nu    = data["nu"]
    q25   = np.percentile(nu, 25)
    q75   = np.percentile(nu, 75)
    rng   = np.random.default_rng(2026)
    low_ops  = rng.choice(np.flatnonzero(nu <= q25),
                          size=min(n_ops, (nu <= q25).sum()), replace=False)
    high_ops = rng.choice(np.flatnonzero(nu >= q75),
                          size=min(n_ops, (nu >= q75).sum()), replace=False)
    times    = data["times"]                          # [K]
    K        = len(times)

    def _op_errors(op_list):
        per_step = [[] for _ in range(K)]
        for op in op_list:
            true_traj = data["u"][op, src_idx]       # [K, N]
            for step_idx in range(K):
                pred = predict_full_grid_step(
                    model, sampler, int(op), src_idx, J, step_idx, device)
                true = true_traj[step_idx]
                rel  = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8)
                per_step[step_idx].append(rel * 100)
        return [np.mean(v) for v in per_step], [np.std(v) for v in per_step]

    low_mean,  low_std  = _op_errors(low_ops)
    high_mean, high_std = _op_errors(high_ops)
    return times, low_mean, low_std, high_mean, high_std


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_scaling(ax, scaling, shot_counts, persistence_pct):
    """Panel (a): few-shot scaling curves."""
    labels = {"precise": "Precise caption", "vague": "Vague caption",
              "none":    "No caption"}
    markers = {"precise": "o", "vague": "s", "none": "^"}

    for mode in ("none", "vague", "precise"):
        xs = list(shot_counts)
        ys = [scaling[mode][s][0] for s in xs]
        es = [scaling[mode][s][1] for s in xs]
        ax.plot(xs, ys, marker=markers[mode], color=C[mode], label=labels[mode])
        ax.fill_between(xs,
                        [y - e for y, e in zip(ys, es)],
                        [y + e for y, e in zip(ys, es)],
                        color=C[mode], alpha=0.12)

    ax.axhline(persistence_pct, color=C["persist"], lw=1.2, ls="--",
               label=f"Persistence ({persistence_pct:.0f}%)")
    ax.set_xlabel("In-context demonstrations (shots)")
    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.set_title("(a)  Few-shot context scaling")
    ax.set_xticks(list(shot_counts))
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)


def plot_zero_shot_bar(ax, scaling, persistence_pct):
    """Panel (b): zero-shot caption benefit bar chart."""
    modes  = ["none", "vague", "precise"]
    labels = ["No caption", "Vague\ncaption", "Precise\ncaption"]
    vals   = [scaling[m][0][0] for m in modes]
    errs   = [scaling[m][0][1] for m in modes]
    cols   = [C[m] for m in modes]

    bars = ax.bar(labels, vals, color=cols, width=0.5,
                  yerr=errs, capsize=4, error_kw={"elinewidth": 1.2})
    ax.axhline(persistence_pct, color=C["persist"], lw=1.2, ls="--",
               label=f"Persistence ({persistence_pct:.0f}%)")

    # Annotate bars with their values.
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.set_title("(b)  Zero-shot caption benefit")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)


def plot_regime(ax, regime_data, persistence_pct):
    """Panel (c): per-regime grouped bar chart."""
    regime_labels = {"low": r"Low $\nu$" + "\n(shocks)",
                     "mid": r"Mid $\nu$",
                     "high": r"High $\nu$" + "\n(smooth)"}
    x      = np.arange(3)
    width  = 0.32
    regimes = ["low", "mid", "high"]

    bars0 = ax.bar(x - width/2,
                   [regime_data[r][0][0] for r in regimes],
                   width, color=[C[r] for r in regimes], alpha=0.5,
                   yerr=[regime_data[r][0][1] for r in regimes],
                   capsize=3, error_kw={"elinewidth":1.1},
                   label="0-shot", hatch="//")
    bars2 = ax.bar(x + width/2,
                   [regime_data[r][2][0] for r in regimes],
                   width, color=[C[r] for r in regimes], alpha=1.0,
                   yerr=[regime_data[r][2][1] for r in regimes],
                   capsize=3, error_kw={"elinewidth":1.1},
                   label="2-shot")

    ax.axhline(persistence_pct, color=C["persist"], lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([regime_labels[r] for r in regimes])
    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.set_title("(c)  Per-regime in-context gain (no caption)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)


def plot_temporal(ax, times, low_mean, low_std, high_mean, high_std):
    """Panel (d): per-timestep error accumulation."""
    ax.plot(times, low_mean,  "o-", color=C["low"],  label=r"Low $\nu$ (shock)")
    ax.fill_between(times,
                    [m - s for m, s in zip(low_mean,  low_std)],
                    [m + s for m, s in zip(low_mean,  low_std)],
                    color=C["low"],  alpha=0.15)

    ax.plot(times, high_mean, "s-", color=C["high"], label=r"High $\nu$ (smooth)")
    ax.fill_between(times,
                    [m - s for m, s in zip(high_mean, high_std)],
                    [m + s for m, s in zip(high_mean, high_std)],
                    color=C["high"], alpha=0.15)

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Relative $L_2$ error (%)")
    ax.set_title("(d)  Temporal error accumulation (4-shot)")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    ax.set_xticks(times)
    ax.set_xticklabels([f"{t:.2f}" for t in times])


# ── persistence baseline ──────────────────────────────────────────────────────

def persistence_error(data):
    rel = 0.0; n = 0
    for op in range(len(data["nu"])):
        for s in range(data["u"].shape[1]):
            for k in range(data["u"].shape[2]):
                true  = data["u"][op, s, k]
                denom = np.linalg.norm(true) + 1e-8
                rel  += np.linalg.norm(true - data["c"][op, s]) / denom
                n    += 1
    return rel / n * 100


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ckpt   = torch.load(ROOT / "checkpoint.pt", map_location="cpu",
                        weights_only=False)
    raw    = np.load(ROOT / "test_data.npz", allow_pickle=True)
    data   = {k: raw[k] for k in raw.files}

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device  = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)

    model = ICONLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("pre-computing sentence embeddings ...")
    sampler = PromptSampler(data, ckpt["J"], ckpt["nC"], ckpt["nU"],
                            ckpt["c_scale"], ckpt["u_scale"],
                            ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
                            rng_seed=2026)

    persist = persistence_error(data)
    print(f"persistence baseline: {persist:.1f}%")

    # J=4 demos during training → max meaningful query position is J-1=3.
    # At J_demo=4 the query sits at demo_idx 12/14, never a training target.
    shot_counts = (0, 1, 2, 3)

    print("evaluating context scaling (3 seeds × 3 modes × 5 shot counts) ...")
    scaling = eval_scaling(model, sampler, device,
                           shot_counts=shot_counts,
                           n_prompts=300, batch_size=8)
    for mode in ("none", "vague", "precise"):
        line = "  %-8s" % mode
        for s in shot_counts:
            m, sd = scaling[mode][s]
            line += f"  {s}-shot:{m:.1f}±{sd:.1f}%"
        print(line)

    print("evaluating per-regime breakdown ...")
    regime_data = eval_regime(model, data, ckpt, device, n_prompts=200)
    for r in ("low", "mid", "high"):
        print(f"  {r:4s}  0-shot={regime_data[r][0][0]:.1f}%  "
              f"2-shot={regime_data[r][2][0]:.1f}%")

    print("computing per-timestep errors ...")
    times, lm, ls, hm, hs = eval_temporal(model, sampler, data, device, n_ops=10)
    for k, t in enumerate(times):
        print(f"  t={t:.2f}  low-ν:{lm[k]:.1f}±{ls[k]:.1f}%  "
              f"high-ν:{hm[k]:.1f}±{hs[k]:.1f}%")

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    plot_scaling(ax_a, scaling, shot_counts, persist)
    plot_zero_shot_bar(ax_b, scaling, persist)
    plot_regime(ax_c, regime_data, persist)
    plot_temporal(ax_d, times, lm, ls, hm, hs)

    fig.suptitle(
        "ICON-LM on 1D Viscous Burgers  "
        r"$u_t + u u_x = \nu u_{xx}$  "
        f"(d={ckpt['config']['d_model']}, "
        f"{ckpt['config']['n_layers']} layers, "
        f"{ckpt['J']}-shot demos)",
        fontsize=10, y=1.01)

    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(ROOT / "results_figure.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "results_figure.png", dpi=200, bbox_inches="tight")
    print("wrote results_figure.pdf and results_figure.png")

    # ── summary table (paper-style) ───────────────────────────────────────────
    # ── summary table ─────────────────────────────────────────────────────────
    col_w = 12
    header = f"{'':20s}" + "".join(f"  {s}-shot".rjust(col_w) for s in shot_counts)
    print()
    print("=" * (20 + col_w * len(shot_counts) + 2 * len(shot_counts)))
    print(header)
    print("-" * (20 + col_w * len(shot_counts) + 2 * len(shot_counts)))
    for mode, label in [("precise", "Precise caption"),
                         ("vague",   "Vague caption  "),
                         ("none",    "No caption     ")]:
        row = f"  {label:18s}"
        for s in shot_counts:
            m, sd = scaling[mode][s]
            row += f"  {m:5.1f}±{sd:.1f}%".rjust(col_w)
        print(row)
    print("-" * (20 + col_w * len(shot_counts) + 2 * len(shot_counts)))
    dashes = "        —   " * (len(shot_counts) - 1)
    print(f"  {'Persistence':18s}  {persist:5.1f}%      {dashes}")
    print("=" * (20 + col_w * len(shot_counts) + 2 * len(shot_counts)))


if __name__ == "__main__":
    main()
