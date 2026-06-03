"""
Evaluate the ICON-LM Burgers experiment.

1. Forward context scaling: relative L2 error vs number of demo shots,
   for three caption modes (precise / vague / none).
2. Inverse viscosity estimation: nearest-centroid classifier on trajectory
   features to estimate which ν-bin a held-out operator belongs to.

Run after train.py: expects checkpoint.pt and test_data.npz.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data import make_caption
from model import ICONLM, build_mask
from train import PromptSampler, _make_types_examples, EMBED_DIM


ROOT = Path(__file__).resolve().parent


# ── helpers ───────────────────────────────────────────────────────────────────

def _condition_indices(rng, N, idx_q, nC):
    extra = np.setdiff1d(np.arange(N), idx_q, assume_unique=True)
    n_ex  = max(0, nC - len(idx_q))
    return np.concatenate([idx_q, rng.choice(extra, size=n_ex, replace=False)])[:nC]


def _layout_for_demo_count(J_demo, nC, nU_total, device):
    block_sizes = []
    for _ in range(J_demo):
        block_sizes.extend([(0, nC), (1, nU_total), (2, nU_total)])
    block_sizes.extend([(0, nC), (2, nU_total)])
    return _make_types_examples(block_sizes, device)


def _sample_sparse_query_batch(sampler: PromptSampler, batch: int,
                                J_demo: int, rng: np.random.Generator,
                                caption_mode: "str | None" = None):
    """Build one batch of sparse-query prompts; returns 5-tuple."""
    nC, nU, nU_total = sampler.nC, sampler.nU_per_frame, sampler.nU_total
    N   = sampler.N
    L   = J_demo * (nC + nU_total + nU_total) + nC + nU_total

    coords      = np.zeros((batch, L, 2), dtype=np.float32)
    values      = np.zeros((batch, L, 1), dtype=np.float32)
    targets     = np.zeros((batch, L),    dtype=np.float32)
    target_mask = np.zeros((batch, L),    dtype=np.float32)
    cap_embs    = (np.zeros((batch, EMBED_DIM), dtype=np.float32)
                   if caption_mode is not None else None)

    for b in range(batch):
        op         = int(rng.integers(sampler.O))
        query_src  = int(rng.integers(sampler.S))
        src_pool   = [i for i in range(sampler.S) if i != query_src]
        demo_src   = rng.choice(src_pool, size=J_demo, replace=False)
        pos        = 0

        for src in demo_src:
            c_field = (sampler.c[op, src] - sampler.c_mean) / sampler.c_scale
            u_traj  = (sampler.u[op, src] - sampler.u_mean) / sampler.u_scale

            idx_q = rng.choice(N, size=nU, replace=False)
            idx_c = _condition_indices(rng, N, idx_q, nC)

            coords[b, pos:pos+nC]    = sampler._coords_at_time(idx_c, 0.0)
            values[b, pos:pos+nC, 0] = c_field[idx_c]
            pos += nC

            for k, t in enumerate(sampler.times):
                sl = slice(pos + k*nU, pos + (k+1)*nU)
                coords[b, sl]    = sampler._coords_at_time(idx_q, float(t))
                values[b, sl, 0] = u_traj[k, idx_q]
            pos += nU_total

            for k, t in enumerate(sampler.times):
                sl = slice(pos + k*nU, pos + (k+1)*nU)
                coords[b, sl]    = sampler._coords_at_time(idx_q, float(t))
                values[b, sl, 0] = c_field[idx_q]
            pos += nU_total

        # query example
        c_query = (sampler.c[op, query_src] - sampler.c_mean) / sampler.c_scale
        u_query = (sampler.u[op, query_src] - sampler.u_mean) / sampler.u_scale
        idx_q   = rng.choice(N, size=nU, replace=False)
        idx_c   = _condition_indices(rng, N, idx_q, nC)

        coords[b, pos:pos+nC]    = sampler._coords_at_time(idx_c, 0.0)
        values[b, pos:pos+nC, 0] = c_query[idx_c]
        pos += nC

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k*nU, pos + (k+1)*nU)
            coords[b, sl]    = sampler._coords_at_time(idx_q, float(t))
            values[b, sl, 0] = c_query[idx_q]
            targets[b, sl]   = u_query[k, idx_q]
            target_mask[b, sl] = 1.0

        if caption_mode is not None:
            emb_idx      = 0 if caption_mode == "precise" else 1
            cap_embs[b]  = sampler.caption_embs[op, emb_idx]

    cap_tensor = (torch.from_numpy(cap_embs).unsqueeze(1)
                  if cap_embs is not None else None)  # [B, 1, 384] or None
    return (torch.from_numpy(coords), torch.from_numpy(values),
            torch.from_numpy(targets), torch.from_numpy(target_mask),
            cap_tensor)


# ── forward context scaling ───────────────────────────────────────────────────

def _eval_one_config(model, sampler, device, shot_counts, prompts,
                     batch_size, seed, caption_mode):
    rng  = np.random.default_rng(seed)
    rows = []
    model.eval()

    for J_demo in shot_counts:
        J_demo = min(J_demo, sampler.S - 1)
        types, examples, demo_idx, L = _layout_for_demo_count(
            J_demo, sampler.nC, sampler.nU_total, device)
        cap_len = 1 if caption_mode is not None else 0
        mask    = build_mask(types, examples, caption_len=cap_len)

        rel_num = mse_num = 0.0
        rel_den = mse_den = 0.0
        n_done  = 0
        while n_done < prompts:
            cb = min(batch_size, prompts - n_done)
            coords, values, targets, target_mask, cap_emb = \
                _sample_sparse_query_batch(sampler, cb, J_demo, rng, caption_mode)
            coords      = coords.to(device)
            values      = values.to(device)
            targets     = targets.to(device)
            target_mask = target_mask.to(device)
            cap_emb     = cap_emb.to(device) if cap_emb is not None else None
            types_b     = types[None].expand(cb, L)

            with torch.no_grad():
                pred = model(types_b, coords, values, mask,
                             demo_idx=demo_idx, caption_emb=cap_emb)

            sq      = (pred - targets) ** 2 * target_mask
            mse_num += float(sq.sum())
            mse_den += float(target_mask.sum())

            pu = pred    * sampler.u_scale + sampler.u_mean
            tu = targets * sampler.u_scale + sampler.u_mean
            rel_num += float(((pu - tu)**2 * target_mask).sum())
            rel_den += float((tu**2        * target_mask).sum())
            n_done  += cb

        rows.append({
            "shots":   J_demo,
            "prompts": prompts,
            "caption": caption_mode or "none",
            "scaled_mse": mse_num / max(mse_den, 1.0),
            "relative_l2_percent": 100.0 * (rel_num / max(rel_den, 1e-8)) ** 0.5,
        })
    return rows


def evaluate_context_scaling(model, sampler, device, shot_counts,
                              prompts, batch_size, seed):
    results = {}
    for mode in ("precise", "vague", None):
        results[mode or "none"] = _eval_one_config(
            model, sampler, device, shot_counts, prompts, batch_size, seed, mode)
    return results


# ── inverse ν estimation ──────────────────────────────────────────────────────

def _traj_features(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """5 scalar features per operator: temporal change, gradient, spectral."""
    S, K, N = u.shape
    c_flat  = c.reshape(S, -1)
    u_flat  = u.reshape(S, K, -1)
    c_norm  = np.linalg.norm(c_flat, axis=1) + 1e-8

    rel_change   = np.mean(np.linalg.norm(u_flat - c_flat[:,None,:], axis=2) / c_norm[:,None])
    last_change  = np.mean(np.linalg.norm(u_flat[:,-1] - c_flat, axis=1) / c_norm)
    frame_change = np.mean(np.linalg.norm(np.diff(u_flat, axis=1), axis=2)
                           / (np.linalg.norm(u_flat[:,:-1,:], axis=2) + 1e-8))

    # High-frequency power ratio (proxy for viscosity: low ν → more HF content)
    freqs  = np.abs(np.fft.rfft(u[:, -1], axis=-1)) ** 2   # [S, N//2+1]
    hf_cut = N // 4
    hf_rat = np.mean(freqs[:, hf_cut:].mean(-1) / (freqs.mean(-1) + 1e-12))

    grad_mag = np.mean(np.abs(np.diff(u[:,-1], axis=-1)))

    return np.array([rel_change, last_change, frame_change, hf_rat, grad_mag],
                    dtype=np.float64)


def evaluate_inverse_nu(data: dict, n_bins: int = 5, seed: int = 0):
    """
    Bin operators by log-ν, then do nearest-centroid classification.
    Returns (accuracy, bin_edges, confusion_matrix).
    """
    nu      = data["nu"]
    log_nu  = np.log(nu)
    edges   = np.linspace(log_nu.min(), log_nu.max(), n_bins + 1)
    labels  = np.digitize(log_nu, edges[1:-1])   # 0 … n_bins-1

    features = np.stack([
        _traj_features(data["c"][op], data["u"][op])
        for op in range(len(nu))
    ])

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in range(n_bins):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        sp  = max(1, len(idx) // 2)
        train_idx.extend(idx[:sp])
        test_idx.extend(idx[sp:] if sp < len(idx) else idx[:sp])

    train_idx = np.array(train_idx, dtype=int)
    test_idx  = np.array(test_idx,  dtype=int)
    mean  = features[train_idx].mean(0)
    scale = features[train_idx].std(0) + 1e-8
    z     = (features - mean) / scale

    centroids = np.stack([z[train_idx[labels[train_idx]==c]].mean(0)
                          for c in range(n_bins)])
    dists = ((z[test_idx,None,:] - centroids[None,:,:]) ** 2).sum(-1)
    pred  = dists.argmin(-1)
    true  = labels[test_idx]
    acc   = float((pred == true).mean())

    confusion = np.zeros((n_bins, n_bins), dtype=int)
    for t, p in zip(true, pred):
        confusion[int(t), int(p)] += 1

    bin_names = [f"nu~{np.exp((edges[i]+edges[i+1])/2):.4f}"
                 for i in range(n_bins)]
    return acc, bin_names, confusion


# ── csv / tex writers ─────────────────────────────────────────────────────────

def write_context_csv(results, path: Path):
    all_rows = [r for rows in results.values() for r in rows]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "shots", "prompts", "caption", "scaled_mse", "relative_l2_percent"])
        writer.writeheader()
        writer.writerows(all_rows)


def write_confusion_csv(confusion, names, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_bin", "pred_bin", "count"])
        writer.writeheader()
        for i, tn in enumerate(names):
            for j, pn in enumerate(names):
                writer.writerow({"true_bin": tn, "pred_bin": pn,
                                 "count": int(confusion[i, j])})


def write_eval_metrics(results, inverse_acc, path: Path):
    rows_nc = results.get("none", next(iter(results.values())))
    by_shot = {int(r["shots"]): r for r in rows_nc}
    s_min, s_max = min(by_shot), max(by_shot)
    rows_pr = {int(r["shots"]): r for r in results.get("precise", [])}

    zs_line = ""
    if s_min in rows_pr and s_min in by_shot:
        nc0 = by_shot[s_min]["relative_l2_percent"]
        pr0 = rows_pr[s_min]["relative_l2_percent"]
        imp = (nc0 - pr0) / (nc0 + 1e-8) * 100
        zs_line = f"\\newcommand{{\\ToyZeroShotImp}}{{{imp:.1f}\\%}}\n"

    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by toy_multiphysics/evaluate.py\n")
        f.write("\\newcommand{\\ToyContextSummary}{"
                f"{s_min}-shot {by_shot[s_min]['relative_l2_percent']:.1f}\\%, "
                f"{s_max}-shot {by_shot[s_max]['relative_l2_percent']:.1f}\\%}}\n")
        f.write(f"\\newcommand{{\\ToyInverseAcc}}{{{100*inverse_acc:.1f}\\%}}\n")
        if zs_line:
            f.write(zs_line)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots",      default="0,1,2,4")
    parser.add_argument("--prompts",    type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed",       type=int, default=2026)
    args = parser.parse_args()

    ckpt = torch.load(ROOT / "checkpoint.pt", map_location="cpu",
                      weights_only=False)
    raw  = np.load(ROOT / "test_data.npz", allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device  = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)

    model = ICONLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print("pre-computing test sentence embeddings ...")
    sampler = PromptSampler(data, ckpt["J"], ckpt["nC"], ckpt["nU"],
                            ckpt["c_scale"], ckpt["u_scale"],
                            ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
                            rng_seed=args.seed)

    shot_counts = [int(x) for x in args.shots.split(",") if x.strip()]
    results     = evaluate_context_scaling(
        model, sampler, device, shot_counts,
        args.prompts, args.batch_size, args.seed)

    print("Forward context scaling (precise / vague / none):")
    print(f"  {'shots':>5}  {'precise':>9}  {'vague':>9}  {'none':>9}")
    pr = {r["shots"]: r for r in results.get("precise", [])}
    vg = {r["shots"]: r for r in results.get("vague",   [])}
    nc = {r["shots"]: r for r in results.get("none",    [])}
    for s in shot_counts:
        print(f"  {s:>5}  {pr.get(s,{}).get('relative_l2_percent',float('nan')):>8.2f}%"
              f"  {vg.get(s,{}).get('relative_l2_percent',float('nan')):>8.2f}%"
              f"  {nc.get(s,{}).get('relative_l2_percent',float('nan')):>8.2f}%")

    acc, bin_names, confusion = evaluate_inverse_nu(data, seed=args.seed)
    print(f"Inverse ν-bin accuracy: {100*acc:.1f}%")
    print("Confusion (rows=true, cols=pred):")
    print("  " + "  ".join(f"{n:>12s}" for n in bin_names))
    for i, name in enumerate(bin_names):
        vals = "  ".join(f"{int(confusion[i,j]):>12d}" for j in range(len(bin_names)))
        print(f"  {name:>12s}  {vals}")

    write_context_csv(results, ROOT / "context_scaling.csv")
    write_confusion_csv(confusion, bin_names, ROOT / "inverse_confusion.csv")
    write_eval_metrics(results, acc, ROOT / "eval_metrics.tex")
    print("wrote context_scaling.csv, inverse_confusion.csv, eval_metrics.tex")


if __name__ == "__main__":
    main()
