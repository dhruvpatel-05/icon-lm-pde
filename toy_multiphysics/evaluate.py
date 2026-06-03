"""
Evaluate the toy multi-regime fluid-tracer experiment.

This script adds two rubric-friendly diagnostics after training:

1. Forward in-context scaling: relative prediction error versus the number of
   demonstration trajectories in the prompt.
2. Inverse regime identification: a small trajectory-feature classifier that
   predicts whether a held-out operator is diffusion, shear, or vortical.

Run after train.py has produced checkpoint.pt and test_data.npz.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch

from model import ICONLM, build_mask
from train import PromptSampler, _make_types_examples


ROOT = Path(__file__).resolve().parent


def _as_str_list(arr) -> list[str]:
    return [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in arr]


def _condition_indices(rng: np.random.Generator, N2: int, idx_q: np.ndarray,
                       nC: int) -> np.ndarray:
    extra_pool = np.setdiff1d(np.arange(N2), idx_q, assume_unique=True)
    n_extra = max(0, nC - len(idx_q))
    idx_extra = rng.choice(extra_pool, size=n_extra, replace=False)
    return np.concatenate([idx_q, idx_extra])[:nC]


def _layout_for_demo_count(J_demo: int, nC: int, nU_total: int, device):
    block_sizes = []
    for _ in range(J_demo):
        block_sizes.extend([(0, nC), (1, nU_total), (2, nU_total)])
    block_sizes.extend([(0, nC), (2, nU_total)])
    return _make_types_examples(block_sizes, device)


def _sample_sparse_query_batch(sampler: PromptSampler, batch: int, J_demo: int,
                               rng: np.random.Generator):
    nC = sampler.nC
    nU = sampler.nU_per_frame
    nU_total = sampler.nU_total
    N2 = sampler.N * sampler.N
    L = J_demo * (nC + nU_total + nU_total) + nC + nU_total

    coords = np.zeros((batch, L, 3), dtype=np.float32)
    values = np.zeros((batch, L, 1), dtype=np.float32)
    targets = np.zeros((batch, L), dtype=np.float32)
    target_mask = np.zeros((batch, L), dtype=np.float32)

    for b in range(batch):
        op = int(rng.integers(sampler.O))
        query_src = int(rng.integers(sampler.S))
        src_pool = [i for i in range(sampler.S) if i != query_src]
        demo_src = rng.choice(src_pool, size=J_demo, replace=False)
        pos = 0

        for src in demo_src:
            c_field = ((sampler.c[op, src].reshape(-1) - sampler.c_mean)
                       / sampler.c_scale)
            u_traj = ((sampler.u[op, src].reshape(sampler.n_frames, -1)
                       - sampler.u_mean) / sampler.u_scale)

            idx_q = rng.choice(N2, size=nU, replace=False)
            idx_c = _condition_indices(rng, N2, idx_q, nC)

            coords[b, pos:pos + nC] = sampler._coords_at_time(idx_c, 0.0)
            values[b, pos:pos + nC, 0] = c_field[idx_c]
            pos += nC

            for k, t in enumerate(sampler.times):
                sl = slice(pos + k * nU, pos + (k + 1) * nU)
                coords[b, sl] = sampler._coords_at_time(idx_q, float(t))
                values[b, sl, 0] = u_traj[k, idx_q]
            pos += nU_total

            for k, t in enumerate(sampler.times):
                sl = slice(pos + k * nU, pos + (k + 1) * nU)
                coords[b, sl] = sampler._coords_at_time(idx_q, float(t))
                values[b, sl, 0] = c_field[idx_q]
            pos += nU_total

        c_query = ((sampler.c[op, query_src].reshape(-1) - sampler.c_mean)
                   / sampler.c_scale)
        u_query = ((sampler.u[op, query_src].reshape(sampler.n_frames, -1)
                   - sampler.u_mean) / sampler.u_scale)
        idx_q = rng.choice(N2, size=nU, replace=False)
        idx_c = _condition_indices(rng, N2, idx_q, nC)

        coords[b, pos:pos + nC] = sampler._coords_at_time(idx_c, 0.0)
        values[b, pos:pos + nC, 0] = c_query[idx_c]
        pos += nC

        for k, t in enumerate(sampler.times):
            sl = slice(pos + k * nU, pos + (k + 1) * nU)
            coords[b, sl] = sampler._coords_at_time(idx_q, float(t))
            values[b, sl, 0] = c_query[idx_q]
            targets[b, sl] = u_query[k, idx_q]
            target_mask[b, sl] = 1.0

    return (
        torch.from_numpy(coords),
        torch.from_numpy(values),
        torch.from_numpy(targets),
        torch.from_numpy(target_mask),
    )


def evaluate_context_scaling(model: ICONLM, sampler: PromptSampler, device: str,
                             shot_counts: list[int], prompts: int,
                             batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    model.eval()

    for J_demo in shot_counts:
        J_demo = min(J_demo, sampler.S - 1)
        types, examples, L = _layout_for_demo_count(
            J_demo, sampler.nC, sampler.nU_total, device
        )
        mask = build_mask(types, examples)
        rel_num = torch.tensor(0.0, device=device)
        rel_den = torch.tensor(0.0, device=device)
        mse_num = torch.tensor(0.0, device=device)
        mse_den = torch.tensor(0.0, device=device)

        n_done = 0
        while n_done < prompts:
            current_batch = min(batch_size, prompts - n_done)
            coords, values, targets, target_mask = _sample_sparse_query_batch(
                sampler, current_batch, J_demo, rng
            )
            coords = coords.to(device)
            values = values.to(device)
            targets = targets.to(device)
            target_mask = target_mask.to(device)
            types_b = types[None].expand(current_batch, L)

            with torch.no_grad():
                pred = model(types_b, coords, values, mask)

            sq = (pred - targets) ** 2 * target_mask
            mse_num += sq.sum()
            mse_den += target_mask.sum()

            pred_u = pred * sampler.u_scale + sampler.u_mean
            target_u = targets * sampler.u_scale + sampler.u_mean
            rel_num += ((pred_u - target_u) ** 2 * target_mask).sum()
            rel_den += (target_u ** 2 * target_mask).sum()
            n_done += current_batch

        rows.append({
            "shots": J_demo,
            "prompts": prompts,
            "scaled_mse": float((mse_num / mse_den.clamp_min(1.0)).item()),
            "relative_l2_percent": float(
                100.0 * torch.sqrt(rel_num / rel_den.clamp_min(1e-8)).item()
            ),
        })

    return rows


def _high_frequency_fraction(fields: np.ndarray) -> np.ndarray:
    N = fields.shape[-1]
    freq = np.fft.fftfreq(N, d=1.0 / N)
    kx, ky = np.meshgrid(freq, freq, indexing="ij")
    mask = np.sqrt(kx**2 + ky**2) > (N / 5.0)
    power = np.abs(np.fft.fft2(fields, axes=(-2, -1))) ** 2
    high = power[..., mask].mean(axis=-1)
    total = power.mean(axis=(-2, -1)) + 1e-12
    return high / total


def _operator_features(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    # c: [S, N, N], u: [S, K, N, N]
    S, K = u.shape[:2]
    c_flat = c.reshape(S, -1)
    u_flat = u.reshape(S, K, -1)
    c_norm = np.linalg.norm(c_flat, axis=1) + 1e-8
    u_norm = np.linalg.norm(u_flat, axis=2) + 1e-8

    rel_change = np.mean(np.linalg.norm(u_flat - c_flat[:, None, :],
                                        axis=2) / c_norm[:, None])
    last_change = np.mean(np.linalg.norm(u_flat[:, -1] - c_flat,
                                         axis=1) / c_norm)
    frame_change = np.mean(np.linalg.norm(np.diff(u_flat, axis=1), axis=2)
                           / u_norm[:, :-1])
    std_ratio = np.mean(np.std(u[:, -1], axis=(-2, -1))
                        / (np.std(c, axis=(-2, -1)) + 1e-8))
    hf_ratio = np.mean(_high_frequency_fraction(u[:, -1])
                       / (_high_frequency_fraction(c) + 1e-8))
    return np.array([rel_change, last_change, frame_change, std_ratio, hf_ratio],
                    dtype=np.float64)


def evaluate_inverse_regime(data: dict, seed: int):
    labels = data["regime"].astype(int)
    n_classes = int(labels.max()) + 1
    features = np.stack([
        _operator_features(data["c"][op], data["u"][op])
        for op in range(labels.shape[0])
    ])

    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in range(n_classes):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        split = max(1, len(idx) // 2)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:] if split < len(idx) else idx[:split])

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    mean = features[train_idx].mean(axis=0)
    scale = features[train_idx].std(axis=0) + 1e-8
    z = (features - mean) / scale
    centroids = np.stack([
        z[train_idx[labels[train_idx] == cls]].mean(axis=0)
        for cls in range(n_classes)
    ])

    distances = ((z[test_idx, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = distances.argmin(axis=1)
    true = labels[test_idx]
    acc = float(np.mean(pred == true))

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        confusion[int(t), int(p)] += 1

    return acc, confusion


def write_context_csv(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["shots", "prompts", "scaled_mse",
                           "relative_l2_percent"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(confusion: np.ndarray, names: list[str], path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_regime", "pred_regime",
                                               "count"])
        writer.writeheader()
        for i, true_name in enumerate(names):
            for j, pred_name in enumerate(names):
                writer.writerow({
                    "true_regime": true_name,
                    "pred_regime": pred_name,
                    "count": int(confusion[i, j]),
                })


def write_eval_metrics(rows, inverse_acc: float, path: Path):
    row_by_shot = {int(row["shots"]): row for row in rows}
    min_shot = min(row_by_shot)
    max_shot = max(row_by_shot)
    min_err = row_by_shot[min_shot]["relative_l2_percent"]
    max_err = row_by_shot[max_shot]["relative_l2_percent"]
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by toy_multiphysics/evaluate.py\n")
        f.write("\\newcommand{\\ToyContextSummary}{"
                f"{min_shot}-shot {min_err:.1f}\\%, "
                f"{max_shot}-shot {max_err:.1f}\\%}}\n")
        f.write(f"\\newcommand{{\\ToyInverseAcc}}{{{100.0 * inverse_acc:.1f}\\%}}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="0,1,2,4",
                        help="Comma-separated demo counts for forward eval.")
    parser.add_argument("--prompts", type=int, default=64,
                        help="Sparse test prompts per shot count.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ckpt_path = ROOT / "checkpoint.pt"
    data_path = ROOT / "test_data.npz"
    if not ckpt_path.exists() or not data_path.exists():
        raise SystemExit("Run train.py first; expected checkpoint.pt and "
                         "test_data.npz in toy_multiphysics/.")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    test = np.load(data_path, allow_pickle=True)
    data = {key: test[key] for key in test.files}
    names = _as_str_list(data.get("regime_names",
                                  np.array(["diffusion", "shear", "vortical"])))

    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if has_mps else "cpu")
    print("device:", device)

    model = ICONLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    sampler = PromptSampler(
        data, ckpt["J"], ckpt["nC"], ckpt["nU"],
        ckpt["c_scale"], ckpt["u_scale"],
        ckpt.get("c_mean", 0.0), ckpt.get("u_mean", 0.0),
        rng_seed=args.seed,
    )

    shot_counts = [int(x) for x in args.shots.split(",") if x.strip()]
    rows = evaluate_context_scaling(
        model, sampler, device, shot_counts, args.prompts, args.batch_size,
        args.seed,
    )
    print("Forward context scaling:")
    for row in rows:
        print(f"  {row['shots']}-shot: rel L2 "
              f"{row['relative_l2_percent']:.2f}% "
              f"(scaled MSE {row['scaled_mse']:.4f})")

    inverse_acc, confusion = evaluate_inverse_regime(data, args.seed)
    print(f"Inverse regime ID accuracy: {100.0 * inverse_acc:.2f}%")
    print("Confusion rows=true, cols=pred:")
    print("  " + " ".join(f"{name:>10s}" for name in names))
    for i, name in enumerate(names):
        vals = " ".join(f"{int(x):10d}" for x in confusion[i])
        print(f"  {name:>10s} {vals}")

    write_context_csv(rows, ROOT / "context_scaling.csv")
    write_confusion_csv(confusion, names, ROOT / "inverse_confusion.csv")
    write_eval_metrics(rows, inverse_acc, ROOT / "eval_metrics.tex")
    print("wrote context_scaling.csv, inverse_confusion.csv, eval_metrics.tex")


if __name__ == "__main__":
    main()
