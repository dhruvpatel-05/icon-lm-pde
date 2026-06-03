"""
Toy multi-regime 2D fluid-tracer data generator.

Each hidden operator maps an initial dyed scalar field q(0, x, y) to several
future snapshots q(t_1), ..., q(t_K) on the periodic square.  The toy PDE is

    q_t + u . grad(q) = kappa Delta q,      div(u) = 0,

where the incompressible velocity u is generated from a hidden stream function.
This is a lightweight fluid-prediction task with three hidden regimes:
diffusion-dominated transport, shear advection, and vortical stirring. The
ICON-LM model must infer the active regime/operator from a few demonstration
trajectories.
"""

from __future__ import annotations

import numpy as np


REGIME_NAMES = np.array(["diffusion", "shear", "vortical"])


def _grid(N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = (np.arange(N) + 0.5) / N
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return xs.astype(np.float32), X, Y


def _lowpass_filter(N: int, cutoff: float) -> np.ndarray:
    m = np.fft.fftfreq(N, d=1.0 / N)
    MX, MY = np.meshgrid(m, m, indexing="ij")
    r = np.sqrt(MX**2 + MY**2)
    filt = np.exp(-((r / cutoff) ** 4))
    filt[0, 0] = 0.0
    return filt


def make_initial_fields(num_fields: int, N: int, rng: np.random.Generator,
                        cutoff: float = 10.0) -> np.ndarray:
    """Sample high-contrast dye fields with a vertical front and fine texture."""
    _, X, Y = _grid(N)
    out = np.zeros((num_fields, N, N), dtype=np.float32)
    filt = _lowpass_filter(N, cutoff)

    for i in range(num_fields):
        phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        front = 0.70 + 0.07 * np.sin(2.0 * np.pi * Y + phase1)
        front += 0.035 * np.sin(4.0 * np.pi * Y + phase2)
        base = 0.25 + 1.45 / (1.0 + np.exp(-(X - front) / 0.055))
        base += 0.10 * np.sin(2.0 * np.pi * (X + 0.35 * Y) + phase1)

        noise = rng.normal(size=(N, N))
        texture = np.real(np.fft.ifft2(np.fft.fft2(noise) * filt))
        texture /= texture.std() + 1e-6
        q = base + 0.10 * texture

        for _ in range(4):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            sigma = float(rng.uniform(0.045, 0.10))
            amp = float(rng.normal(0.0, 0.14))
            dx = np.minimum(np.abs(X - cx), 1.0 - np.abs(X - cx))
            dy = np.minimum(np.abs(Y - cy), 1.0 - np.abs(Y - cy))
            q += amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

        out[i] = np.clip(q, 0.12, 1.90).astype(np.float32)
    return out


def sample_operator(rng: np.random.Generator) -> tuple[int, float, float]:
    """Draw one hidden physics regime: regime, diffusivity, velocity scale."""
    regime = int(rng.integers(0, len(REGIME_NAMES)))
    if regime == 0:
        # Diffusion-dominated: almost no transport, stronger smoothing.
        kappa = float(rng.uniform(7.0e-4, 1.5e-3))
        speed = 0.0
    elif regime == 1:
        # Shear advection: coherent transport with weak diffusion.
        kappa = float(rng.uniform(5.0e-5, 1.5e-4))
        speed = float(rng.uniform(0.35, 0.55))
    else:
        # Vortical stirring: spatially complex incompressible flow.
        kappa = float(rng.uniform(1.2e-4, 3.5e-4))
        speed = float(rng.uniform(0.40, 0.70))
    return regime, kappa, speed


def make_velocity_field(N: int, rng: np.random.Generator,
                        speed: float, regime: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a smooth incompressible velocity for the selected regime."""
    _, X, Y = _grid(N)

    if regime == 0:
        z = np.zeros((N, N), dtype=np.float32)
        return z, z

    psi = np.zeros((N, N), dtype=np.float64)

    if regime == 1:
        phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        psi += np.sin(2.0 * np.pi * Y + phase1)
        psi += 0.35 * np.sin(4.0 * np.pi * Y + phase2)
        psi += 0.10 * np.sin(2.0 * np.pi * X + phase1)
    else:
        for _ in range(10):
            kx = int(rng.integers(1, 5))
            ky = int(rng.integers(1, 5))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            amp = float(rng.normal(0.0, 1.0) / (kx * kx + ky * ky))
            psi += amp * np.sin(2.0 * np.pi * (kx * X + ky * Y) + phase)

        for _ in range(5):
            cx, cy = rng.uniform(0.0, 1.0, size=2)
            sigma = float(rng.uniform(0.06, 0.14))
            amp = float(rng.normal(0.0, 0.40))
            dx = np.minimum(np.abs(X - cx), 1.0 - np.abs(X - cx))
            dy = np.minimum(np.abs(Y - cy), 1.0 - np.abs(Y - cy))
            psi += amp * np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))

    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    psi_hat = np.fft.fft2(psi)
    ux = np.real(np.fft.ifft2(1j * KY * psi_hat))
    uy = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    rms = np.sqrt(np.mean(ux**2 + uy**2)) + 1e-8
    ux = speed * ux / rms
    uy = speed * uy / rms
    return ux.astype(np.float32), uy.astype(np.float32)


def _bilinear_periodic(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a 2D field or batch of fields at periodic coordinates in [0, 1)."""
    N = field.shape[-1]
    xf = (x % 1.0) * N - 0.5
    yf = (y % 1.0) * N - 0.5
    i0 = np.floor(xf).astype(np.int64) % N
    j0 = np.floor(yf).astype(np.int64) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    wx = xf - np.floor(xf)
    wy = yf - np.floor(yf)

    f00 = field[..., i0, j0]
    f10 = field[..., i1, j0]
    f01 = field[..., i0, j1]
    f11 = field[..., i1, j1]
    return ((1.0 - wx) * (1.0 - wy) * f00
            + wx * (1.0 - wy) * f10
            + (1.0 - wx) * wy * f01
            + wx * wy * f11)


def _backtrace_map(ux: np.ndarray, uy: np.ndarray, t: float,
                   substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Backtrace grid points under a steady velocity field with RK2 steps."""
    _, X, Y = _grid(ux.shape[0])
    xb = X.astype(np.float64).copy()
    yb = Y.astype(np.float64).copy()
    dt = t / substeps
    for _ in range(substeps):
        u1 = _bilinear_periodic(ux, xb, yb)
        v1 = _bilinear_periodic(uy, xb, yb)
        xm = xb - 0.5 * dt * u1
        ym = yb - 0.5 * dt * v1
        u2 = _bilinear_periodic(ux, xm, ym)
        v2 = _bilinear_periodic(uy, xm, ym)
        xb = (xb - dt * u2) % 1.0
        yb = (yb - dt * v2) % 1.0
    return xb, yb


def _diffuse(fields: np.ndarray, kappa: float, t: float) -> np.ndarray:
    N = fields.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    prop = np.exp(-kappa * (KX**2 + KY**2) * t)
    return np.real(np.fft.ifft2(np.fft.fft2(fields, axes=(-2, -1)) * prop,
                                axes=(-2, -1)))


def evolve_tracer(q0: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                  kappa: float, times: np.ndarray) -> np.ndarray:
    """Semi-Lagrangian advection plus spectral diffusion for a batch of fields."""
    frames = []
    for t in times:
        substeps = max(8, int(np.ceil(float(t) / 0.025)))
        xb, yb = _backtrace_map(ux, uy, float(t), substeps)
        advected = _bilinear_periodic(q0, xb, yb)
        smoothed = _diffuse(advected, kappa, float(t))
        frames.append(np.clip(smoothed, 0.10, 1.95).astype(np.float32))
    return np.stack(frames, axis=1)


def make_dataset(num_operators: int, sources_per_operator: int, N: int,
                 seed: int = 0, num_frames: int = 4,
                 steps_per_frame: int = 8, dt: float = 0.02) -> dict:
    """Generate the in-context swirly fluid-tracer dataset."""
    rng = np.random.default_rng(seed)
    times = (np.arange(1, num_frames + 1) * steps_per_frame * dt).astype(np.float32)
    c_all = np.zeros((num_operators, sources_per_operator, N, N), dtype=np.float32)
    u_all = np.zeros((num_operators, sources_per_operator, num_frames, N, N),
                     dtype=np.float32)
    regime_all = np.zeros(num_operators, dtype=np.int64)
    kappa_all = np.zeros(num_operators, dtype=np.float32)
    speed_all = np.zeros(num_operators, dtype=np.float32)

    for i in range(num_operators):
        regime, kappa, speed = sample_operator(rng)
        ux, uy = make_velocity_field(N, rng, speed, regime)
        q0 = make_initial_fields(sources_per_operator, N, rng)
        traj = evolve_tracer(q0, ux, uy, kappa, times)
        c_all[i] = q0
        u_all[i] = traj
        regime_all[i] = regime
        kappa_all[i] = kappa
        speed_all[i] = speed

    grid = ((np.arange(N) + 0.5) / N).astype(np.float32)
    return dict(c=c_all, u=u_all, regime=regime_all, nu=kappa_all,
                speed=speed_all, times=times, grid=grid,
                regime_names=REGIME_NAMES,
                steps_per_frame=np.int64(steps_per_frame), dt=np.float32(dt))


if __name__ == "__main__":
    import os

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__),
                                                       ".matplotlib"))
    import matplotlib.pyplot as plt

    data = make_dataset(num_operators=2, sources_per_operator=2, N=64, seed=42)
    print("q0", data["c"].shape, "trajectory", data["u"].shape)
    print("regime", data["regime"], REGIME_NAMES[data["regime"]])
    print("kappa", data["nu"], "speed", data["speed"], "times", data["times"])

    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for row in range(2):
        fields = [data["c"][row, 0], *data["u"][row, 0]]
        for col, field in enumerate(fields):
            ax = axes[row, col]
            ax.imshow(field, cmap="RdBu_r", origin="lower", vmin=0.15, vmax=1.85,
                      interpolation="bicubic")
            ax.set_title("t=0" if col == 0 else f"t={data['times'][col-1]:.2f}",
                         fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("sanity.png", dpi=180)
    print("wrote sanity.png")
