"""
1D viscous Burgers equation data generator for ICON-LM.

Hidden operator: viscosity ν ∈ [ν_min, ν_max] (log-uniform).
Each operator maps an initial condition u(x, 0) to a trajectory
u(x, t_1 .. t_K) on the periodic domain [0, 2π].

PDE:  u_t + u·u_x = ν·u_xx
Solver: spectral (FFT) RHS with SciPy RK45.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# ── grid / time defaults ───────────────────────────────────────────────────────
N_GRID     = 64
T_END      = 1.0
NUM_FRAMES = 4
NU_MIN     = 0.01
NU_MAX     = 0.10


# ── solver ─────────────────────────────────────────────────────────────────────

def solve_burgers(nu: float, u0: np.ndarray, x: np.ndarray,
                  t_eval: np.ndarray) -> np.ndarray:
    """
    Solve  u_t + u·u_x = ν·u_xx  on [0, 2π] periodic via spectral RHS + RK45.

    Parameters
    ----------
    nu     : viscosity (float > 0)
    u0     : initial condition, shape [N]
    x      : uniform grid in [0, 2π), shape [N]
    t_eval : output times (excluding t=0), shape [K]

    Returns
    -------
    [K, N] float32 array of solutions at t_eval.
    """
    N = len(x)
    k = np.fft.rfftfreq(N, d=1.0 / N)   # wavenumbers for [0, 2π] with N pts

    def rhs(t, u):
        u_hat = np.fft.rfft(u)
        ux    = np.fft.irfft(1j * k * u_hat, n=N)
        uxx   = np.fft.irfft(-(k ** 2) * u_hat, n=N)
        return -u * ux + nu * uxx

    sol = solve_ivp(rhs, [0.0, float(t_eval[-1])],
                    u0.astype(np.float64),
                    t_eval=t_eval.astype(np.float64),
                    method='RK45', rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed for nu={nu}: {sol.message}")
    return sol.y.T.astype(np.float32)   # [K, N]


def random_ic(x: np.ndarray, rng: np.random.Generator,
              n_modes: int = 5) -> np.ndarray:
    """Random superposition of Fourier modes as initial condition."""
    u = np.zeros(len(x), dtype=np.float64)
    for m in range(1, n_modes + 1):
        a   = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        u  += a * np.sin(m * x + phi)
    return (u / n_modes).astype(np.float32)


# ── caption text pools ─────────────────────────────────────────────────────────

_PRECISE = [
    "Viscous Burgers equation: u_t + u*u_x = nu*u_xx, nu={nu:.5f}, periodic.",
    "Burgers PDE with viscosity nu={nu:.5f}: u_t + u*u_x = nu*u_xx.",
    "1D Burgers on [0,2pi] periodic: u_t + u*u_x = {nu:.5f}*u_xx.",
    "Nonlinear advection-diffusion, Burgers form, viscosity nu={nu:.5f}.",
    "u_t + u*u_x = nu*u_xx with nu={nu:.5f} on a periodic domain.",
]

# High ν (> 0.04): diffusion-dominated, smooth solutions.
_VAGUE_HIGH = [
    "A nonlinear wave equation with strong viscous damping.",
    "A Burgers-type PDE in the diffusion-dominated regime.",
    "A strongly smoothed nonlinear advection equation.",
    "Burgers equation with high viscosity and smooth solutions.",
    "A nonlinear PDE where diffusion dominates over advection.",
]

# Low ν (≤ 0.04): advection-dominated, sharp gradients / near-shock.
_VAGUE_LOW = [
    "A nonlinear wave equation with weak viscosity and sharp gradients.",
    "A Burgers-type PDE near the inviscid limit with shock formation.",
    "A convection-dominated Burgers equation with thin shock layers.",
    "Burgers equation at low viscosity with near-discontinuous solutions.",
    "A nonlinear hyperbolic-like PDE with mild viscous regularization.",
]


def make_caption(nu: float, precise: bool, rng: np.random.Generator) -> str:
    """Return one caption string for a Burgers operator with viscosity nu."""
    if precise:
        tmpl = _PRECISE[int(rng.integers(len(_PRECISE)))]
        return tmpl.format(nu=nu)
    pool = _VAGUE_HIGH if nu > 0.04 else _VAGUE_LOW
    return pool[int(rng.integers(len(pool)))]


# ── dataset generation ─────────────────────────────────────────────────────────

def make_dataset(nu_values: np.ndarray, sources_per_op: int,
                 N: int = N_GRID, seed: int = 0,
                 num_frames: int = NUM_FRAMES,
                 T: float = T_END) -> dict:
    """
    Generate a Burgers trajectory dataset.

    Returns a dict with keys:
      c     [O, S, N]      initial conditions
      u     [O, S, K, N]   solutions at K future times
      nu    [O]            viscosity values (the hidden operator parameter)
      grid  [N]            spatial grid in [0, 2π)
      times [K]            evaluation times
    """
    rng    = np.random.default_rng(seed)
    x      = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).astype(np.float32)
    t_eval = np.linspace(T / num_frames, T, num_frames).astype(np.float32)
    O      = len(nu_values)

    c_all = np.zeros((O, sources_per_op, N),             dtype=np.float32)
    u_all = np.zeros((O, sources_per_op, num_frames, N), dtype=np.float32)

    for i, nu in enumerate(nu_values):
        for s in range(sources_per_op):
            u0       = random_ic(x, rng)
            traj     = solve_burgers(float(nu), u0, x, t_eval)
            c_all[i, s] = u0
            u_all[i, s] = traj
        if (i + 1) % 40 == 0 or i == 0:
            print(f"  operator {i+1:3d}/{O}  nu={nu:.5f}")

    return dict(c=c_all, u=u_all,
                nu=nu_values.astype(np.float32),
                grid=x, times=t_eval)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng  = np.random.default_rng(0)
    x    = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    tevl = np.linspace(T_END / NUM_FRAMES, T_END, NUM_FRAMES)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, nu in zip(axes, [0.01, 0.04, 0.10]):
        u0   = random_ic(x, rng)
        traj = solve_burgers(nu, u0, x, tevl)
        ax.plot(x, u0, 'k--', lw=1, label='t=0')
        for ki, t in enumerate(tevl):
            ax.plot(x, traj[ki], label=f't={t:.2f}')
        ax.set_title(f'ν={nu}')
        ax.set_xlabel('x')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('burgers_sanity.png', dpi=120)
    print("wrote burgers_sanity.png")

    for precise in (True, False):
        print(make_caption(0.012, precise, rng))
        print(make_caption(0.08,  precise, rng))
