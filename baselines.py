"""
Non-PINN baselines for CR3BP mu estimation: gradient matching (grid
scan), single shooting + nonlinear least squares, and an extended
Kalman filter. All three take the same (t, x, y, vx, vy) arrays used in
train.py and return a mu estimate.

classical_scan()        -- differentiates the data and scans a mu grid
                            against the algebraic CR3BP equations.
estimate_mu_shooting()  -- integrates forward from the known initial
                            condition and fits mu with least_squares.
estimate_mu_ekf()       -- extended Kalman filter with mu as an
                            augmented constant state.

Caveat: the dataset and estimate_mu_shooting/estimate_mu_ekf both use
scipy.integrate.solve_ivp against the same CR3BP equations, which is an
inverse crime and flatters those two methods (not classical_scan).
"""

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares

import cr3bp

DEFAULT_MU_BOUNDS = (1e-7, 0.5)
INTEGRATOR_TOL = dict(atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Gradient matching (grid scan)
# ---------------------------------------------------------------------------
def classical_scan(t, x, y, vx, vy, n_points=300):
    """Network-free baseline: scan mu against finite-difference accelerations."""
    t_f, x_f, y_f = t.flatten(), x.flatten(), y.flatten()
    vx_f, vy_f = vx.flatten(), vy.flatten()
    ax = np.gradient(vx_f, t_f)
    ay = np.gradient(vy_f, t_f)

    mu_grid = np.linspace(0.001, 0.499, n_points)
    losses = []
    for m in mu_grid:
        r1 = np.sqrt((x_f + m) ** 2 + y_f ** 2)
        r2 = np.sqrt((x_f - (1 - m)) ** 2 + y_f ** 2)
        pred_ax = x_f - (1 - m) * (x_f + m) / r1 ** 3 - m * (x_f - (1 - m)) / r2 ** 3 + 2 * vy_f
        pred_ay = y_f - (1 - m) * y_f / r1 ** 3 - m * y_f / r2 ** 3 - 2 * vx_f
        losses.append(np.mean((pred_ax - ax) ** 2) + np.mean((pred_ay - ay) ** 2))
    losses = np.array(losses)
    return float(mu_grid[losses.argmin()]), float(losses.max() / losses.min())


# ---------------------------------------------------------------------------
# Shooting + nonlinear least squares
# ---------------------------------------------------------------------------
def _shoot(mu, t, ic6, **tol):
    """Integrate the CR3BP EOMs for a scalar mu, returning the solve_ivp result."""
    eoms = cr3bp.EOMConstructor(mu)
    return solve_ivp(eoms, (t[0], t[-1]), ic6, t_eval=t, **tol)


def _shooting_residual(log_mu, t, ic6, x, y, vx, vy, use_velocity, tol):
    mu = 10.0 ** log_mu[0]
    sol = _shoot(mu, t, ic6, **tol)
    n = len(t)
    if not sol.success or sol.y.shape[1] != n:
        # Penalize failed/truncated integrations (e.g. a candidate mu that
        # sends the third body into a primary) without crashing the optimizer.
        fill = 1e3
        return np.full(4 * n if use_velocity else 2 * n, fill)
    xp, yp, vxp, vyp = sol.y[0], sol.y[1], sol.y[3], sol.y[4]
    if use_velocity:
        return np.concatenate([xp - x, yp - y, vxp - vx, vyp - vy])
    return np.concatenate([xp - x, yp - y])


def estimate_mu_shooting(t, x, y, vx, vy, use_velocity=True,
                          mu_bounds=DEFAULT_MU_BOUNDS, n_coarse=25,
                          tol=None, verbose=False):
    """
    Recover mu by single shooting from the trajectory's own initial
    condition, refined with scipy.optimize.least_squares.

    Parameters
    ----------
    t, x, y, vx, vy : array-like
        Observed trajectory, same layout as train.load_trajectory.
    use_velocity : bool
        If True (default) the residual includes vx, vy as well as x, y.
        Set False for a position-only fit.
    mu_bounds : (lo, hi)
        Search range for mu, worked in log10(mu) space.
    n_coarse : int
        Log-spaced points used only to seed the optimizer's starting
        point; final precision comes from least_squares, not this grid.
    tol : dict or None
        Overrides for solve_ivp's atol/rtol (default 1e-12/1e-12).

    Returns
    -------
    dict with mu_pred, success, cost, nfev, coarse_log_mu0,
    coarse_costs, elapsed_s.
    """
    tol = tol or INTEGRATOR_TOL
    t = np.asarray(t, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    vx = np.asarray(vx, dtype=float).flatten()
    vy = np.asarray(vy, dtype=float).flatten()
    ic6 = np.array([x[0], y[0], 0.0, vx[0], vy[0], 0.0])

    t_start = time.time()
    log_lo, log_hi = np.log10(mu_bounds[0]), np.log10(mu_bounds[1])
    coarse_grid = np.linspace(log_lo, log_hi, n_coarse)
    coarse_costs = np.array([
        np.sum(_shooting_residual([lm], t, ic6, x, y, vx, vy, use_velocity, tol) ** 2)
        for lm in coarse_grid
    ])
    log_mu0 = float(coarse_grid[np.argmin(coarse_costs)])

    result = least_squares(
        _shooting_residual, x0=[log_mu0],
        args=(t, ic6, x, y, vx, vy, use_velocity, tol),
        bounds=([log_lo], [log_hi]), method="trf",
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    mu_pred = float(10.0 ** result.x[0])
    elapsed = time.time() - t_start

    if verbose:
        print(f"  shooting: coarse mu0={10**log_mu0:.6g} -> refined mu={mu_pred:.6g} "
              f"cost={result.cost:.3e} nfev={result.nfev} success={result.success} "
              f"({elapsed:.2f}s)")

    return {
        "mu_pred": mu_pred,
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "coarse_log_mu0": log_mu0,
        "coarse_costs": (coarse_grid, coarse_costs),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Extended Kalman filter, mu as an augmented constant parameter
# ---------------------------------------------------------------------------
def _augmented_dynamics(t, s):
    """CR3BP EOMs on state [x, y, vx, vy, mu], with d(mu)/dt = 0."""
    x, y, vx, vy, mu = s
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    ax = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    ay = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return np.array([vx, vy, ax, ay, 0.0])


def _augmented_jacobian(s, eps=1e-6):
    """Finite-difference Jacobian of _augmented_dynamics at state s (5x5)."""
    n = len(s)
    f0 = _augmented_dynamics(0.0, s)
    J = np.zeros((n, n))
    for i in range(n):
        step = eps * max(1.0, abs(s[i]))
        s_pert = s.copy()
        s_pert[i] += step
        J[:, i] = (_augmented_dynamics(0.0, s_pert) - f0) / step
    return J


def estimate_mu_ekf(t, x, y, vx, vy, mu0=0.05, use_velocity=True,
                     state_var0=1e-10, mu_var0=0.25 ** 2,
                     measurement_std=1e-6, process_var_state=1e-14,
                     process_var_mu=0.0, mu_bounds=DEFAULT_MU_BOUNDS,
                     verbose=False):
    """
    Recover mu with an extended Kalman filter that treats it as a
    constant, unobserved state variable alongside (x, y, vx, vy).

    Processes the trajectory once, sequentially: predicts the next state
    with the nonlinear CR3BP dynamics, propagates covariance by
    linearizing around the current estimate, then corrects against the
    next observed sample. No grid, no global search.

    Parameters mirror estimate_mu_shooting() where they overlap. Additional:
        mu0             -- initial mu guess
        state_var0      -- initial variance on x, y, vx, vy
        mu_var0         -- initial variance on mu
        measurement_std -- assumed observation noise std
        process_var_state / process_var_mu -- process noise per step

    Returns
    -------
    dict with mu_pred, mu_history, P_final, elapsed_s.
    """
    t = np.asarray(t, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    vx = np.asarray(vx, dtype=float).flatten()
    vy = np.asarray(vy, dtype=float).flatten()
    n = len(t)

    t_start = time.time()
    s = np.array([x[0], y[0], vx[0], vy[0], mu0])
    P = np.diag([state_var0] * 4 + [mu_var0])
    Q = np.diag([process_var_state] * 4 + [process_var_mu])

    dim_z = 4 if use_velocity else 2
    H = np.eye(5)[:dim_z]
    R = np.eye(dim_z) * measurement_std ** 2
    I5 = np.eye(5)

    mu_history = np.empty(n)
    mu_history[0] = s[4]

    for k in range(1, n):
        dt = t[k] - t[k - 1]

        sol = solve_ivp(_augmented_dynamics, (t[k - 1], t[k]), s,
                         t_eval=[t[k]], **INTEGRATOR_TOL)
        s_pred = sol.y[:, -1]

        F = _augmented_jacobian(s)
        Phi = expm(F * dt)
        P_pred = Phi @ P @ Phi.T + Q

        z = np.array([x[k], y[k], vx[k], vy[k]]) if use_velocity else np.array([x[k], y[k]])
        innovation = z - H @ s_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        s = s_pred + K @ innovation
        s[4] = np.clip(s[4], mu_bounds[0], mu_bounds[1])
        P = (I5 - K @ H) @ P_pred

        mu_history[k] = s[4]

    elapsed = time.time() - t_start
    if verbose:
        print(f"  ekf: mu0={mu0:.6g} -> mu={s[4]:.6g} ({elapsed:.2f}s)")

    return {
        "mu_pred": float(s[4]),
        "mu_history": mu_history,
        "P_final": P,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Self-test on four representative trajectories
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from train import load_trajectory, min_primary_distance

    test_indices = [646, 821, 626, 740]  # easy, clean far, close approach x2

    print(f"{'traj':>6} {'mu_true':>10} {'min_r':>8} | "
          f"{'classical%':>11} | {'shoot_mu':>10} {'shoot%':>9} {'shoot_s':>8} | "
          f"{'ekf_mu':>10} {'ekf%':>9} {'ekf_s':>8}")

    for idx in test_indices:
        t, x, y, vx, vy, mu_true = load_trajectory("cr3bp_dataset_final.npy", idx)
        min_r = min_primary_distance(x, y, mu_true)

        cls_mu, _ = classical_scan(t, x, y, vx, vy)
        cls_err = abs(cls_mu - mu_true) / mu_true * 100

        shoot = estimate_mu_shooting(t, x, y, vx, vy)
        shoot_err = abs(shoot["mu_pred"] - mu_true) / mu_true * 100

        ekf = estimate_mu_ekf(t, x, y, vx, vy)
        ekf_err = abs(ekf["mu_pred"] - mu_true) / mu_true * 100

        print(f"{idx:>6} {mu_true:>10.6f} {min_r:>8.4f} | "
              f"{cls_err:>10.2f}% | "
              f"{shoot['mu_pred']:>10.6f} {shoot_err:>8.2f}% {shoot['elapsed_s']:>7.2f}s | "
              f"{ekf['mu_pred']:>10.6f} {ekf_err:>8.2f}% {ekf['elapsed_s']:>7.2f}s")
