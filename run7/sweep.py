"""run7 - position-only sweep of every network-free estimator.

207 trajectories - every one that runs 1-6 processed - x 6 sample
counts x 5 noise levels x 8 estimators.
Positions only: no estimator sees an observed velocity (verified by
figures/scripts/check_position_only.py, which replaces the velocity
columns with garbage and checks the answer is bit-identical).

Writes run7/raw.csv incrementally, one block of rows per trajectory, so
the file is always valid and always current. On restart it reads back
which trajectories are already complete and skips them, which makes an
interrupted run cost nothing but a relaunch.

Run from the project root or from run7/; output paths are absolute.
Launch detached, redirect stdout to run7/sweep.log, and follow it with
    Get-Content E:\\cr3bp_inverse_pinn\\run7\\sweep.log -Wait -Tail 20
"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool

# one BLAS thread per worker: with 8 processes the default thread pools
# oversubscribe the CPU and everything gets slower
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train
from baselines import (estimate_mu_spline, estimate_mu_smoothspline,
                       estimate_mu_fourier, estimate_mu_savgol,
                       estimate_mu_tvdiff, estimate_mu_bls, estimate_mu_ekf,
                       estimate_mu_shooting, DEFAULT_MU_BOUNDS)

ROOT = "E:/cr3bp_inverse_pinn"
DATA = f"{ROOT}/cr3bp_dataset_final.npy"
RAW = f"{ROOT}/run7/raw.csv"
N_WORKERS = 8
# every trajectory any previous run actually processed, so run7 joins to
# all of them by traj_index: the 1, 6, 11 ... 996 set of run1/run4, plus
# the seven one-off test cases from run1/run3 (626, 646 and 740 of run2's
# grid are already inside that set or in the seven). run3/prescreen.csv is
# deliberately not counted - it lists all 1000 indices but is a geometry
# table, not a run.
PRIOR_RUN_EXTRAS = (300, 400, 500, 650, 700, 740, 860)
TRAJ_INDICES = tuple(sorted(set(range(1, 1000, 5)) | set(PRIOR_RUN_EXTRAS)))
N_POINTS = (1000, 500, 200, 100, 50, 25)
SIGMAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
PROGRESS_EVERY = 10

# family, function, and which sample counts it runs at. The pinned-state
# shooting is an ablation (it freezes the initial state at the first
# observation), so it only needs the full-sampling column.
METHODS = (
    ("spline",       "gradient_matching",   estimate_mu_spline,       None),
    ("smoothspline", "gradient_matching",   estimate_mu_smoothspline, None),
    ("fourier",      "gradient_matching",   estimate_mu_fourier,      None),
    ("savgol",       "gradient_matching",   estimate_mu_savgol,       None),
    ("tvdiff",       "gradient_matching",   estimate_mu_tvdiff,       None),
    ("bls",          "orbit_determination", estimate_mu_bls,          None),
    ("ekf",          "orbit_determination", estimate_mu_ekf,          None),
    ("shooting",     "ablation",            estimate_mu_shooting,     (1000,)),
)

COLUMNS = [
    # trajectory
    "traj_index", "mu_true", "min_r", "rms_a_true", "max_a_true",
    "tau", "tau_over_dt_full", "mu_sens", "n_rows_available",
    # cell
    "n_points_requested", "n_points_used", "sigma", "noise_seed", "dt",
    # result
    "method", "family", "mu_est", "abs_error", "error_pct",
    "saturated_low", "saturated_high", "elapsed_s",
    # diagnostics
    "contrast", "residual_at_est", "accel_rmse", "vel_rmse", "hyperparam",
    "success", "nfev", "state0_pos_err", "state0_vel_err",
]


def true_derivatives(mu, x, y, vx, vy):
    """Analytic CR3BP acceleration at the given clean states. Used only to
    score the estimators afterwards - no estimator ever sees it."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    ax = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    ay = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return ax, ay


def mu_sensitivity(mu, x, y, vx, vy, h=1e-6):
    """RMS d(acceleration)/d(mu) at the true mu: how strong the mu signal
    is on this trajectory. Central difference on the analytic equations."""
    ax_p, ay_p = true_derivatives(mu + h, x, y, vx, vy)
    ax_m, ay_m = true_derivatives(mu - h, x, y, vx, vy)
    return float(np.sqrt(np.mean(((ax_p - ax_m) / (2 * h)) ** 2
                                 + ((ay_p - ay_m) / (2 * h)) ** 2)))


def describe(mu, x, y, vx, vy, dt_full):
    """Per-trajectory geometry and identifiability descriptors."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    min_r = float(min(r1.min(), r2.min()))
    # encounter timescale at closest approach: sqrt(r^3 / m_nearest)
    m_near = (1 - mu) if r1.min() <= r2.min() else mu
    tau = float(np.sqrt(min_r ** 3 / max(m_near, 1e-300)))
    ax, ay = true_derivatives(mu, x, y, vx, vy)
    a_mag = np.sqrt(ax ** 2 + ay ** 2)
    return dict(min_r=min_r, rms_a_true=float(np.sqrt(np.mean(a_mag ** 2))),
                max_a_true=float(a_mag.max()), tau=tau,
                tau_over_dt_full=tau / dt_full,
                mu_sens=mu_sensitivity(mu, x, y, vx, vy))


def hyperparam_str(name, res):
    if name == "savgol":
        return f"w={res.get('window')},po={res.get('polyorder')}"
    if name == "tvdiff":
        return f"alpha={res.get('alpha')}"
    if name == "smoothspline":
        return f"lam={res.get('lam')}"
    if name == "fourier":
        return f"K={res.get('harmonics')}"
    return ""


def run_trajectory(idx):
    """All cells and all methods for one trajectory. Returns (idx, rows,
    notes, seconds)."""
    t0_all = time.perf_counter()
    data = np.load(DATA, mmap_mode="r")
    t_col = data[:, 6]
    bounds = np.r_[0, np.where(np.diff(t_col) < 0)[0] + 1, len(data)]
    traj = np.asarray(data[bounds[idx]:bounds[idx + 1]])
    t = traj[:, 6:7]
    x, y = traj[:, 0:1], traj[:, 1:2]
    vx, vy = traj[:, 3:4], traj[:, 4:5]
    mu = float(traj[0, 7])
    n_avail = len(t)
    dt_full = float(np.median(np.diff(t.flatten())))

    desc = describe(mu, x.flatten(), y.flatten(), vx.flatten(), vy.flatten(), dt_full)
    lo, hi = DEFAULT_MU_BOUNDS
    rows, notes = [], []
    if n_avail < max(N_POINTS):
        notes.append(f"only {n_avail} samples available")

    for n_req in N_POINTS:
        # clean subsample first, for scoring; then the same indices noisy
        tc, xc, yc, vxc, vyc = train.perturb_trajectory(t, x, y, vx, vy, n_req, 0.0)
        ax_true, ay_true = true_derivatives(mu, xc.flatten(), yc.flatten(),
                                            vxc.flatten(), vyc.flatten())
        a_ref = np.sqrt(np.mean(ax_true ** 2 + ay_true ** 2))
        for si, sigma in enumerate(SIGMAS):
            seed = 1000 * idx + si
            tt, xx, yy, vxx, vyy = train.perturb_trajectory(
                t, x, y, vx, vy, n_req, sigma, seed=seed)
            n_used = len(tt)
            dt = float(np.median(np.diff(tt.flatten())))
            saturated = 0
            for name, family, fn, only_at in METHODS:
                if only_at is not None and n_req not in only_at:
                    continue
                row = dict.fromkeys(COLUMNS, "")
                row.update(desc)
                row.update(traj_index=idx, mu_true=mu, n_rows_available=n_avail,
                           n_points_requested=n_req, n_points_used=n_used,
                           sigma=sigma, noise_seed=seed, dt=dt,
                           method=name, family=family)
                t0 = time.perf_counter()
                try:
                    res = fn(tt, xx, yy, vxx, vyy, use_velocity=False)
                    mu_est = float(res["mu_pred"])
                    row["elapsed_s"] = time.perf_counter() - t0
                    row.update(
                        mu_est=mu_est,
                        abs_error=abs(mu_est - mu),
                        error_pct=100 * abs(mu_est - mu) / mu,
                        saturated_low=int(mu_est <= lo * 1.001),
                        saturated_high=int(mu_est >= hi * 0.999),
                        contrast=res.get("contrast", ""),
                        hyperparam=hyperparam_str(name, res),
                        success=int(res["success"]) if "success" in res else "",
                        nfev=res.get("nfev", ""),
                    )
                    saturated += row["saturated_low"] or row["saturated_high"]
                    if family == "gradient_matching":
                        # how good were this smoother's derivatives? the
                        # paired (derivative error, mu error) that Figure 2
                        # needs, at 150k points instead of 1
                        d = res.get("derivs")
                        if d is not None:
                            row["accel_rmse"] = float(np.sqrt(np.mean(
                                (d[4] - ax_true) ** 2 + (d[5] - ay_true) ** 2)))
                            row["vel_rmse"] = float(np.sqrt(np.mean(
                                (d[2] - vxc.flatten()) ** 2 + (d[3] - vyc.flatten()) ** 2)))
                            row["residual_at_est"] = res.get("residual", "")
                    if name == "bls" and "state0" in res:
                        s0 = res["state0"]
                        row["state0_pos_err"] = float(np.hypot(s0[0] - xc[0, 0],
                                                               s0[1] - yc[0, 0]))
                        row["state0_vel_err"] = float(np.hypot(s0[2] - vxc[0, 0],
                                                               s0[3] - vyc[0, 0]))
                except Exception as exc:
                    row["elapsed_s"] = time.perf_counter() - t0
                    row["method"] = name
                    row["hyperparam"] = f"ERROR:{type(exc).__name__}"
                    notes.append(f"{name} n={n_req} sigma={sigma:g}: "
                                 f"{type(exc).__name__}: {exc}")
                rows.append([row[c] for c in COLUMNS])
            if saturated >= 5:
                notes.append(f"n={n_req} sigma={sigma:g}: {saturated} methods "
                             f"saturated at a bound")
    del a_ref
    return idx, rows, notes, time.perf_counter() - t0_all


def completed_indices():
    """Trajectory indices already fully written to raw.csv."""
    if not os.path.exists(RAW):
        return set()
    seen = {}
    with open(RAW, newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != COLUMNS:
            print("run7: raw.csv header does not match; refusing to append. "
                  "Move it aside to start fresh.", flush=True)
            sys.exit(1)
        for row in r:
            if row:
                seen[row[0]] = seen.get(row[0], 0) + 1
    expected = len(SIGMAS) * (len(N_POINTS) * (len(METHODS) - 1) + 1)
    return {int(k) for k, v in seen.items() if v >= expected}


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    done = completed_indices()
    todo = [i for i in TRAJ_INDICES if i not in done]
    fresh = not os.path.exists(RAW)

    n_cells = len(N_POINTS) * len(SIGMAS)
    print(f"run7: {len(TRAJ_INDICES)} trajectories x {n_cells} cells x {len(METHODS)} "
          f"estimators, positions only, {N_WORKERS} workers", flush=True)
    print(f"run7: {'raw.csv empty, starting fresh' if fresh else f'raw.csv has {len(done)} trajectories already'}"
          f", {len(todo)} to go", flush=True)
    if not todo:
        print("run7: nothing to do", flush=True)
        return

    with open(RAW, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(COLUMNS)
            f.flush()
        start = time.time()
        finished = 0
        with Pool(N_WORKERS) as pool:
            for idx, rows, notes, secs in pool.imap_unordered(run_trajectory, todo):
                w.writerows(rows)
                f.flush()
                finished += 1
                for note in notes:
                    print(f"  traj {idx}: {note}", flush=True)
                if finished % PROGRESS_EVERY == 0 or finished == len(todo):
                    el = time.time() - start
                    rate = finished / el
                    left = (len(todo) - finished) / rate if rate else 0
                    eta = datetime.now() + timedelta(seconds=left)
                    print(f"[{finished:5d}/{len(todo):5d}] {100*finished/len(todo):5.1f}%  "
                          f"elapsed {timedelta(seconds=int(el))}  "
                          f"{rate*60:.1f} traj/min  "
                          f"remaining {timedelta(seconds=int(left))}  "
                          f"finish ~{eta:%H:%M}", flush=True)
        total = time.time() - start
    print(f"run7: done in {timedelta(seconds=int(total))}. "
          f"{len(todo)} trajectories written to raw.csv", flush=True)


if __name__ == "__main__":
    main()
