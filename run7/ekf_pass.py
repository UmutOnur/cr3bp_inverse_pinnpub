"""run7 - fair EKF column, to replace the one in raw.csv.

raw.csv's ekf rows come from the filter's old defaults: measurement_std
1e-6 and state_var0 1e-10 regardless of the cell's noise, and a single
start at mu0 = 0.05. So it was told the data was exact in every noisy
cell (it diverges outright in some, the LinAlgError notes in the log)
and it started in the wrong basin for mu >= 0.15.

Standard orbit determination sets the measurement covariance R from the
known measurement noise, and a non-convex parameter search is seeded
from several starting points. This pass does both: R = sigma^2 I with
the cell's own sigma, and the best of MU0_LADDER by CR3BP residual at
its own answer (truth never consulted). Same trajectories, same cells,
same noise seeds as sweep.py, so the output joins to raw.csv on
(traj_index, n_points_requested, sigma).

Run after sweep.py finishes - it is cheap (~1 min per trajectory across
all 30 cells) but there is no point competing with the sweep for cores.
Writes run7/ekf_fair.csv incrementally and skips trajectories already
in it, so an interrupted run costs only a relaunch.
"""
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train
from baselines import estimate_mu_ekf, MU0_LADDER

ROOT = "E:/cr3bp_inverse_pinn"
DATA = f"{ROOT}/cr3bp_dataset_final.npy"
RAW = f"{ROOT}/run7/raw.csv"
OUT = f"{ROOT}/run7/ekf_fair.csv"
N_WORKERS = 8
N_POINTS = (1000, 500, 200, 100, 50, 25)
SIGMAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)
PROGRESS_EVERY = 10

COLUMNS = ["traj_index", "mu_true", "n_points_requested", "n_points_used",
           "sigma", "noise_seed", "method", "family", "mu_est", "abs_error",
           "error_pct", "mu0_best", "measurement_std", "diverged",
           "steps_used", "n_steps", "elapsed_s", "note"]


def run_trajectory(idx):
    t0_all = time.perf_counter()
    t, x, y, vx, vy, mu = train.load_trajectory(DATA, idx)
    rows, notes = [], []
    for n_req in N_POINTS:
        for si, sigma in enumerate(SIGMAS):
            seed = 1000 * idx + si
            tt, xx, yy, vxx, vyy = train.perturb_trajectory(
                t, x, y, vx, vy, n_req, sigma, seed=seed)
            row = dict.fromkeys(COLUMNS, "")
            row.update(traj_index=idx, mu_true=mu, n_points_requested=n_req,
                       n_points_used=len(tt), sigma=sigma, noise_seed=seed,
                       method="ekf", family="orbit_determination",
                       measurement_std=max(sigma, 1e-8))
            t0 = time.perf_counter()
            try:
                res = estimate_mu_ekf(tt, xx, yy, vxx, vyy,
                                      use_velocity=False, sigma=sigma)
                mu_est = float(res["mu_pred"])
                row.update(mu_est=mu_est, mu0_best=res.get("mu0_best", ""),
                           abs_error=abs(mu_est - mu),
                           error_pct=100 * abs(mu_est - mu) / mu,
                           diverged=int(res.get("diverged", 0)),
                           steps_used=res.get("steps_used", ""),
                           n_steps=res.get("n_steps", ""))
                if not np.isfinite(mu_est):
                    notes.append(f"n={n_req} sigma={sigma:g}: every mu0 diverged")
                    row["note"] = "all_starts_failed"
            except Exception as exc:
                row["note"] = f"ERROR:{type(exc).__name__}"
                notes.append(f"n={n_req} sigma={sigma:g}: "
                             f"{type(exc).__name__}: {exc}")
            row["elapsed_s"] = time.perf_counter() - t0
            rows.append([row[c] for c in COLUMNS])
    return idx, rows, notes, time.perf_counter() - t0_all


def indices_to_do():
    """Every trajectory present in raw.csv, so the bonus trajectories from
    the first all-1000 launch get a fair EKF column too."""
    if not os.path.exists(RAW):
        print("run7: raw.csv not found; nothing to match", flush=True)
        sys.exit(1)
    with open(RAW, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        present = {int(row[0]) for row in r if row}
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header != COLUMNS:
                print("run7: ekf_fair.csv header does not match; refusing to "
                      "append. Move it aside to start fresh.", flush=True)
                sys.exit(1)
            seen = {}
            for row in r:
                if row:
                    seen[int(row[0])] = seen.get(int(row[0]), 0) + 1
            expected = len(N_POINTS) * len(SIGMAS)
            done = {k for k, v in seen.items() if v >= expected}
    return sorted(present - done), len(present), len(done)


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    todo, n_present, n_done = indices_to_do()
    fresh = not os.path.exists(OUT)
    print(f"run7 ekf pass: {n_present} trajectories in raw.csv, {n_done} "
          f"already done here, {len(todo)} to go", flush=True)
    print(f"run7 ekf pass: positions only, {len(N_POINTS) * len(SIGMAS)} cells "
          f"x {len(MU0_LADDER)} starting guesses, {N_WORKERS} workers",
          flush=True)
    if not todo:
        print("run7 ekf pass: nothing to do", flush=True)
        return

    with open(OUT, "a", newline="") as f:
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
                    print(f"[{finished:5d}/{len(todo):5d}] "
                          f"{100 * finished / len(todo):5.1f}%  "
                          f"elapsed {timedelta(seconds=int(el))}  "
                          f"{rate * 60:.1f} traj/min  "
                          f"finish ~{eta:%H:%M}", flush=True)
        total = time.time() - start
    print(f"run7 ekf pass: done in {timedelta(seconds=int(total))}, "
          f"{len(todo)} trajectories written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
