"""Is the inverse crime inflating bls / ekf / shooting?

The dataset was generated with scipy's RK45. baselines.py inverts it with
RK45 too, so the three methods that integrate the equations are solving
with the very discretisation that produced the data - an inverse crime
that flatters them. The project notes call the switch load-bearing, because
those methods are the honest ceiling the PINN is compared against.

Rather than re-run the whole 30-cell grid with DOP853 (~20 h), this runs
a spot-check: 40 trajectories stratified by closest approach (close
approaches are where the crime should help most, since that is where the
dynamics are stiff), two representative cells, all three integrating
methods, each with RK45 and DOP853 on identical data. If the answers
barely move, one sentence in the paper closes the caveat.

Writes run7/dop853_check.csv incrementally; skips trajectories already
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
import baselines
from baselines import estimate_mu_bls, estimate_mu_ekf, estimate_mu_shooting

ROOT = "E:/cr3bp_inverse_pinn"
DATA = f"{ROOT}/cr3bp_dataset_final.npy"
RAW = f"{ROOT}/run7/raw.csv"
OUT = f"{ROOT}/run7/dop853_check.csv"
N_WORKERS = 8
N_CLOSE = 20          # trajectories with min_r < 0.05
N_FAR = 20            # trajectories with min_r >= 0.05
CELLS = ((1000, 0.0), (1000, 1e-4))
METHODS = (("bls", estimate_mu_bls), ("ekf", estimate_mu_ekf),
           ("shooting", estimate_mu_shooting))
INTEGRATORS = ("RK45", "DOP853")
SIGMA_INDEX = {0.0: 0, 1e-6: 1, 1e-5: 2, 1e-4: 3, 1e-3: 4}

COLUMNS = ["traj_index", "mu_true", "min_r", "n_points", "sigma", "noise_seed",
           "method", "integrator", "mu_est", "abs_error", "error_pct",
           "elapsed_s", "note"]


def run_trajectory(idx):
    t0_all = time.perf_counter()
    t, x, y, vx, vy, mu = train.load_trajectory(DATA, idx)
    min_r = train.min_primary_distance(x, y, mu)
    rows, notes = [], []
    for n, sigma in CELLS:
        seed = 1000 * idx + SIGMA_INDEX[sigma]
        tt, xx, yy, vxx, vyy = train.perturb_trajectory(
            t, x, y, vx, vy, n, sigma, seed=seed)
        for integrator in INTEGRATORS:
            # rebinding the module global is what switches the integrator
            baselines.INTEGRATOR_TOL = dict(atol=1e-12, rtol=1e-12,
                                            method=integrator)
            for name, fn in METHODS:
                row = dict.fromkeys(COLUMNS, "")
                row.update(traj_index=idx, mu_true=mu, min_r=min_r,
                           n_points=len(tt), sigma=sigma, noise_seed=seed,
                           method=name, integrator=integrator)
                t0 = time.perf_counter()
                try:
                    kw = dict(use_velocity=False)
                    if name == "ekf":
                        kw["sigma"] = sigma
                    res = fn(tt, xx, yy, vxx, vyy, **kw)
                    m = float(res["mu_pred"])
                    row.update(mu_est=m, abs_error=abs(m - mu),
                               error_pct=100 * abs(m - mu) / mu)
                except Exception as exc:
                    row["note"] = f"ERROR:{type(exc).__name__}"
                    notes.append(f"{name}/{integrator} n={n} sigma={sigma:g}: "
                                 f"{type(exc).__name__}: {exc}")
                row["elapsed_s"] = time.perf_counter() - t0
                rows.append([row[c] for c in COLUMNS])
    baselines.INTEGRATOR_TOL = dict(atol=1e-12, rtol=1e-12, method="RK45")
    return idx, rows, notes, time.perf_counter() - t0_all


def pick_trajectories():
    """Stratify by closest approach, using run7's own min_r column."""
    seen = {}
    with open(RAW, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            i = int(row["traj_index"])
            if i not in seen:
                seen[i] = float(row["min_r"])
    close = sorted([i for i, v in seen.items() if v < 0.05])
    far = sorted([i for i, v in seen.items() if v >= 0.05])
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(close, min(N_CLOSE, len(close)), replace=False).tolist()
                  + rng.choice(far, min(N_FAR, len(far)), replace=False).tolist())
    done = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            r = csv.reader(f)
            if next(r, None) != COLUMNS:
                print("run7: dop853_check.csv header mismatch; move it aside.",
                      flush=True)
                sys.exit(1)
            counts = {}
            for row in r:
                if row:
                    counts[int(row[0])] = counts.get(int(row[0]), 0) + 1
            expected = len(CELLS) * len(INTEGRATORS) * len(METHODS)
            done = {k for k, v in counts.items() if v >= expected}
    return [i for i in pick if i not in done], len(pick), len(done)


def main():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

    todo, n_pick, n_done = pick_trajectories()
    fresh = not os.path.exists(OUT)
    print(f"run7 dop853: {n_pick} trajectories picked "
          f"({N_CLOSE} close + {N_FAR} far), {n_done} already done, "
          f"{len(todo)} to go", flush=True)
    print(f"run7 dop853: {len(CELLS)} cells x {len(INTEGRATORS)} integrators "
          f"x {len(METHODS)} methods, positions only, {N_WORKERS} workers",
          flush=True)
    if not todo:
        print("run7 dop853: nothing to do", flush=True)
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
                el = time.time() - start
                rate = finished / el
                left = (len(todo) - finished) / rate if rate else 0
                print(f"[{finished:3d}/{len(todo):3d}] traj {idx} in "
                      f"{secs/60:.1f} min  finish "
                      f"~{(datetime.now() + timedelta(seconds=left)):%H:%M}",
                      flush=True)
        total = time.time() - start
    print(f"run7 dop853: done in {timedelta(seconds=int(total))}", flush=True)


if __name__ == "__main__":
    main()
