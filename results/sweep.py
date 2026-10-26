"""The current model and the network-free methods on the 999 orbits in this
folder (all 1000 without the truncated 938). Non-spline PINN,
position-only, gradient-norm weighting, K = 6, 40k Adam cap, L-BFGS 2500.
n = 1000, sigma = 1e-3, noise seeds 1000*idx + 4; every network-free method
sees the identical samples.

Main number: the strong-form phase 2 (mu_pred). Also recorded from each net:
the strong form on the inner 80% of the samples, and the weak form (half-widths
0.4, 0.8) and Jacobi constant on the net's curve.

  python sweep.py pinn        GPU: phase 1 and the read-outs -> pinn_results.csv
  python sweep.py baselines   CPU: network-free methods     -> baseline_results.csv
  python sweep.py crb         CPU: Cramer-Rao bound         -> crb.csv
  python ../summary.py .
The two CSVs here carry crb.csv's crb_sd_mu as a column. Rows with cost_first
empty (indices 1, 6, ..., 996) come from an earlier sweep on the same samples:
BLS there ran from a single start, as the restart rule needs the first cost.
Run from this folder. Each mode resumes from its CSV, skipping finished
orbits. Env overrides: SWEEP_ORBITS (comma-separated indices), SWEEP_CAP,
SWEEP_LBFGS, SWEEP_WORKERS.
"""
import csv
import os
import sys
import threading
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = str(Path(__file__).resolve().parents[1])       # repository root
sys.path.insert(0, ROOT)
import train
import baselines as B

DATA = f"{ROOT}/cr3bp_dataset_final.npy"
N_POINTS = 1000
SIGMA, SIGMA_INDEX = 1e-3, 4                      # run7 sigma order
train.USE_VELOCITY = False
train.ACCEL_TARGET = "none"
train.PHASE_ONE_WEIGHTING = "gradnorm"
train.SAVE_NETS = False                           # saved below as net_traj<idx>.pt
train.PHASE_ONE_ITERS = int(os.environ.get("SWEEP_CAP", 40000))
train.PHASE_ONE_LBFGS_ITERS = int(os.environ.get("SWEEP_LBFGS", train.PHASE_ONE_LBFGS_ITERS))
WORKERS = int(os.environ.get("SWEEP_WORKERS", 8))
TRIM = 0.10                                       # trimmed strong form drops 10% at each end
WIDTHS = (0.4, 0.8)                               # weak-form bump half-widths
T_FINE = np.linspace(0, 2 * np.pi, 4001)          # net curve for the weak and Jacobi read-outs
EDGE = 0.05                                       # accel_rel_edges: outer 5% at each end
SMOOTHERS = ("spline", "smoothspline", "fourier", "savgol", "tvdiff")
BLS_RESTARTS = (-6.0, -4.0, -2.5, -1.5, -0.7)     # log10 mu starts when the first fit fails,
BLS_RESTART_FACTOR = 10                           # i.e. cost > 10x the noise-only n*sigma^2

PINN_COLS = ["traj_index", "group", "n_points", "sigma", "noise_seed", "mu_true", "min_r",
             "mu_pred", "error_pct", "abs_error", "phase2_residual", "scan_argmin",
             "scan_contrast", "mu_strong_trim", "contrast_strong_trim",
             "mu_weak_net_0.4", "contrast_weak_net_0.4", "mu_weak_net_0.8",
             "contrast_weak_net_0.8", "mu_jacobi_net", "contrast_jacobi_net",
             "accel_rel", "accel_rel_interior", "accel_rel_edges",
             "phase1_iters", "phase1_hit_cap", "phase1_data_l", "phase1_vel_l",
             "w_data_final", "w_vel_final", "num_frequencies", "features", "weighting",
             "gradnorm_every", "gradnorm_alpha", "adam_cap", "lbfgs_iters", "accel_target",
             "use_velocity", "noise_gate_mult", "w_jerk", "seed", "trim", "elapsed_s", "error"]
BASE_COLS = ["traj_index", "group", "n_points", "sigma", "noise_seed", "mu_true", "min_r",
             "method", "mu_est", "abs_error", "error_pct", "contrast", "hyperparam", "cost",
             "cost_first", "mu_first", "n_fits", "success", "nfev", "elapsed_s", "error"]
CRB_COLS = ["traj_index", "mu_true", "crb_sd_mu", "reint_pos_err", "seconds"]


def orbits():
    """(index, min_r, group); closest approach >= 0.05 first."""
    idx = ([int(i) for i in os.environ["SWEEP_ORBITS"].split(",")] if "SWEEP_ORBITS" in os.environ
           else [i for i in range(1000) if i != 938])
    out = []
    for i in idx:
        _, x, y, _, _, mu = train.load_trajectory(DATA, i)
        r = train.min_primary_distance(x, y, mu)
        out.append((i, r, "close" if r < 0.05 else ("far" if r > 0.5 else "middle")))
    return [o for o in out if o[1] >= 0.05] + [o for o in out if o[1] < 0.05]


def samples(idx):
    """Clean record, and the noisy samples every method sees."""
    clean = train.load_trajectory(DATA, idx)
    seed = 1000 * idx + SIGMA_INDEX
    return clean, train.perturb_trajectory(*clean[:5], N_POINTS, SIGMA, seed=seed), seed


def net_curve(net, t):
    """Net outputs and their autograd time derivatives on t:
    x, y, vx, vy, dx/dt, dy/dt, dvx/dt, dvy/dt."""
    tt = torch.tensor(t.reshape(-1, 1), dtype=train.DTYPE, device=train.DEVICE, requires_grad=True)
    out = net(tt)
    grad = lambda k: torch.autograd.grad(out[:, k].sum(), tt, retain_graph=True)[0].flatten()
    return [v.detach().cpu().numpy() for v in [out[:, k] for k in range(4)] + [grad(k) for k in range(4)]]


def run_pinn(idx, min_r, group):
    t0 = time.time()
    train.set_seed(train.SEED)
    (tc, xc, yc, vxc, vyc, mu_true), (t, x, y, vx, vy), seed = samples(idx)
    net, data_l, _, vel_l, iters, hit_cap = train.train_phaseone(t, x, y, vx, vy, idx, SIGMA)
    w = train.train_phaseone.weights
    torch.save(net.state_dict(), f"net_traj{idx}.pt")

    mu_pred, argmin, contrast = train.train_phasetwo(net, t, mu_true, idx)    # the main number
    os.replace(f"mu_history_traj{idx}.csv", f"mu_history_strong_traj{idx}.csv")
    m = int(round(TRIM * len(t)))
    mu_trim, _, contrast_trim = train.train_phasetwo(net, t[m:-m], mu_true, idx)
    os.replace(f"mu_history_traj{idx}.csv", f"mu_history_trim_traj{idx}.csv")
    X, Y, _, _, DX, DY, _, _ = net_curve(net, T_FINE)
    weak = [B.estimate_mu_weak(T_FINE, X, Y, half_width=a) for a in WIDTHS]
    jac = B.estimate_mu_jacobi(X, Y, DX, DY)

    # secondary: net acceleration against the truth, and the residual at mu_pred
    *_, AX, AY = net_curve(net, tc.ravel())
    fx, fy = B.effective_force(xc.ravel(), yc.ravel(), mu_true)
    e2 = (AX - fx - 2 * vyc.ravel()) ** 2 + (AY - fy + 2 * vxc.ravel()) ** 2
    rms_a = np.sqrt(np.mean((fx + 2 * vyc.ravel()) ** 2 + (fy - 2 * vxc.ravel()) ** 2))
    k = int(round(EDGE * len(e2)))
    rel = lambda v: float(np.sqrt(np.mean(v)) / rms_a)
    res = train.physics_residual(net, torch.tensor(mu_pred, dtype=train.DTYPE, device=train.DEVICE),
                                 torch.tensor(t.reshape(-1, 1), dtype=train.DTYPE, device=train.DEVICE))
    resid = sum(torch.mean(r ** 2) for r in res).item()

    err = abs(mu_pred - mu_true)
    return [idx, group, N_POINTS, SIGMA, seed, mu_true, min_r, mu_pred, err / mu_true * 100, err,
            resid, argmin, contrast, mu_trim, contrast_trim, weak[0]["mu_pred"], weak[0]["contrast"],
            weak[1]["mu_pred"], weak[1]["contrast"], jac["mu_pred"], jac["contrast"],
            rel(e2), rel(e2[k:-k]), rel(np.r_[e2[:k], e2[-k:]]), iters, hit_cap, data_l, vel_l,
            w["data"], w["vel"], net.num_frequencies, train.FEATURES, train.PHASE_ONE_WEIGHTING,
            train.GRADNORM_EVERY, train.GRADNORM_ALPHA, train.PHASE_ONE_ITERS,
            train.PHASE_ONE_LBFGS_ITERS, train.ACCEL_TARGET, train.USE_VELOCITY,
            train.NOISE_GATE_MULT, train.W_JERK, train.SEED, TRIM, time.time() - t0, ""]


def run_baselines(job):
    """Every network-free method on one orbit's samples; one row per method."""
    idx, min_r, group = job
    (_, _, _, _, _, mu_true), (t, x, y, vx, vy), seed = samples(idx)
    head = [idx, group, N_POINTS, SIGMA, seed, mu_true, min_r]
    rows = []

    def attempt(method, call, hyper=lambda r: ""):
        t0 = time.time()
        try:
            r = call()
            mu = float(r["mu_pred"])
            rows.append(head + [method, mu, abs(mu - mu_true), abs(mu - mu_true) / mu_true * 100,
                                r.get("contrast", ""), hyper(r), r.get("cost", ""),
                                r.get("cost_first", ""), r.get("mu_first", ""), r.get("n_fits", ""),
                                r.get("success", ""), r.get("nfev", ""), time.time() - t0, ""])
            return r
        except Exception as exc:
            rows.append(head + [method] + [""] * (len(BASE_COLS) - 10)
                        + [time.time() - t0, f"{type(exc).__name__}: {exc}"])
            return None

    hypers = dict(spline=lambda r: "", smoothspline=lambda r: f"lam={r.get('lam')}",
                  fourier=lambda r: f"K={r.get('harmonics')}",
                  savgol=lambda r: f"w={r.get('window')},po={r.get('polyorder')}",
                  tvdiff=lambda r: f"alpha={r.get('alpha')}")
    res = {}
    for name in SMOOTHERS:
        fn = getattr(B, f"estimate_mu_{name}")
        res[name] = attempt(name, lambda fn=fn: fn(t, x, y, vx, vy, use_velocity=False), hypers[name])
    attempt("bls", lambda: B.estimate_mu_bls(t, x, y, vx, vy, use_velocity=False,
                                             restart_log_mu=BLS_RESTARTS,
                                             restart_above=BLS_RESTART_FACTOR * len(t) * SIGMA ** 2),
            lambda r: f"start_log_mu={r['start_log_mu']:.3g}")
    attempt("ekf", lambda: B.estimate_mu_ekf(t, x, y, vx, vy, use_velocity=False, sigma=SIGMA),
            lambda r: f"mu0={r.get('mu0_best')},diverged={r.get('diverged', '')}")
    for a in WIDTHS:
        attempt(f"weak_raw_{a}", lambda a=a: B.estimate_mu_weak(t, x, y, half_width=a))
    sg = res.get("savgol")
    if sg is not None:
        attempt("jacobi_savgol",
                lambda: B.estimate_mu_jacobi_savgol(t, x, y, sg["window"], sg["polyorder"]),
                lambda r: f"w={sg['window']},po={sg['polyorder']}")
    return idx, rows


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pinn"
    out, cols, log_name = {"pinn": ("pinn_results.csv", PINN_COLS, "sweep.log"),
                           "baselines": ("baseline_results.csv", BASE_COLS, "baselines.log"),
                           "crb": ("crb.csv", CRB_COLS, "crb.log")}[mode]
    console = sys.stdout
    tty = console.isatty()
    log = open(log_name, "a", buffering=1, encoding="utf-8")
    sys.stdout = sys.stderr = log
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(f"sweep {mode}")
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)   # keep the machine awake
    except Exception:
        pass

    jobs = orbits()
    done = set()
    if os.path.exists(out):
        done = set(pd.read_csv(out).traj_index.astype(int))
    else:
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(cols)
    todo = [j for j in jobs if j[0] not in done]
    state = {"job": None, "k": 0, "t0": time.time(), "stop": False}
    start, times = time.time(), []

    def say(msg):
        console.write(("\r" + " " * 110 + "\r" if tty else "") + msg + "\n")
        console.flush()
        print(msg, flush=True)

    def write(rows):
        with open(out, "a", newline="") as f:
            csv.writer(f).writerows(rows)
            f.flush()
            os.fsync(f.fileno())

    def ticker():
        while not state["stop"]:
            if state["job"] is not None:
                eta = np.mean(times) * (len(todo) - state["k"]) if times else None
                console.write("\r" + f"  orbit {state['k'] + 1}/{len(todo)} (traj {state['job']})"
                              f" | this orbit {fmt(time.time() - state['t0'])}"
                              f" | elapsed {fmt(time.time() - start)}"
                              + (f" | rough ETA {fmt(eta)}" if eta else "") + "   ")
                console.flush()
            time.sleep(1)

    say(f"sweep {mode}: {len(jobs)} orbits, {len(jobs) - len(todo)} already done, {len(todo)} to run. "
        + (f"Device {train.DEVICE}, Adam cap {train.PHASE_ONE_ITERS}, L-BFGS "
           f"{train.PHASE_ONE_LBFGS_ITERS}." if mode == "pinn" else f"{WORKERS} workers."))
    say("Safe to close this window at any time; rerun the same command to resume.")
    try:
        if mode == "pinn":
            if tty:
                threading.Thread(target=ticker, daemon=True).start()
            for k, (idx, min_r, group) in enumerate(todo):
                state.update(job=idx, k=k, t0=time.time())
                print(f"=== traj {idx} ({group}) ===", flush=True)
                try:
                    row = run_pinn(idx, min_r, group)
                    status = (f"err {row[8]:.3g}%  abs {row[9]:.2g}  iters {row[24]}"
                              f"{' (cap)' if row[25] else ''}")
                except Exception as exc:
                    print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                    row = [idx, group, N_POINTS, SIGMA, 1000 * idx + SIGMA_INDEX, "", min_r] \
                        + [""] * (len(PINN_COLS) - 9) + [time.time() - state["t0"],
                                                         f"{type(exc).__name__}: {exc}"]
                    status = f"FAILED ({type(exc).__name__})"
                write([row])
                times.append(time.time() - state["t0"])
                say(f"[{k + 1}/{len(todo)}] traj {idx} ({group}): {status}  "
                    f"({fmt(times[-1])}, total {fmt(time.time() - start)})")
        elif mode == "baselines":
            with Pool(WORKERS) as pool:
                for k, (idx, rows) in enumerate(pool.imap_unordered(run_baselines, todo)):
                    write(rows)
                    bad = sum(1 for r in rows if r[-1])
                    say(f"[{k + 1}/{len(todo)}] traj {idx}: {len(rows)} methods"
                        f"{f', {bad} failed' if bad else ''}  (total {fmt(time.time() - start)})")
        else:
            import crb
            with Pool(WORKERS) as pool:
                for k, row in enumerate(pool.imap_unordered(crb.job, [j[0] for j in todo])):
                    write([row])
            say(f"crb: {len(todo)} orbits in {fmt(time.time() - start)}")
        state["stop"] = True
        say(f"sweep {mode} finished in {fmt(time.time() - start)}.")
    except KeyboardInterrupt:
        state["stop"] = True
        say("Stopped. Finished orbits are saved; rerun to resume.")
    finally:
        sys.stdout, sys.stderr = console, sys.__stderr__
        log.close()
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        except Exception:
            pass


if __name__ == "__main__":
    main()
