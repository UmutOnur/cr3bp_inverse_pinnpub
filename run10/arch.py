"""run10: is the no-target PINN's derivative floor a property of PINNs or
of this particular network?

The project notes attribute the floor to bandwidth - "6 Fourier input frequencies,
tanh" - but that has never been tested, and it is the first thing a referee
will ask. Same orbit, same data, same seed, same gates as run9 cell H
(position-only, no target, clean, n=1000, 40k Adam cap); only the
architecture changes.

The acceleration error is also *measured* here rather than inferred from the
mu error through run7's relation, so the inference chain itself gets checked
on the K=6 row against run8's implied 0.090 for 646.
"""
import csv
import functools
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train

DATA = "E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy"
_FNN = train.FNN
W4 = (128, 128, 128, 128, 4)
W256 = (256, 256, 256, 256, 4)
# label, num_frequencies, layer_sizes, orbits
CONFIGS = [
    ("K6_w128",  6,  W4,   [646, 821]),      # run9 cell H, the baseline
    ("K12_w128", 12, W4,   [646]),
    ("K24_w128", 24, W4,   [646, 821]),
    ("K12_w256", 12, W256, [646]),
]
train.PHASE_ONE_ITERS = 40000
train.USE_VELOCITY = False
train.ACCEL_TARGET = "none"
train.SAVE_NETS = True
N_POINTS, SIGMA = 1000, 0.0


def true_acceleration(x, y, vx, vy, mu):
    """Analytic planar CR3BP acceleration in the rotating frame."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    ax = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    ay = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return ax, ay


def net_acceleration(net, t):
    """The net's own d(vx)/dt, d(vy)/dt by autograd, as numpy."""
    tg = t.clone().detach().requires_grad_(True)
    state = net(tg)
    ones = torch.ones_like(state[:, 0:1])
    ax = torch.autograd.grad(state[:, 2:3], tg, grad_outputs=ones, retain_graph=True)[0]
    ay = torch.autograd.grad(state[:, 3:4], tg, grad_outputs=ones)[0]
    return ax.detach().cpu().numpy().ravel(), ay.detach().cpu().numpy().ravel()


COLS = ["config", "num_frequencies", "width", "n_params", "traj_index", "mu_true",
        "min_r", "mu_pred", "error_pct", "abs_error", "accel_rmse", "rms_a_true",
        "accel_rel", "phase1_iters", "phase1_hit_cap", "phase1_data_l",
        "phase1_data_gate", "phase1_vel_l", "scan_argmin", "scan_contrast",
        "elapsed_s"]
out = "results.csv"
with open(out, "w", newline="") as f:
    csv.writer(f).writerow(COLS)

n_runs = sum(len(c[3]) for c in CONFIGS)
print(f"run10: {n_runs} runs, device {train.DEVICE}, cap {train.PHASE_ONE_ITERS}",
      flush=True)
for label, K, sizes, orbits in CONFIGS:
    train.FNN = functools.partial(_FNN, num_frequencies=K, layer_sizes=sizes)
    for idx in orbits:
        print(f"=== {label} traj {idx} ===", flush=True)
        t0 = time.time()
        try:
            train.set_seed(train.SEED)
            t, x, y, vx, vy, mu_true = train.load_trajectory(DATA, idx)
            min_r = train.min_primary_distance(x, y, mu_true)
            t, x, y, vx, vy = train.perturb_trajectory(t, x, y, vx, vy, N_POINTS,
                                                       SIGMA, seed=1000 * idx)
            xa, ya = x.ravel(), y.ravel()
            ax_t, ay_t = true_acceleration(xa, ya, vx.ravel(), vy.ravel(), mu_true)
            rms_a = float(np.sqrt(np.mean(ax_t ** 2 + ay_t ** 2)))

            net, data_l, accel_l, vel_l, iters, hit_cap = \
                train.train_phaseone(t, x, y, vx, vy, idx, SIGMA)
            # train.py saves to a fixed net_traj{idx}.pt, so each config would
            # overwrite the last (run6's bug); put the architecture in the name
            src = f"net_traj{idx}.pt"
            if os.path.exists(src):
                os.replace(src, f"net_{label}_traj{idx}.pt")
            _, _, gates = train.build_phaseone_loss(net, t, x, y, vx, vy, SIGMA)

            mu_pred, argmin, contrast = train.train_phasetwo(net, t, mu_true, idx)
            try:
                t_t = torch.tensor(t.reshape(-1, 1), dtype=train.DTYPE,
                                   device=train.DEVICE)
                ax_n, ay_n = net_acceleration(net, t_t)
                accel_rmse = float(np.sqrt(np.mean((ax_n - ax_t) ** 2
                                                   + (ay_n - ay_t) ** 2)))
            except Exception as exc:
                print(f"  accel measurement failed: {exc}", flush=True)
                accel_rmse = float("nan")
            n_par = sum(p.numel() for p in net.parameters())
            print(f"  [{label} {idx}] accel_rmse={accel_rmse:.4g} "
                  f"rel={accel_rmse / rms_a:.4g} mu_err={abs(mu_pred - mu_true) / mu_true * 100:.4g}%",
                  flush=True)
            row = [label, K, sizes[0], n_par, idx, mu_true, min_r, mu_pred,
                   abs(mu_pred - mu_true) / mu_true * 100, abs(mu_pred - mu_true),
                   accel_rmse, rms_a, accel_rmse / rms_a, iters, hit_cap, data_l,
                   gates["data"], vel_l, argmin, contrast, time.time() - t0]
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            row = [label, K, sizes[0], "", idx] + [""] * 15 + [time.time() - t0]
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
print("run10: done", flush=True)
