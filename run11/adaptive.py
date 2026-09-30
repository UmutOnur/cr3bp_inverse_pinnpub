"""run11: do the hand-picked loss weights (10/10/5/5) cost the PINN anything?

Phase 1 minimises w_ic*ic + w_data*data + w_accel*accel + w_vel*vel with those
four numbers fixed by hand and never justified. The four terms carry different
units and different natural sizes, so whichever is numerically largest supplies
most of the gradient. Standard practice since ~2021 is to rebalance them during
training from the gradients themselves.

Two arms per orbit in one process: FIXED (the current constants, the control)
and ADAPTIVE (gradient balancing). Everything else - orbit, data, seed, gates,
optimizer, cap - is identical, because run-to-run scatter on a fixed setup is
about 3x and a comparison against older files would measure that instead.

Full state with the spline target, where all four terms are active. In the
positions-only no-target setting accel is identically zero and ic is dropped
under noise, so there is almost nothing to rebalance.
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train

DATA = "E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy"

# 14 orbits, one per occupied (mu band x closest-approach band) cell of run5's
# 58, plus 646 as the regression anchor. All are in run5 and in run7's index
# set, so both provide a reference.
ORBITS = [81, 131, 166, 396, 381, 391, 581, 611, 616, 651, 646, 706, 841, 861, 896]
if os.environ.get("RUN11_ORBITS"):
    ORBITS = [int(v) for v in os.environ["RUN11_ORBITS"].split(",")]

# env overrides exist so the loop can be smoke-tested cheaply without editing
CAP = int(os.environ.get("RUN11_CAP", 40000))   # run5 used 100k; 33% would cap
LBFGS = int(os.environ.get("RUN11_LBFGS", train.PHASE_ONE_LBFGS_ITERS))
BALANCE_EVERY = 500         # how often the adaptive arm rebalances
BALANCE_ALPHA = 0.1         # smoothing on the weight update
WEIGHT_CLIP = (1e-3, 1e4)   # keep a term from being switched off entirely

train.PHASE_ONE_ITERS = CAP
train.USE_VELOCITY = True
train.ACCEL_TARGET = "spline"
train.SAVE_NETS = False     # this script saves with the arm in the name


def true_acceleration(x, y, vx, vy, mu):
    """Analytic planar CR3BP acceleration in the rotating frame."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    ax = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    ay = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return ax, ay


def net_acceleration(net, t):
    tg = t.clone().detach().requires_grad_(True)
    state = net(tg)
    ones = torch.ones_like(state[:, 0:1])
    ax = torch.autograd.grad(state[:, 2:3], tg, grad_outputs=ones, retain_graph=True)[0]
    ay = torch.autograd.grad(state[:, 3:4], tg, grad_outputs=ones)[0]
    return ax.detach().cpu().numpy().ravel(), ay.detach().cpu().numpy().ravel()


def grad_norm(term, params):
    """Mean absolute gradient of one loss term w.r.t. the network weights."""
    g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    tot = cnt = 0.0
    for v in g:
        if v is not None:
            tot += v.abs().sum().item()
            cnt += v.numel()
    return tot / max(cnt, 1)


def run_phase_one(t, x, y, vx, vy, idx, adaptive):
    """Mirrors train.train_phaseone, with an optional weight rebalance.

    Both arms go through this same loop, so the control is not a different
    code path - the only difference is whether the weights are updated.
    """
    net = train.FNN().to(train.DEVICE)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=train.ADAM_LR)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=train.LR_DECAY_GAMMA)
    compute_loss, a_scale, gates = train.build_phaseone_loss(net, t, x, y, vx, vy, 0.0)

    accel_gate = gates["accel"] if gates["accel"] is not None else float("inf")
    vel_gate = gates["vel"] if gates["vel"] is not None else float("inf")
    data_gate = gates["data"]

    it, data_l, accel_l, vel_l = 0, float("inf"), float("inf"), float("inf")
    t0 = time.time()
    while (data_l > data_gate or accel_l > accel_gate
           or vel_l > vel_gate) and it < CAP:
        opt.zero_grad()
        loss, ic_l, data_l, accel_l, vel_l = compute_loss()

        if adaptive and it % BALANCE_EVERY == 0:
            # level the terms: each weight scales so its weighted gradient
            # matches the largest raw one, smoothed so the loss surface does
            # not jump from one iteration to the next
            parts = compute_loss.parts
            norms = {k: grad_norm(v, params) for k, v in parts.items()
                     if v.requires_grad and v.detach().item() > 0}
            if norms:
                big = max(norms.values())
                for k, n in norms.items():
                    if n > 0:
                        target = big / n
                        w = compute_loss.weights[k]
                        compute_loss.weights[k] = float(np.clip(
                            (1 - BALANCE_ALPHA) * w + BALANCE_ALPHA * target,
                            *WEIGHT_CLIP))
            loss, ic_l, data_l, accel_l, vel_l = compute_loss()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, train.GRAD_CLIP_NORM)
        opt.step()
        sched.step()
        it += 1
        if it % 5000 == 0:
            w = compute_loss.weights
            print(f"    it={it:6d} data={data_l:.2e} accel={accel_l:.2e} "
                  f"vel={vel_l:.2e} w=({w['ic']:.2f},{w['data']:.2f},"
                  f"{w['accel']:.2f},{w['vel']:.2f})", flush=True)

    adam_s = time.time() - t0
    hit_cap = it >= CAP
    data_l, accel_l, vel_l, _ = train.lbfgs_phaseone(net, compute_loss, LBFGS, idx)
    return net, compute_loss, data_l, accel_l, vel_l, it, hit_cap, adam_s


COLS = ["arm", "traj_index", "mu_true", "min_r", "mu_pred", "error_pct", "abs_error",
        "accel_rmse", "rms_a_true", "accel_rel", "phase1_iters", "phase1_hit_cap",
        "phase1_data_l", "phase1_accel_l", "phase1_vel_l", "w_ic", "w_data",
        "w_accel", "w_vel", "scan_argmin", "scan_contrast", "adam_s", "elapsed_s"]
with open("results.csv", "w", newline="") as f:
    csv.writer(f).writerow(COLS)

print(f"run11: {len(ORBITS)} orbits x 2 arms, device {train.DEVICE}, cap {CAP}",
      flush=True)
for idx in ORBITS:
    for arm in ("fixed", "adaptive"):
        print(f"=== {arm} traj {idx} ===", flush=True)
        t_start = time.time()
        try:
            train.set_seed(train.SEED)          # same init for both arms
            t, x, y, vx, vy, mu_true = train.load_trajectory(DATA, idx)
            min_r = train.min_primary_distance(x, y, mu_true)
            ax_t, ay_t = true_acceleration(x.ravel(), y.ravel(), vx.ravel(),
                                           vy.ravel(), mu_true)
            rms_a = float(np.sqrt(np.mean(ax_t ** 2 + ay_t ** 2)))

            net, cl, data_l, accel_l, vel_l, iters, hit_cap, adam_s = \
                run_phase_one(t, x, y, vx, vy, idx, arm == "adaptive")
            torch.save(net.state_dict(), f"net_{arm}_traj{idx}.pt")

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

            w = cl.weights
            print(f"  [{arm} {idx}] accel_rel={accel_rmse / rms_a:.4g} "
                  f"mu_err={abs(mu_pred - mu_true) / mu_true * 100:.4g}%", flush=True)
            row = [arm, idx, mu_true, min_r, mu_pred,
                   abs(mu_pred - mu_true) / mu_true * 100, abs(mu_pred - mu_true),
                   accel_rmse, rms_a, accel_rmse / rms_a, iters, hit_cap,
                   data_l, accel_l, vel_l, w["ic"], w["data"], w["accel"], w["vel"],
                   argmin, contrast, adam_s, time.time() - t_start]
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            row = [arm, idx] + [""] * 20 + [time.time() - t_start]
        with open("results.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  done in {time.time() - t_start:.0f}s", flush=True)
print("run11: done", flush=True)
