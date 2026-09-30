"""
CR3BP mu-discovery PINN, two-phase pipeline.

Phase one (train_phaseone): fit a smooth surrogate (x, y, vx, vy) of the
trajectory from data + ic + acceleration losses. No mu involved.

Phase two (train_phasetwo): freeze the network permanently. mu is the
only trainable variable, optimized with Adam then L-BFGS against the
CR3BP physics residual evaluated at the observed times.
"""

import csv
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import make_interp_spline

from baselines import (classical_scan, estimate_mu_shooting, estimate_mu_ekf,
                       estimate_mu_spline, estimate_mu_fourier, FourierFit)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
torch.set_default_dtype(torch.float64)

# phase one
ADAM_LR = 1e-3
LR_DECAY_GAMMA = 0.99995   # applied per iteration in phase one
# Adam stops when each loss is below the LOOSER of an absolute gate and a
# relative gate (rel * mean(target**2)). Orbits that converged in run4 are
# unchanged (absolute wins); high-|a| orbits no longer sit at the cap
# (relative wins). L-BFGS does the real tightening afterwards.
PHASE_ONE_ACCEL_ABS_THRESHOLD = 1e-6
PHASE_ONE_ACCEL_REL_THRESHOLD = 6e-8
PHASE_ONE_VEL_ABS_THRESHOLD   = 1e-6
PHASE_ONE_VEL_REL_THRESHOLD   = 6e-8
PHASE_ONE_DATAERR_THRESHOLD = 1e-3    # ceiling on the state MSE gate
PHASE_ONE_DATA_TIDAL_REL = 1e-6       # state gate at rms|a| = 1. Near a
                                      # primary the residual amplifies a
                                      # position error by ~2|a|^1.5, so the
                                      # gate tightens with a_scale**1.5
PHASE_ONE_DATA_GATE_FLOOR = 1e-8      # a_scale**1.5 overshoots on smooth
                                      # high-|a| orbits; L-BFGS does the
                                      # tightening below this anyway
SAVE_NETS = True                      # torch.save each phase-1 net to cwd
PHASE_ONE_ITERS = 100000
USE_VELOCITY = True           # False: positions only (the proposal's
                              # setting). Data and ic losses on x, y; the
                              # net's vx, vy are tied to d(x)/dt, d(y)/dt by
                              # a self-consistency loss instead of observed v
ACCEL_TARGET = "spline"       # "spline": interpolating spline derivative of
                              # observed v (septic on x, y twice if positions
                              # only); "fd": np.gradient; "fourier": global
                              # cross-validated Fourier fit (noise-tolerant);
                              # "none": no manufactured target at all
NOISE_GATE_MULT = 2.0         # with noise sigma, no loss against noisy data
                              # is asked to go below MULT * channels * sigma^2
PHASE_ONE_LBFGS_ITERS = 2500  # L-BFGS polish after the Adam gates pass;
                              # losses plateau by ~2000-2500 (tested 2026-09-14)
W_IC = 10.0
W_DATA = 10.0
W_ACCEL = 5.0
W_VEL = 5.0
GRAD_CLIP_NORM = 5.0
DISPLAY_EVERY = 2000
MU_LOG_EVERY = 50

# phase two (permanent freeze: mu is the only trainable variable)
PHASE_TWO_ADAM_ITERS = 500    # flat well before this in every run4 trace
PHASE_TWO_ADAM_LR = 1e-2      # single scalar -- a higher LR is fine here
LBFGS_ITERS = 1000
MU_LOG_LO = -7.0              # mu is optimized in log10 space over
MU_LOG_HI = np.log10(0.55)    # [1e-7, 0.55]; 0.5 must be interior
SCAN_POINTS = 400             # log-spaced grid for the phase-2 global scan


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Data loading (format: [x,y,z,vx,vy,vz,t,mu], 1000 rows per trajectory)
# ---------------------------------------------------------------------------
def load_trajectory(dataset_path, traj_index=0):
    data = np.load(dataset_path)
    t_col = data[:, 6]
    boundaries = np.where(np.diff(t_col) < 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(data)]])

    start = boundaries[traj_index]
    end = boundaries[traj_index + 1]
    traj = data[start:end]

    t = traj[:, 6:7]
    x = traj[:, 0:1]
    y = traj[:, 1:2]
    vx = traj[:, 3:4]
    vy = traj[:, 4:5]
    mu_true = float(traj[0, 7])
    return t, x, y, vx, vy, mu_true


def estimate_acceleration(t, x, y, vx, vy):
    """Acceleration targets from the data, per ACCEL_TARGET. No mu needed.
    Returns (ax, ay) as [N,1] arrays, or None when ACCEL_TARGET == "none"."""
    if ACCEL_TARGET == "none":
        return None
    t_f = t.flatten()
    if USE_VELOCITY:
        ux, uy, order = vx.flatten(), vy.flatten(), 1
    else:
        ux, uy, order = x.flatten(), y.flatten(), 2
    if ACCEL_TARGET == "spline":
        k = 5 if order == 1 else 7
        ax = make_interp_spline(t_f, ux, k=k).derivative(order)(t_f)
        ay = make_interp_spline(t_f, uy, k=k).derivative(order)(t_f)
    elif ACCEL_TARGET == "fourier":
        # K is no longer chosen by cross-validation inside FourierFit (that
        # selection moved into the baseline estimator and is made on the
        # physics residual, which phase 1 must not see). Use the largest K
        # that keeps the fit comfortably overdetermined, 2K+3 <= n/8: as a
        # training target the point is a smooth fit, not an interpolating one.
        K = max(K for K in FourierFit.HARMONICS if 2 * K + 3 <= max(len(t_f) // 8, 7))             if any(2 * K + 3 <= max(len(t_f) // 8, 7) for K in FourierFit.HARMONICS) else 2
        ax = FourierFit(t_f, ux, K)(t_f, order)
        ay = FourierFit(t_f, uy, K)(t_f, order)
    elif ACCEL_TARGET == "fd":
        ax, ay = ux, uy
        for _ in range(order):
            ax, ay = np.gradient(ax, t_f), np.gradient(ay, t_f)
    else:
        raise ValueError(f"unknown ACCEL_TARGET {ACCEL_TARGET!r}")
    return ax.reshape(-1, 1), ay.reshape(-1, 1)


def perturb_trajectory(t, x, y, vx, vy, n_points=None, noise_sigma=0.0, seed=0):
    """Add Gaussian noise to every state channel, then keep n_points samples
    spread evenly over the arc. Same recipe as the CPU sparse/noise study,
    so PINN and baselines see identical data."""
    rng = np.random.default_rng(seed)
    state = np.hstack([x, y, vx, vy])
    if noise_sigma > 0:
        state = state + noise_sigma * rng.standard_normal(state.shape)
    if n_points is not None and n_points < len(t):
        # a fixed stride where it divides evenly, so the kept samples are
        # exactly uniform in time: Savitzky-Golay assumes uniform spacing,
        # and np.linspace here gives alternating gaps (5 and 6 at n=200),
        # a ~20% error in the spacing it is handed
        if len(t) % n_points == 0:
            sel = np.arange(0, len(t), len(t) // n_points)
        else:
            sel = np.linspace(0, len(t) - 1, n_points).round().astype(int)
        t, state = t[sel], state[sel]
    return t, state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]


def min_primary_distance(x, y, mu_true):
    """Closest approach to either primary, in units of primary separation."""
    r1 = np.sqrt((x.flatten() + mu_true) ** 2 + y.flatten() ** 2)
    r2 = np.sqrt((x.flatten() - (1 - mu_true)) ** 2 + y.flatten() ** 2)
    return float(min(r1.min(), r2.min()))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
def fourier_features(t, num_frequencies=6):
    """Expand t into [t, sin(kt), cos(kt)] for k = 1..num_frequencies.
    Input [N,1] -> output [N, 1 + 2*num_frequencies]."""
    freqs = torch.arange(1, num_frequencies + 1, device=t.device, dtype=t.dtype)
    angles = t * freqs
    return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)


class FNN(nn.Module):
    def __init__(self, num_frequencies=6, layer_sizes=(128, 128, 128, 128, 4)):
        super().__init__()
        self.num_frequencies = num_frequencies
        input_dim = 1 + 2 * num_frequencies
        full_sizes = (input_dim,) + layer_sizes

        layers = []
        for i in range(len(full_sizes) - 2):
            lin = nn.Linear(full_sizes[i], full_sizes[i + 1])
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
        lin_out = nn.Linear(full_sizes[-2], full_sizes[-1])
        nn.init.xavier_normal_(lin_out.weight)
        nn.init.zeros_(lin_out.bias)
        layers.append(lin_out)
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        t_expanded = fourier_features(t, self.num_frequencies)
        return self.net(t_expanded)


def mu_from_raw(mu_raw):
    """log10(mu) = lo + (hi - lo) * sigmoid(mu_raw): bounded, log-scaled."""
    log_mu = MU_LOG_LO + (MU_LOG_HI - MU_LOG_LO) * torch.sigmoid(mu_raw)
    return 10.0 ** log_mu


def inv_sigmoid_init(mu0):
    """Pick mu_raw so that mu_from_raw(mu_raw) == mu0, for warm starts."""
    p = (np.log10(mu0) - MU_LOG_LO) / (MU_LOG_HI - MU_LOG_LO)
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


# ---------------------------------------------------------------------------
# Physics residual
# ---------------------------------------------------------------------------
def physics_residual(net, mu, t):
    t = t.clone().requires_grad_(True)
    state = net(t)
    x, y, vx, vy = state[:, 0:1], state[:, 1:2], state[:, 2:3], state[:, 3:4]

    ones = torch.ones_like(x)
    dx = torch.autograd.grad(x, t, grad_outputs=ones, create_graph=True)[0]
    dy = torch.autograd.grad(y, t, grad_outputs=ones, create_graph=True)[0]
    dvx = torch.autograd.grad(vx, t, grad_outputs=ones, create_graph=True)[0]
    dvy = torch.autograd.grad(vy, t, grad_outputs=ones, create_graph=True)[0]

    r1 = torch.sqrt((x + mu) ** 2 + y ** 2)
    r2 = torch.sqrt((x - (1 - mu)) ** 2 + y ** 2)

    res_x = dx - vx
    res_y = dy - vy
    res_vx = dvx - (x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy)
    res_vy = dvy - (y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx)
    return res_x, res_y, res_vx, res_vy


# ---------------------------------------------------------------------------
# Phase one: fit the trajectory WITH the acceleration constraint
# ---------------------------------------------------------------------------
def build_phaseone_loss(net, t, x, y, vx, vy, noise_sigma=0.0):
    """Return (loss closure, a_scale, gates) for `net` on this trajectory.
    gates is a dict of Adam stopping thresholds: data, vel, accel (None
    when there is no acceleration target)."""
    t0 = float(t[0, 0])
    use_accel = ACCEL_TARGET != "none"

    t_data = torch.tensor(t, dtype=DTYPE, device=DEVICE)
    x_data = torch.tensor(x, dtype=DTYPE, device=DEVICE)
    y_data = torch.tensor(y, dtype=DTYPE, device=DEVICE)
    vx_data = torch.tensor(vx, dtype=DTYPE, device=DEVICE)
    vy_data = torch.tensor(vy, dtype=DTYPE, device=DEVICE)

    n_ch = 4 if USE_VELOCITY else 2
    t_ic = torch.tensor([[t0]], dtype=DTYPE, device=DEVICE)
    ic_target = torch.tensor([[x[0, 0], y[0, 0], vx[0, 0], vy[0, 0]][:n_ch]],
                             dtype=DTYPE, device=DEVICE)
    # a noisy first sample is not a known initial state; W_IC would pin the
    # net to one noisy point with 1000x the weight of the others
    w_ic = W_IC if noise_sigma == 0 else 0.0

    targets = estimate_acceleration(t, x, y, vx, vy)
    if use_accel:
        ax_target = torch.tensor(targets[0], dtype=DTYPE, device=DEVICE)
        ay_target = torch.tensor(targets[1], dtype=DTYPE, device=DEVICE)
        a_scale = float(torch.mean(ax_target ** 2 + ay_target ** 2))
    else:
        # no target to train on; still need |a| for the relative gates
        t_f = t.flatten()
        ux, uy, order = (vx, vy, 1) if USE_VELOCITY else (x, y, 2)
        if noise_sigma > 0:
            # np.gradient on noisy samples carries ~sigma^2/dt^(2*order) and
            # would inflate a_scale by orders of magnitude; smooth first
            K = max([k for k in FourierFit.HARMONICS
                     if 2 * k + 3 <= max(len(t_f) // 8, 7)] or [2])
            ax_s = FourierFit(t_f, ux.flatten(), K)(t_f, order)
            ay_s = FourierFit(t_f, uy.flatten(), K)(t_f, order)
        else:
            ax_s, ay_s = ux.flatten(), uy.flatten()
            for _ in range(order):
                ax_s = np.gradient(ax_s, t_f)
                ay_s = np.gradient(ay_s, t_f)
        a_scale = float(np.mean(ax_s ** 2 + ay_s ** 2))

    # Adam stopping gates: the looser of absolute and relative, and never
    # below the noise floor of whatever the loss is compared against
    dt = float(np.median(np.diff(t.flatten())))
    noise_floor = NOISE_GATE_MULT * noise_sigma ** 2
    if ACCEL_TARGET in ("spline", "fd"):
        # an interpolating target through noisy samples carries ~sigma^2/dt^2
        # per differentiation; a Fourier target is smoothed, so ~sigma^2
        target_noise = 2 * noise_floor * (1 if USE_VELOCITY else 2) / dt ** 2
    else:
        target_noise = 2 * noise_floor
    # the data gate has to be the binding one: absolute 1e-3 never bound, so
    # whichever other gate happened to be loosest decided when Adam stopped
    data_scaled = max(PHASE_ONE_DATA_GATE_FLOOR,
                      min(PHASE_ONE_DATAERR_THRESHOLD,
                          PHASE_ONE_DATA_TIDAL_REL / max(a_scale, 1.0) ** 1.5))
    gates = dict(
        data=max(data_scaled, n_ch * noise_floor),
        # positions only: vel_loss compares the net against its own output, so
        # it measures internal tidiness, not fit, and cannot stop training
        vel=None if not USE_VELOCITY else max(
            PHASE_ONE_VEL_ABS_THRESHOLD, PHASE_ONE_VEL_REL_THRESHOLD * a_scale,
            2 * noise_floor),
        accel=None if not use_accel else max(
            PHASE_ONE_ACCEL_ABS_THRESHOLD, PHASE_ONE_ACCEL_REL_THRESHOLD * a_scale,
            target_noise),
    )

    def compute_loss():
        # data loss on the observed channels
        pred = net(t_data)
        data_loss = torch.mean((pred[:, 0:1] - x_data) ** 2) \
            + torch.mean((pred[:, 1:2] - y_data) ** 2)
        if USE_VELOCITY:
            data_loss = data_loss + torch.mean((pred[:, 2:3] - vx_data) ** 2) \
                + torch.mean((pred[:, 3:4] - vy_data) ** 2)

        # ic loss
        pred_ic = net(t_ic)[:, :n_ch]
        ic_loss = torch.mean((pred_ic - ic_target) ** 2)

        # acceleration matching: net's d(vx)/dt, d(vy)/dt vs finite-diff targets
        t_grad = t_data.clone().requires_grad_(True)
        state = net(t_grad)
        x_pred, y_pred = state[:, 0:1], state[:, 1:2]
        vx_pred, vy_pred = state[:, 2:3], state[:, 3:4]
        ones = torch.ones_like(vx_pred)
        dx_pred = torch.autograd.grad(x_pred, t_grad, grad_outputs=ones, create_graph=True)[0]
        dy_pred = torch.autograd.grad(y_pred, t_grad, grad_outputs=ones, create_graph=True)[0]
        ax_pred = torch.autograd.grad(vx_pred, t_grad, grad_outputs=ones, create_graph=True)[0]
        ay_pred = torch.autograd.grad(vy_pred, t_grad, grad_outputs=ones, create_graph=True)[0]
        # velocity match: d(x)/dt vs observed v, or vs the net's own v
        # output when v is not observed (positions only)
        if USE_VELOCITY:
            vel_loss = torch.mean((dx_pred - vx_data) ** 2) + torch.mean((dy_pred - vy_data) ** 2)
        else:
            vel_loss = torch.mean((dx_pred - vx_pred) ** 2) + torch.mean((dy_pred - vy_pred) ** 2)
        if use_accel:
            accel_loss = torch.mean((ax_pred - ax_target) ** 2) + torch.mean((ay_pred - ay_target) ** 2)
        else:
            accel_loss = torch.zeros((), dtype=DTYPE, device=DEVICE)
        w = compute_loss.weights
        total = (w["ic"] * ic_loss + w["data"] * data_loss
                 + w["accel"] * accel_loss + w["vel"] * vel_loss)
        # the untouched component tensors, for callers that rebalance w
        compute_loss.parts = dict(ic=ic_loss, data=data_loss,
                                  accel=accel_loss, vel=vel_loss)
        return total, ic_loss.item(), data_loss.item(), accel_loss.item(), vel_loss.item()

    compute_loss.weights = dict(ic=w_ic, data=W_DATA, accel=W_ACCEL, vel=W_VEL)
    return compute_loss, a_scale, gates


def lbfgs_phaseone(net, compute_loss, iters, traj_idx):
    """L-BFGS polish of a phase-1 net. Returns (data_l, accel_l, vel_l, seconds)."""
    params = list(net.parameters())
    # default tolerances (1e-7 grad, 1e-9 change) stop it after a handful of
    # steps at these loss scales; disable them so it runs the iterations asked
    lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=iters, history_size=50,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        lbfgs.zero_grad()
        loss, *_ = compute_loss()
        loss.backward()
        return loss

    t_start_time = time.time()
    lbfgs.step(closure)
    elapsed = time.time() - t_start_time
    _, ic_l, data_l, accel_l, vel_l = compute_loss()
    print(f"  [traj {traj_idx}] phase1 L-BFGS ({iters} it, {elapsed:.1f}s): "
          f"ic={ic_l:.2e} data={data_l:.2e} accel={accel_l:.2e} vel={vel_l:.2e}")
    return data_l, accel_l, vel_l, elapsed


def train_phaseone(t, x, y, vx, vy, traj_idx, noise_sigma=0.0):
    net = FNN().to(DEVICE)
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=ADAM_LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)
    compute_loss, a_scale, gates = build_phaseone_loss(net, t, x, y, vx, vy, noise_sigma)
    # both derivative gates scale with the acceleration power: it is the
    # sharpness of the orbit, not the size of v, that sets how well a smooth
    # net can match d(x)/dt (641/751 capped on the vel gate with v_scale ~2)
    accel_gate = gates["accel"] if gates["accel"] is not None else float("inf")
    vel_gate = gates["vel"] if gates["vel"] is not None else float("inf")
    data_gate = gates["data"]

    it = 0
    data_l = float("inf")
    accel_l = float("inf")
    vel_l = float("inf")
    t_start_time = time.time()
    while (data_l > data_gate
           or accel_l > accel_gate
           or vel_l > vel_gate) and it < PHASE_ONE_ITERS:
        optimizer.zero_grad()
        loss, ic_l, data_l, accel_l, vel_l = compute_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()
        it += 1
        if it % DISPLAY_EVERY == 0:
            print(f"  [traj {traj_idx}] phase1 it={it:6d} loss={loss.item():.4e} "
                  f"ic={ic_l:.2e} data={data_l:.2e} accel={accel_l:.2e} vel={vel_l:.2e} "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t_start_time
    print(f"  [traj {traj_idx}] phase one Adam done in {elapsed:.1f}s, iters={it}, "
          f"data_l={data_l:.4e} (gate {data_gate:.1e}), accel_l={accel_l:.4e} (gate {accel_gate:.1e}), "
          f"vel_l={vel_l:.4e} (gate {vel_gate:.1e}), rms|a|={a_scale ** 0.5:.3g}")

    hit_cap = it >= PHASE_ONE_ITERS
    if PHASE_ONE_LBFGS_ITERS > 0:
        data_l, accel_l, vel_l, _ = lbfgs_phaseone(net, compute_loss, PHASE_ONE_LBFGS_ITERS, traj_idx)
    if SAVE_NETS:
        torch.save(net.state_dict(), f"net_traj{traj_idx}.pt")
    return net, data_l, accel_l, vel_l, it, hit_cap


# ---------------------------------------------------------------------------
# Phase two: PERMANENT freeze -- mu is the only trainable variable
# ---------------------------------------------------------------------------
def train_phasetwo(net, t, mu_true, traj_idx):
    t_data = torch.tensor(t, dtype=DTYPE, device=DEVICE)

    # frozen for the ENTIRE phase; there is no unfreeze anywhere below
    for p in net.parameters():
        p.requires_grad = False

    # --- global scan of the frozen net's residual over a log-spaced mu grid.
    # The residual is non-convex in mu (a barrier sits between small mu and
    # the truth for most mu >= 0.15), so descent from a fixed guess lands in
    # the wrong basin; the scan argmin is used as the starting point below.
    mu_grid = np.logspace(MU_LOG_LO, np.log10(0.5), SCAN_POINTS)
    scan_losses = []
    for m in mu_grid:
        m_t = torch.tensor(m, dtype=DTYPE, device=DEVICE)
        rx, ry, rvx, rvy = physics_residual(net, m_t, t_data)
        scan_losses.append((torch.mean(rx**2) + torch.mean(ry**2)
                            + torch.mean(rvx**2) + torch.mean(rvy**2)).item())
    scan_losses = np.array(scan_losses)
    scan_argmin = float(mu_grid[scan_losses.argmin()])
    scan_contrast = float(scan_losses.max() / scan_losses.min())
    print(f"  [traj {traj_idx}] NET SCAN argmin_mu={scan_argmin:.6g} "
          f"true={mu_true:.6g} min={scan_losses.min():.4e} max={scan_losses.max():.4e} "
          f"contrast={scan_contrast:.3e}")

    mu_raw = torch.tensor(inv_sigmoid_init(scan_argmin), dtype=DTYPE, device=DEVICE,
                          requires_grad=True)
    optimizer = torch.optim.Adam([mu_raw], lr=PHASE_TWO_ADAM_LR)

    # --- one-off diagnostic: which residual component dominates at the start?
    with torch.no_grad():
        mu_diag = mu_from_raw(mu_raw)
    res_x, res_y, res_vx, res_vy = physics_residual(net, mu_diag, t_data)
    print(f"  [traj {traj_idx}] RESIDUAL SPLIT at mu={mu_diag.item():.6g}  "
          f"res_x={torch.mean(res_x**2).item():.4e}  res_y={torch.mean(res_y**2).item():.4e}  "
          f"res_vx={torch.mean(res_vx**2).item():.4e}  res_vy={torch.mean(res_vy**2).item():.4e}")
    mu_history = []

    def compute_phys_loss():
        mu = mu_from_raw(mu_raw)
        # t_data only: no random collocation resampling (it caused oscillation)
        res_x, res_y, res_vx, res_vy = physics_residual(net, mu, t_data)
        loss = torch.mean(res_x ** 2) + torch.mean(res_y ** 2) \
            + torch.mean(res_vx ** 2) + torch.mean(res_vy ** 2)
        return loss, mu

    t_start_time = time.time()

    # ---- Adam on mu alone ----
    for it in range(PHASE_TWO_ADAM_ITERS):
        optimizer.zero_grad()
        phys_loss, mu = compute_phys_loss()
        phys_loss.backward()
        optimizer.step()

        if it % MU_LOG_EVERY == 0:
            mu_history.append((it, mu.item(), phys_loss.item()))
        if it % DISPLAY_EVERY == 0:
            print(f"  [traj {traj_idx}] phase2 it={it:6d} "
                  f"phys={phys_loss.item():.4e} mu={mu.item():.5f}")

    # ---- L-BFGS refinement on mu alone ----
    lbfgs = torch.optim.LBFGS([mu_raw], lr=1.0, max_iter=LBFGS_ITERS,
                               history_size=50, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        loss, _ = compute_phys_loss()
        loss.backward()
        return loss

    lbfgs.step(closure)

    with torch.no_grad():
        mu_final = mu_from_raw(mu_raw).item()
    mu_history.append((PHASE_TWO_ADAM_ITERS, mu_final, float("nan")))

    elapsed = time.time() - t_start_time
    print(f"  [traj {traj_idx}] phase two done in {elapsed:.1f}s, "
          f"mu_true={mu_true:.6f} mu_pred={mu_final:.6f}")

    with open(f"mu_history_traj{traj_idx}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "mu", "phys_loss"])
        w.writerows(mu_history)

    return mu_final, scan_argmin, scan_contrast


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(dataset_path, traj_indices, results_path="results.csv",
              n_points=None, noise_sigma=0.0, noise_seed=0):
    # Columns grouped by method: geometry, then PINN, classical scan,
    # shooting, EKF, spline, Fourier. Each group orders mu, error_pct,
    # abs_error, time_s, then that method's own diagnostics. n_points and
    # noise_sigma thin and perturb every trajectory (perturb_trajectory);
    # the per-trajectory noise seed is noise_seed + traj_index.
    with open(results_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "traj_index", "mu_true", "min_r",
            "pinn_mu", "pinn_error_pct", "pinn_abs_error", "pinn_time_s",
            "pinn_phase1_iters", "pinn_phase1_hit_iter_cap",
            "pinn_phase1_data_loss", "pinn_phase1_accel_loss", "pinn_phase1_vel_loss",
            "pinn_scan_argmin", "pinn_scan_contrast",
            "classical_mu", "classical_error_pct", "classical_abs_error", "classical_time_s",
            "classical_contrast",
            "shooting_mu", "shooting_error_pct", "shooting_abs_error", "shooting_time_s",
            "shooting_success", "shooting_nfev",
            "ekf_mu", "ekf_error_pct", "ekf_abs_error", "ekf_time_s",
            "spline_mu", "spline_error_pct", "spline_abs_error", "spline_time_s",
            "fourier_mu", "fourier_error_pct", "fourier_abs_error", "fourier_time_s",
            "fourier_harmonics",
            "n_points", "noise_sigma", "use_velocity", "accel_target",
        ])

    for traj_idx in traj_indices:
        try:
            set_seed(SEED)
            t, x, y, vx, vy, mu_true = load_trajectory(dataset_path, traj_idx)
            min_r = min_primary_distance(x, y, mu_true)
            t, x, y, vx, vy = perturb_trajectory(t, x, y, vx, vy, n_points, noise_sigma,
                                                 seed=noise_seed + traj_idx)

            pinn_t0 = time.time()
            net, p1_data_loss, p1_accel_loss, p1_vel_loss, p1_iters, p1_hit_cap = \
                train_phaseone(t, x, y, vx, vy, traj_idx, noise_sigma)
            pinn_mu, pinn_scan_argmin, pinn_scan_contrast = \
                train_phasetwo(net, t, mu_true, traj_idx)
            pinn_time_s = time.time() - pinn_t0
            pinn_error_pct = abs(pinn_mu - mu_true) / mu_true * 100
            pinn_abs_error = abs(pinn_mu - mu_true)

            classical_t0 = time.time()
            classical_mu, classical_contrast = classical_scan(t, x, y, vx, vy)
            classical_time_s = time.time() - classical_t0
            classical_error_pct = abs(classical_mu - mu_true) / mu_true * 100
            classical_abs_error = abs(classical_mu - mu_true)

            shooting = estimate_mu_shooting(t, x, y, vx, vy, use_velocity=USE_VELOCITY)
            shooting_mu = shooting["mu_pred"]
            shooting_error_pct = abs(shooting_mu - mu_true) / mu_true * 100
            shooting_abs_error = abs(shooting_mu - mu_true)

            ekf = estimate_mu_ekf(t, x, y, vx, vy, use_velocity=USE_VELOCITY)
            ekf_mu = ekf["mu_pred"]
            ekf_error_pct = abs(ekf_mu - mu_true) / mu_true * 100
            ekf_abs_error = abs(ekf_mu - mu_true)

            spline = estimate_mu_spline(t, x, y, vx, vy, use_velocity=USE_VELOCITY)
            spline_mu = spline["mu_pred"]
            spline_error_pct = abs(spline_mu - mu_true) / mu_true * 100
            fourier = estimate_mu_fourier(t, x, y, vx, vy, use_velocity=USE_VELOCITY)
            fourier_mu = fourier["mu_pred"]
            fourier_error_pct = abs(fourier_mu - mu_true) / mu_true * 100

            print(f"traj {traj_idx} | true={mu_true:.6f} "
                  f"pinn={pinn_mu:.6f} ({pinn_error_pct:.2f}%, {pinn_time_s:.1f}s) "
                  f"classical={classical_mu:.6f} ({classical_error_pct:.2f}%, {classical_time_s:.1f}s) "
                  f"shooting={shooting_mu:.6f} ({shooting_error_pct:.2f}%, {shooting['elapsed_s']:.1f}s) "
                  f"ekf={ekf_mu:.6f} ({ekf_error_pct:.2f}%, {ekf['elapsed_s']:.1f}s) "
                  f"spline={spline_mu:.6f} ({spline_error_pct:.2g}%) "
                  f"fourier={fourier_mu:.6f} ({fourier_error_pct:.2g}%, K={fourier['harmonics']}) "
                  f"min_r={min_r:.4f}")

            with open(results_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    traj_idx, mu_true, min_r,
                    pinn_mu, pinn_error_pct, pinn_abs_error, pinn_time_s,
                    p1_iters, p1_hit_cap,
                    p1_data_loss, p1_accel_loss, p1_vel_loss,
                    pinn_scan_argmin, pinn_scan_contrast,
                    classical_mu, classical_error_pct, classical_abs_error, classical_time_s,
                    classical_contrast,
                    shooting_mu, shooting_error_pct, shooting_abs_error, shooting["elapsed_s"],
                    shooting["success"], shooting["nfev"],
                    ekf_mu, ekf_error_pct, ekf_abs_error, ekf["elapsed_s"],
                    spline_mu, spline_error_pct, abs(spline_mu - mu_true), spline["elapsed_s"],
                    fourier_mu, fourier_error_pct, abs(fourier_mu - mu_true),
                    fourier["elapsed_s"], fourier["harmonics"],
                    len(t), noise_sigma, USE_VELOCITY, ACCEL_TARGET,
                ])
        except Exception as e:
            print(f"traj {traj_idx} FAILED: {e}")
            continue


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    # Sanity check on a few trajectories: indices = [646, 821, 626, 740]
    indices = list(range(1, 1000, 5))
    run_sweep("cr3bp_dataset_final.npy", indices)
