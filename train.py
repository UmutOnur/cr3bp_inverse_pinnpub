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

from baselines import classical_scan, estimate_mu_shooting, estimate_mu_ekf

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
PHASE_ONE_ACCELERR_THRESHOLD = 1e-6
PHASE_ONE_VELERR_THRESHOLD   = 1e-6
PHASE_ONE_DATAERR_THRESHOLD = 1e-3
PHASE_ONE_ITERS = 100000
W_IC = 10.0
W_DATA = 10.0
W_ACCEL = 5.0
W_VEL = 5.0
GRAD_CLIP_NORM = 5.0
DISPLAY_EVERY = 2000
MU_LOG_EVERY = 500

# phase two (permanent freeze: mu is the only trainable variable)
PHASE_TWO_ADAM_ITERS = 6000
PHASE_TWO_ADAM_LR = 1e-2      # single scalar -- a higher LR is fine here
LBFGS_ITERS = 1000


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


def estimate_acceleration(t, vx, vy):
    """Finite-difference acceleration targets from raw data. No mu needed."""
    t_flat = t.flatten()
    ax = np.gradient(vx.flatten(), t_flat).reshape(-1, 1)
    ay = np.gradient(vy.flatten(), t_flat).reshape(-1, 1)
    return ax, ay


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
    """Bounded reparametrization: mu always strictly in (0, 0.5)."""
    return 0.5 * torch.sigmoid(mu_raw)


def inv_sigmoid_init(mu0, lo=0.0, hi=0.5):
    """Pick mu_raw so that mu_from_raw(mu_raw) == mu0, for warm starts."""
    p = (mu0 - lo) / (hi - lo)
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

    r1 = torch.sqrt((x + mu) ** 2 + y ** 2 + 1e-6)
    r2 = torch.sqrt((x - (1 - mu)) ** 2 + y ** 2 + 1e-6)

    res_x = dx - vx
    res_y = dy - vy
    res_vx = dvx - (x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy)
    res_vy = dvy - (y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx)
    return res_x, res_y, res_vx, res_vy


# ---------------------------------------------------------------------------
# Phase one: fit the trajectory WITH the acceleration constraint
# ---------------------------------------------------------------------------
def train_phaseone(t, x, y, vx, vy, traj_idx):
    t0, t_end = float(t[0, 0]), float(t[-1, 0])

    t_data = torch.tensor(t, dtype=DTYPE, device=DEVICE)
    x_data = torch.tensor(x, dtype=DTYPE, device=DEVICE)
    y_data = torch.tensor(y, dtype=DTYPE, device=DEVICE)
    vx_data = torch.tensor(vx, dtype=DTYPE, device=DEVICE)
    vy_data = torch.tensor(vy, dtype=DTYPE, device=DEVICE)

    t_ic = torch.tensor([[t0]], dtype=DTYPE, device=DEVICE)
    ic_target = torch.tensor([[x[0, 0], y[0, 0], vx[0, 0], vy[0, 0]]], dtype=DTYPE, device=DEVICE)

    # finite-difference acceleration targets (no mu involved)
    ax_np, ay_np = estimate_acceleration(t, vx, vy)
    ax_target = torch.tensor(ax_np, dtype=DTYPE, device=DEVICE)
    ay_target = torch.tensor(ay_np, dtype=DTYPE, device=DEVICE)

    net = FNN().to(DEVICE)
    params = list(net.parameters())
    optimizer = torch.optim.Adam(params, lr=ADAM_LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)

    def compute_loss():
        # data loss
        pred = net(t_data)
        data_loss = torch.mean((pred[:, 0:1] - x_data) ** 2) \
            + torch.mean((pred[:, 1:2] - y_data) ** 2) \
            + torch.mean((pred[:, 2:3] - vx_data) ** 2) \
            + torch.mean((pred[:, 3:4] - vy_data) ** 2)

        # ic loss
        pred_ic = net(t_ic)
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
        vel_loss = torch.mean((dx_pred - vx_data) ** 2) + torch.mean((dy_pred - vy_data) ** 2)
        accel_loss = torch.mean((ax_pred - ax_target) ** 2) + torch.mean((ay_pred - ay_target) ** 2)
        total = W_IC * ic_loss + W_DATA * data_loss + W_ACCEL * accel_loss + W_VEL * vel_loss
        return total, ic_loss.item(), data_loss.item(), accel_loss.item(), vel_loss.item()

    it = 0
    data_l = float("inf")
    accel_l = float("inf")
    vel_l = float("inf")
    t_start_time = time.time()
    while (data_l > PHASE_ONE_DATAERR_THRESHOLD
           or accel_l > PHASE_ONE_ACCELERR_THRESHOLD
           or vel_l > PHASE_ONE_VELERR_THRESHOLD) and it < PHASE_ONE_ITERS:
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
    print(f"  [traj {traj_idx}] phase one done in {elapsed:.1f}s, iters={it}, "
          f"final data_l={data_l:.4e}, final accel_l={accel_l:.4e}, final vel_l={vel_l:.4e}")

    hit_cap = it >= PHASE_ONE_ITERS
    return net, data_l, accel_l, vel_l, it, hit_cap


# ---------------------------------------------------------------------------
# Phase two: PERMANENT freeze -- mu is the only trainable variable
# ---------------------------------------------------------------------------
def train_phasetwo(net, t, mu_true, traj_idx):
    t_data = torch.tensor(t, dtype=DTYPE, device=DEVICE)

    # frozen for the ENTIRE phase; there is no unfreeze anywhere below
    for p in net.parameters():
        p.requires_grad = False

    mu_raw = torch.tensor(inv_sigmoid_init(0.05), dtype=DTYPE, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([mu_raw], lr=PHASE_TWO_ADAM_LR)
    # --- one-off diagnostic: which residual component dominates? ---
    with torch.no_grad():
        mu_diag = mu_from_raw(mu_raw)
    res_x, res_y, res_vx, res_vy = physics_residual(net, mu_diag, t_data)
    print(f"  [traj {traj_idx}] RESIDUAL SPLIT at mu={mu_diag.item():.5f}  "
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

    # --- one-off diagnostic: where is the frozen net's residual minimum? ---
    mu_grid = np.linspace(0.001, 0.499, 300)
    scan_losses = []
    for m in mu_grid:
        m_t = torch.tensor(m, dtype=DTYPE, device=DEVICE)
        rx, ry, rvx, rvy = physics_residual(net, m_t, t_data)
        scan_losses.append((torch.mean(rx**2) + torch.mean(ry**2)
                            + torch.mean(rvx**2) + torch.mean(rvy**2)).item())
    scan_losses = np.array(scan_losses)
    scan_argmin = float(mu_grid[scan_losses.argmin()])
    scan_contrast = float(scan_losses.max() / scan_losses.min())
    print(f"  [traj {traj_idx}] NET SCAN argmin_mu={scan_argmin:.5f} "
          f"true={mu_true:.5f} min={scan_losses.min():.4e} max={scan_losses.max():.4e} "
          f"contrast={scan_contrast:.3e}")
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
def run_sweep(dataset_path, traj_indices, results_path="results.csv"):
    # Columns grouped by method: geometry, then PINN, classical scan,
    # shooting, EKF. Each group orders mu, error_pct, abs_error, time_s,
    # then that method's own diagnostics.
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
        ])

    for traj_idx in traj_indices:
        try:
            set_seed(SEED)
            t, x, y, vx, vy, mu_true = load_trajectory(dataset_path, traj_idx)
            min_r = min_primary_distance(x, y, mu_true)

            pinn_t0 = time.time()
            net, p1_data_loss, p1_accel_loss, p1_vel_loss, p1_iters, p1_hit_cap = \
                train_phaseone(t, x, y, vx, vy, traj_idx)
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

            shooting = estimate_mu_shooting(t, x, y, vx, vy)
            shooting_mu = shooting["mu_pred"]
            shooting_error_pct = abs(shooting_mu - mu_true) / mu_true * 100
            shooting_abs_error = abs(shooting_mu - mu_true)

            ekf = estimate_mu_ekf(t, x, y, vx, vy)
            ekf_mu = ekf["mu_pred"]
            ekf_error_pct = abs(ekf_mu - mu_true) / mu_true * 100
            ekf_abs_error = abs(ekf_mu - mu_true)

            print(f"traj {traj_idx} | true={mu_true:.6f} "
                  f"pinn={pinn_mu:.6f} ({pinn_error_pct:.2f}%, {pinn_time_s:.1f}s) "
                  f"classical={classical_mu:.6f} ({classical_error_pct:.2f}%, {classical_time_s:.1f}s) "
                  f"shooting={shooting_mu:.6f} ({shooting_error_pct:.2f}%, {shooting['elapsed_s']:.1f}s) "
                  f"ekf={ekf_mu:.6f} ({ekf_error_pct:.2f}%, {ekf['elapsed_s']:.1f}s) "
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
                ])
        except Exception as e:
            print(f"traj {traj_idx} FAILED: {e}")
            continue


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    # Sanity check on a few trajectories: indices = [646, 821, 626, 740]
    indices = list(range(1, 1000, 5))
    run_sweep("cr3bp_dataset_final.npy", indices)
