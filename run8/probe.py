"""run8 probe: the phase-1 data-gate fix, on the two regression orbits.

Cells R/A/B/F from run6, re-run with run7's noise seeds (1000*idx + sigma
index) so every row joins straight onto run7/raw.csv and needs no baseline
of its own. R is the clean full-state regression check; A, B and F are the
position-only cells whose run6 numbers came from under-trained nets."""
import csv
import sys
import time

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train

DATA = "E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy"
ORBITS = [646, 821]
SIGMAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)      # run7 order; index sets the seed
# name, use_velocity, accel_target, n_points, sigma
CELLS = [
    ("R_full_spline_1000_0",  True,  "spline", 1000, 0.0),
    ("A_pos_none_1000_1e-5",  False, "none",   1000, 1e-5),
    ("B_pos_none_200_1e-4",   False, "none",   200,  1e-4),
    ("F_pos_none_1000_1e-3",  False, "none",   1000, 1e-3),
]
train.PHASE_ONE_ITERS = 40000               # bound the probe; report cap hits
train.SAVE_NETS = False

COLS = ["cell", "traj_index", "mu_true", "min_r", "use_velocity", "accel_target",
        "n_points", "sigma", "noise_seed", "mu_pred", "error_pct", "abs_error",
        "phase1_iters", "phase1_hit_cap", "phase1_data_l", "phase1_data_gate",
        "phase1_accel_l", "phase1_vel_l", "scan_argmin", "scan_contrast",
        "elapsed_s"]
out = "results.csv"
with open(out, "w", newline="") as f:
    csv.writer(f).writerow(COLS)

print(f"run8 probe: {len(CELLS)} cells x {ORBITS}, device {train.DEVICE}, "
      f"iter cap {train.PHASE_ONE_ITERS}", flush=True)
for name, use_v, target, n, sigma in CELLS:
    train.USE_VELOCITY = use_v
    train.ACCEL_TARGET = target
    for idx in ORBITS:
        seed = 1000 * idx + SIGMAS.index(sigma)
        print(f"=== {name} traj {idx} (seed {seed}) ===", flush=True)
        t0 = time.time()
        try:
            train.set_seed(train.SEED)
            t, x, y, vx, vy, mu_true = train.load_trajectory(DATA, idx)
            min_r = train.min_primary_distance(x, y, mu_true)
            t, x, y, vx, vy = train.perturb_trajectory(t, x, y, vx, vy, n, sigma,
                                                       seed=seed)
            # the gate the run is actually being stopped on, for the record
            net0 = train.FNN().to(train.DEVICE)
            gate = train.build_phaseone_loss(net0, t, x, y, vx, vy, sigma)[2]["data"]

            net, data_l, accel_l, vel_l, iters, hit_cap = \
                train.train_phaseone(t, x, y, vx, vy, idx, sigma)
            mu_pred, argmin, contrast = train.train_phasetwo(net, t, mu_true, idx)
            row = [name, idx, mu_true, min_r, use_v, target, n, sigma, seed,
                   mu_pred, abs(mu_pred - mu_true) / mu_true * 100,
                   abs(mu_pred - mu_true), iters, hit_cap, data_l, gate,
                   accel_l, vel_l, argmin, contrast, time.time() - t0]
        except Exception as exc:                       # keep the rest of the run
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            row = [name, idx, "", "", use_v, target, n, sigma, seed,
                   "", "", "", "", "", "", "", "", "", "", "", time.time() - t0]
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
print("run8 probe: done", flush=True)
