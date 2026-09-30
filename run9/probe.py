"""run9 probe: the two things run8 left open.

Cell G — the close-approach orbits the tidally-scaled data gate was built
for. 236 and 606 failed phase 1 in run5 (data_l ~5e-6 after L-BFGS, mu
errors 99.9% and 21.8%); 641 and 751 passed at a slightly larger min_r and
are the controls. Full-state, spline target, clean, so the only difference
from run5 is the gate.

Cell H — the two clean position-only no-target cells that complete the
method table. run8 gave 3.5-4.4% on 646 at every noise level with a nearly
constant absolute error, which says floor rather than noise response; this
is the test.

Clean data throughout, so run7's sigma=0 rows join straight on (its seed
index for sigma=0 is 0), and run5 is the full-state control for cell G.
"""
import csv
import sys
import time

sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train

DATA = "E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy"
# name, orbits, use_velocity, accel_target, n_points, sigma
CELLS = [
    ("G_close_full_spline_1000_0", [236, 606, 641, 751], True,  "spline", 1000, 0.0),
    ("H_pos_none_1000_0",          [646, 821],           False, "none",   1000, 0.0),
]
train.PHASE_ONE_ITERS = 40000          # same bound as run8; report cap hits
train.SAVE_NETS = True                 # phase-2 follow-ups then cost 5 s

COLS = ["cell", "traj_index", "mu_true", "min_r", "use_velocity", "accel_target",
        "n_points", "sigma", "noise_seed", "mu_pred", "error_pct", "abs_error",
        "phase1_iters", "phase1_hit_cap", "phase1_data_l", "phase1_data_gate",
        "phase1_accel_l", "phase1_vel_l", "rms_a", "scan_argmin", "scan_contrast",
        "elapsed_s"]
out = "results.csv"
with open(out, "w", newline="") as f:
    csv.writer(f).writerow(COLS)

print(f"run9 probe: {sum(len(c[1]) for c in CELLS)} runs, device {train.DEVICE}, "
      f"iter cap {train.PHASE_ONE_ITERS}", flush=True)
for name, orbits, use_v, target, n, sigma in CELLS:
    train.USE_VELOCITY = use_v
    train.ACCEL_TARGET = target
    for idx in orbits:
        seed = 1000 * idx                      # run7's seed for sigma = 0
        print(f"=== {name} traj {idx} ===", flush=True)
        t0 = time.time()
        try:
            train.set_seed(train.SEED)
            t, x, y, vx, vy, mu_true = train.load_trajectory(DATA, idx)
            min_r = train.min_primary_distance(x, y, mu_true)
            t, x, y, vx, vy = train.perturb_trajectory(t, x, y, vx, vy, n, sigma,
                                                       seed=seed)
            # the gate the run is actually stopped on, and the |a| that set it
            net0 = train.FNN().to(train.DEVICE)
            _, a_scale, gates = train.build_phaseone_loss(net0, t, x, y, vx, vy, sigma)

            net, data_l, accel_l, vel_l, iters, hit_cap = \
                train.train_phaseone(t, x, y, vx, vy, idx, sigma)
            mu_pred, argmin, contrast = train.train_phasetwo(net, t, mu_true, idx)
            row = [name, idx, mu_true, min_r, use_v, target, n, sigma, seed,
                   mu_pred, abs(mu_pred - mu_true) / mu_true * 100,
                   abs(mu_pred - mu_true), iters, hit_cap, data_l, gates["data"],
                   accel_l, vel_l, a_scale ** 0.5, argmin, contrast,
                   time.time() - t0]
        except Exception as exc:                       # keep the rest of the run
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            row = [name, idx, "", "", use_v, target, n, sigma, seed] + [""] * 12 + \
                  [time.time() - t0]
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
print("run9 probe: done", flush=True)
