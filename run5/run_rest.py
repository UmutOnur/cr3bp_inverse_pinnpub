"""Finish the 18 run5 orbits that the 2026-09-15 reboot skipped.
Waits for the run6 probe to finish before using the GPU. Writes
results_rest.csv (extra columns vs results.csv: spline, Fourier,
perturbation settings); join the two on traj_index."""
import os
import sys
import time
sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train

PROBE_LOG = "E:/cr3bp_inverse_pinn/run6/probe.log"
DONE = "run6 probe: done"
indices = [381, 751, 266, 586, 641, 16, 131, 126, 841, 236, 606, 406, 836, 256, 396, 621, 611, 651]

while True:
    if os.path.exists(PROBE_LOG) and DONE in open(PROBE_LOG, encoding="utf-8", errors="ignore").read():
        break
    print("run5 rest: waiting for probe", time.strftime("%H:%M:%S"), flush=True)
    time.sleep(60)

print(f"run5 rest: {len(indices)} orbits, device {train.DEVICE}, "
      f"use_velocity={train.USE_VELOCITY} accel_target={train.ACCEL_TARGET} "
      f"lbfgs {train.PHASE_ONE_LBFGS_ITERS}", flush=True)
train.run_sweep("E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy", indices,
                results_path="results_rest.csv")
print("run5 rest: done", flush=True)
