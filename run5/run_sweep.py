import sys; sys.path.insert(0, "E:/cr3bp_inverse_pinn")
import train
indices = [int(l) for l in open("indices.txt") if l.strip()]
print(f"run5: {len(indices)} orbits, device {train.DEVICE}, lbfgs {train.PHASE_ONE_LBFGS_ITERS}, "
      f"accel target {train.ACCEL_TARGET}, gates abs {train.PHASE_ONE_ACCEL_ABS_THRESHOLD} rel {train.PHASE_ONE_ACCEL_REL_THRESHOLD}", flush=True)
train.run_sweep("E:/cr3bp_inverse_pinn/cr3bp_dataset_final.npy", indices, results_path="results.csv")
print("run5: done", flush=True)
