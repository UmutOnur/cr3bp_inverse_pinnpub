"""One orbit end to end with the current model: noisy positions in, mu out.
  python demo.py               orbit 646 (about 3 min on a GPU)
  python demo.py 821           any orbit index 0-999
  python demo.py 646 --quick   short training, only checks the installation
Output files (loss weights, mu history) go to demo_output/.
"""
import os
import sys
from pathlib import Path

import train

train.USE_VELOCITY = False              # positions only
train.ACCEL_TARGET = "none"             # no manufactured acceleration target
train.PHASE_ONE_WEIGHTING = "gradnorm"  # loss weights from gradient norms
train.PHASE_ONE_ITERS = 40000           # Adam cap, then 2500 L-BFGS steps
train.SAVE_NETS = False
N_POINTS, SIGMA = 1000, 1e-3            # samples kept, position noise
DATA = Path(__file__).resolve().parent / "cr3bp_dataset_final.npy"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
idx = int(args[0]) if args else 646
if "--quick" in sys.argv:
    train.PHASE_ONE_ITERS, train.PHASE_ONE_LBFGS_ITERS = 300, 10
os.makedirs("demo_output", exist_ok=True)
os.chdir("demo_output")

train.set_seed(train.SEED)
t, x, y, vx, vy, mu_true = train.load_trajectory(str(DATA), idx)
t, x, y, vx, vy = train.perturb_trajectory(t, x, y, vx, vy, N_POINTS, SIGMA, seed=1000 * idx + 4)

# phase 1: smooth curve through the noisy positions (no mu, no physics)
net, *_ = train.train_phaseone(t, x, y, vx, vy, idx, SIGMA)
# phase 2: net frozen, mu from the equations of motion (mu_true is used only in prints)
mu_pred, _, contrast = train.train_phasetwo(net, t, mu_true, idx)

err = abs(mu_pred - mu_true)
print(f"\norbit {idx}: true mu {mu_true:.6g}, estimate {mu_pred:.6g}, "
      f"error {err / mu_true * 100:.3g}% (abs {err:.2g}), scan contrast {contrast:.3g}")
