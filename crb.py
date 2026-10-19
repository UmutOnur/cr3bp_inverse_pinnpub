"""Cramer-Rao bound for mu: the smallest standard
deviation any unbiased estimator can reach from n = 1000 position samples with
noise sigma = 1e-3. Fisher information from the variational equations
(sensitivities of x, y to x0, y0, vx0, vy0, mu), evaluated at the true values.
Checked against finite differences on 646 and 821 (agreement 5e-7 or better).
Run directly for the 599 orbits in results/: writes crb.csv to the current
folder (CPU, a few minutes on 7 workers)."""
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent
SIGMA = 1e-3
D = np.load(ROOT / "cr3bp_dataset_final.npy")
B = np.r_[0, np.where(np.diff(D[:, 6]) < 0)[0] + 1, len(D)]


def rhs(t, u, mu):
    """State (4) plus its 4x5 sensitivity matrix to (x0, y0, vx0, vy0, mu)."""
    x, y, vx, vy = u[:4]
    S = u[4:].reshape(4, 5)
    xp, xm = x + mu, x - 1 + mu
    r1, r2 = np.hypot(xp, y), np.hypot(xm, y)
    i13, i23, i15, i25 = r1**-3, r2**-3, r1**-5, r2**-5
    ax = x - (1 - mu) * xp * i13 - mu * xm * i23 + 2 * vy
    ay = y - (1 - mu) * y * i13 - mu * y * i23 - 2 * vx
    oxx = 1 - (1 - mu) * i13 + 3 * (1 - mu) * xp**2 * i15 - mu * i23 + 3 * mu * xm**2 * i25
    oyy = 1 - (1 - mu) * i13 + 3 * (1 - mu) * y**2 * i15 - mu * i23 + 3 * mu * y**2 * i25
    oxy = 3 * (1 - mu) * xp * y * i15 + 3 * mu * xm * y * i25
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [oxx, oxy, 0, 2], [oxy, oyy, -2, 0]])
    dax = xp * i13 - (1 - mu) * i13 + 3 * (1 - mu) * xp**2 * i15 - xm * i23 - mu * i23 + 3 * mu * xm**2 * i25
    day = y * i13 + 3 * (1 - mu) * y * xp * i15 - y * i23 + 3 * mu * y * xm * i25
    dS = A @ S
    dS[2, 4] += dax
    dS[3, 4] += day
    return np.concatenate([[vx, vy, ax, ay], dS.ravel()])


def job(i):
    s = D[B[i]:B[i + 1]]
    t, mu = s[:, 6], s[0, 7]
    u0 = np.concatenate([s[0, [0, 1, 3, 4]], np.hstack([np.eye(4), np.zeros((4, 1))]).ravel()])
    t0 = time.time()
    sol = solve_ivp(rhs, (t[0], t[-1]), u0, t_eval=t, args=(mu,), method="DOP853", rtol=1e-11, atol=1e-12)
    if not sol.success:
        return i, mu, np.nan, np.nan, time.time() - t0
    pos_err = np.abs(sol.y[:2].T - s[:, :2]).max()
    S = sol.y[4:].T.reshape(-1, 4, 5)[:, :2, :].reshape(-1, 5)
    scale = np.linalg.norm(S, axis=0)          # column scaling keeps the inverse well conditioned
    F = (S / scale).T @ (S / scale) / SIGMA**2
    cov = np.linalg.pinv(F) / np.outer(scale, scale)
    return i, mu, float(np.sqrt(cov[4, 4])), pos_err, time.time() - t0


if __name__ == "__main__":
    with Pool(7) as p, open("crb.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_index", "mu_true", "crb_sd_mu", "reint_pos_err", "seconds"])
        for row in p.imap_unordered(job, [i for i in range(1000) if i % 5 in (0, 1, 3) and i != 938]):
            w.writerow(row)
