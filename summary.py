"""Results table: how many orbits each method gets under 10% and under 1%
error, from pinn_results.csv (PINN) and baseline_results.csv (methods without a
network). The Cramer-Rao bound comes from their crb_sd_mu column, or from a
crb.csv beside them (sweep.py's own output).
  python summary.py [folder]      (default: results)
"""
import sys
from pathlib import Path

import pandas as pd


def summary(folder):
    folder = Path(folder)
    p = pd.read_csv(folder / "pinn_results.csv").set_index("traj_index")
    b = pd.read_csv(folder / "baseline_results.csv")
    mu, group = p.mu_true, p.group
    if "crb_sd_mu" in p:
        bound = p.crb_sd_mu / mu
    else:
        c = pd.read_csv(folder / "crb.csv").set_index("traj_index")
        bound = (c.crb_sd_mu / c.mu_true).reindex(p.index)
    # estimate and the top of its search range (strong-form phase 2: 0.55; the rest: 0.5)
    est = {"non-spline PINN, strong (main)": (p.mu_pred, 0.55),
           "  strong, inner 80%": (p.mu_strong_trim, 0.55),
           "  weak on net, 0.4": (p["mu_weak_net_0.4"], 0.5),
           "  weak on net, 0.8": (p["mu_weak_net_0.8"], 0.5),
           "  Jacobi on net": (p.mu_jacobi_net, 0.5)}
    for m in b.method.unique():
        est[m] = (b[b.method == m].set_index("traj_index").mu_est.reindex(p.index), 0.5)
    bls = b[b.method == "bls"].set_index("traj_index").reindex(p.index)
    est["  bls, first fit only"] = (bls.mu_first, 0.5)

    capped = p.phase1_hit_cap.astype(str).str.lower().eq("true").sum()
    print(f"{folder}: {len(p)} orbits; failed PINN rows {int(p.error.notna().sum())}; "
          f"phase 1 hit the iteration cap on {int(capped)}")
    print(f"bound > 10% (unidentifiable at this noise): {int((bound > 0.1).sum())}")
    print(f"BLS: multi-start rule applied on {int(bls.cost_first.notna().sum())} orbits "
          f"(restarted on {int((bls.n_fits > 1).sum())}), single start on the rest")
    print(f"{'method':34s} {'valid':>5s} {'<10%':>5s} {'<1%':>5s}  {'at bound':>8s}"
          "   <10% where bound < 10%")
    for name, (s, hi) in est.items():
        s = pd.to_numeric(s, errors="coerce")
        e = (s - mu).abs() / mu
        sat = int(((s >= hi * 0.999) | (s <= 1e-7 * 1.001)).sum())
        print(f"{name:34s} {int(s.notna().sum()):5d} {int((e < 0.1).sum()):5d} "
              f"{int((e < 0.01).sum()):5d}  {sat:8d}"
              f"   {int((e[bound < 0.1] < 0.1).sum())} of {int((bound < 0.1).sum())}")
    e = (p.mu_pred - mu).abs() / mu
    print("\nnon-spline PINN strong form, under 10% by mu band and geometry:")
    bands = pd.cut(mu, [0, 1e-3, 1e-2, 0.1, 1], labels=["<1e-3", "1e-3-1e-2", "1e-2-0.1", ">0.1"])
    print(pd.crosstab(bands, group, values=(e < 0.1).astype(int), aggfunc="sum", margins=True)
          .fillna(0).astype(int).to_string())


if __name__ == "__main__":
    summary(sys.argv[1] if len(sys.argv) > 1 else "results")
