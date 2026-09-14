"""
One-off integrity check for cr3bp_dataset_final.npy.

train.py and baselines.py both rely on two assumptions this checks:
trajectory boundaries found via `np.where(np.diff(t_col) < 0)[0] + 1`
(t resets to 0 at each new trajectory), and every trajectory being a
well-formed integration result.

The file is 999,438 rows, not the full 1,000,000 (50 mu x 20 IC x 1000
pts), because rows with NaN/Inf were stripped after generation.

Run: python verify_dataset.py
"""

import numpy as np

DATASET_PATH = "cr3bp_dataset_final.npy"
N_EXPECTED_TRAJECTORIES = 1000  # 50 mu values x 20 initial conditions
T_END_EXPECTED = 2 * np.pi
T_TOL = 1e-6


def expected_mu_values():
    """Recompute the mu grid exactly as data_gen.py does, for cross-checking."""
    mu_log = np.logspace(np.log10(1e-5), np.log10(0.02), 35)
    mu_lin = np.linspace(0.02, 0.5, 15)
    return np.unique(np.concatenate((mu_log, mu_lin)))


def find_boundaries(t_col):
    boundaries = np.where(np.diff(t_col) < 0)[0] + 1
    return np.concatenate([[0], boundaries, [len(t_col)]])


def main():
    data = np.load(DATASET_PATH)
    problems = []

    print(f"loaded {DATASET_PATH}: shape={data.shape}")
    if data.shape[1] != 8:
        problems.append(f"expected 8 columns [x,y,z,vx,vy,vz,t,mu], got {data.shape[1]}")

    # --- finiteness -------------------------------------------------------
    finite_mask = np.isfinite(data)
    n_bad_rows = (~finite_mask.all(axis=1)).sum()
    if n_bad_rows:
        problems.append(f"{n_bad_rows} rows contain NaN/Inf")
    print(f"NaN/Inf rows: {n_bad_rows}")

    t_col = data[:, 6]
    mu_col = data[:, 7]
    boundaries = find_boundaries(t_col)
    n_traj = len(boundaries) - 1
    print(f"trajectories found (via diff(t)<0 boundary trick): {n_traj}")
    if n_traj != N_EXPECTED_TRAJECTORIES:
        problems.append(
            f"expected {N_EXPECTED_TRAJECTORIES} trajectories (50 mu x 20 IC), found {n_traj}"
        )

    # --- per-trajectory checks ---------------------------------------------
    row_counts = []
    truncated = []       # last t noticeably short of 2*pi (integration cut short)
    non_monotonic = []   # t decreases somewhere other than at a trajectory reset
    mu_inconsistent = [] # mu column not constant within the trajectory
    mu_per_traj = []

    for i in range(n_traj):
        start, end = boundaries[i], boundaries[i + 1]
        t_traj = t_col[start:end]
        mu_traj = mu_col[start:end]

        row_counts.append(end - start)

        if not np.all(np.diff(t_traj) > 0):
            non_monotonic.append(i)

        if t_traj[-1] < T_END_EXPECTED - T_TOL:
            truncated.append((i, float(t_traj[-1])))

        if not np.allclose(mu_traj, mu_traj[0]):
            mu_inconsistent.append(i)
        mu_per_traj.append(mu_traj[0])

    row_counts = np.array(row_counts)
    mu_per_traj = np.array(mu_per_traj)

    print(f"row count per trajectory: min={row_counts.min()} max={row_counts.max()} "
          f"(expected 1000 if nothing was ever dropped)")
    short = np.where(row_counts < 1000)[0]
    print(f"trajectories with < 1000 rows: {len(short)}"
          f"{' -> ' + str(list(short[:10])) + ('...' if len(short) > 10 else '') if len(short) else ''}")

    if non_monotonic:
        problems.append(f"{len(non_monotonic)} trajectories have non-monotonic t "
                         f"mid-trajectory (breaks the boundary-detection trick): {non_monotonic[:10]}")
    else:
        print("t is strictly increasing within every trajectory: OK")

    if mu_inconsistent:
        problems.append(f"{len(mu_inconsistent)} trajectories have a non-constant mu column: "
                         f"{mu_inconsistent[:10]}")
    else:
        print("mu is constant within every trajectory: OK")

    if truncated:
        print(f"trajectories ending before t={T_END_EXPECTED:.4f} (2*pi): {len(truncated)}")
        worst = sorted(truncated, key=lambda p: p[1])[:5]
        for idx, t_last in worst:
            print(f"    traj {idx}: last t = {t_last:.4f} "
                  f"({100 * t_last / T_END_EXPECTED:.1f}% of one full rotation)")
    else:
        print(f"every trajectory reaches t={T_END_EXPECTED:.4f} (2*pi): OK")

    # --- mu grid structure ---------------------------------------------------
    expected_mus = expected_mu_values()
    print(f"expected distinct mu values (from data_gen.py's formula): {len(expected_mus)}")
    found_mus = np.unique(mu_per_traj)
    print(f"distinct mu values actually present: {len(found_mus)}")

    matched = [np.any(np.isclose(m, expected_mus, rtol=1e-9)) for m in found_mus]
    unexpected = found_mus[~np.array(matched)]
    if len(unexpected):
        problems.append(f"{len(unexpected)} mu values don't match data_gen.py's grid: {unexpected[:10]}")

    missing_mus = [m for m in expected_mus if not np.any(np.isclose(m, found_mus, rtol=1e-9))]
    if missing_mus:
        problems.append(f"{len(missing_mus)} expected mu values are entirely absent from the "
                         f"dataset (every trajectory for that mu was removed): {missing_mus}")

    # exact equality: found_mus itself came from np.unique(mu_per_traj), and the
    # log-grid/linear-grid boundary can produce two mu's one ULP apart (both
    # legitimately "0.02") that isclose() would wrongly merge.
    counts_per_mu = {m: int(np.sum(mu_per_traj == m)) for m in found_mus}
    off_count = {m: c for m, c in counts_per_mu.items() if c != 20}
    if off_count:
        print(f"mu values without exactly 20 trajectories ({len(off_count)} of {len(found_mus)}):")
        for m, c in list(off_count.items())[:10]:
            print(f"    mu={m:.6g}: {c} trajectories")
    else:
        print("every mu value has exactly 20 trajectories: OK")

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    if problems:
        print(f"FAIL -- {len(problems)} issue(s) found:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("PASS -- no structural problems found.")
        print(f"({data.shape[0]} rows across {n_traj} trajectories, "
              f"{row_counts.sum() - N_EXPECTED_TRAJECTORIES * 1000} rows short of a "
              f"full 1000x1000 grid, consistent with targeted row removal rather than "
              f"corruption.)")


if __name__ == "__main__":
    main()
