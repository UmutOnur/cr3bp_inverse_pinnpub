run4 - PINN sweep, 200 trajectories

Two-phase pipeline on 200 trajectories: indices 1, 6, 11, ..., 996.

Setting: positions + velocities, 1000 points, no noise.
Phase 1: acceleration targets from finite differences (np.gradient) of the
observed velocities; Adam until every loss is below 1e-6, capped at 100,000
iterations; no L-BFGS.
Phase 2: mu starts at 0.05 and is optimised with Adam, then L-BFGS.

The classical_ columns are a gradient-matching scan on a linear mu grid
(0.001 to 0.499).

Files: results.csv, mu_history_traj*.csv (mu during the phase-2 search)
