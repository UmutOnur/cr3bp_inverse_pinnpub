run5 - PINN sweep, 58 trajectories

I ran the updated pipeline on 58 trajectories (listed in indices.txt): controls
that worked in run4, plus trajectories run4 failed on (large mu, small mu,
close approaches).

Setting: positions + velocities, 1000 points, no noise.
Phase 1: acceleration targets from a quintic spline through the observed
velocities; Adam with scale-aware stopping gates, capped at 100,000 iterations;
then 2500 iterations of L-BFGS.
Phase 2: global scan of mu on a log grid, then Adam + L-BFGS from the scan's
best point, with mu searched in log space.

The run was interrupted after 40 trajectories and the remaining 18 were run
separately.

Files: results.csv (first 40), results_rest.csv (remaining 18), join on
traj_index. run_sweep.py, run_rest.py, indices.txt
