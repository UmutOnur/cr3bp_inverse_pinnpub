run8 - PINN under noise and sparse sampling, after changing the phase-1 stopping rule

I changed the rule that decides when phase-1 training stops, then reran these
cases on trajectories 646 and 821:

  R: positions + velocities, spline target, 1000 points, no noise
  A: positions only,         no target,     1000 points, noise sigma = 1e-5
  B: positions only,         no target,     200 points,  noise sigma = 1e-4
  F: positions only,         no target,     1000 points, noise sigma = 1e-3

Adam capped at 40,000 iterations, then L-BFGS. Noise uses the same random
seeds as run7, so rows match run7's rows for the same trajectory, point count
and noise.

Files: probe.py, results.csv
