run9 - close approaches, and positions only without noise

Two cases, with the phase-1 stopping rule from run8:

  G: trajectories 236, 606, 641, 751 (closest approach 0.054 to 0.092)
     positions + velocities, spline target, 1000 points, no noise
  H: trajectories 646, 821
     positions only, no target, 1000 points, no noise

Adam capped at 40,000 iterations, then L-BFGS.

Files: probe.py, results.csv
