run2 - weight grid search for joint training

Before the two-phase design, the network and mu were trained together in one
optimisation. I didn't know which settings to use, so I grid searched them:

  physics loss weight:     15, 20, 25, 30, 35, 40
  freeze iteration:        2000, 4000, 6000, 8000  (network frozen after this)
  physics ramp length:     2000, 4000, 6000, 8000  (physics term phased in)

Trajectories 626, 646 and 740, 60 combinations each, 180 runs in total.

Files: grid.csv
