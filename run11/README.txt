run11 - fixed vs adaptive loss weights

I didn't know whether to keep the fixed phase-1 loss weights (10/10/5/5) or let
them adapt during training, so I trained each trajectory both ways.

Adaptive: every 500 iterations, weights move 10% toward equalising each loss
term's gradient size.
Trajectories: 81, 131, 166, 381, 391, 396, 581, 611, 616, 646, 651, 706, 841,
861, 896 (one per mu / closest-approach band).
Setting: positions + velocities, spline target, no noise, 1000 points.
Adam capped at 40,000 iterations, then L-BFGS.

Files: adaptive.py, results.csv (arm column: fixed / adaptive)
