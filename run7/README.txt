run7 - benchmark of eight non-network methods

I ran eight methods on 319 trajectories, positions only:

  gradient matching: spline, smoothing spline, Fourier, Savitzky-Golay,
                     total-variation derivative
  orbit determination: batch least squares (bls), extended Kalman filter (ekf)
  shooting with the initial state fixed to the first sample (1000 points only)

Every trajectory at every combination of
  number of points: 1000, 500, 200, 100, 50, 25
  noise sigma:      0, 1e-6, 1e-5, 1e-4, 1e-3
Noise seed for each case: 1000 * traj_index + noise index (0-4).

ekf_pass.py reruns the EKF on the same cases with its measurement noise set to
the true sigma and several starting guesses for mu.

dop853_check.py reruns bls, ekf and shooting on 40 trajectories (20 with
min_r < 0.05, 20 above) at 1000 points, sigma 0 and 1e-4, once with the RK45
integrator used to generate the data and once with DOP853.

Files: sweep.py -> raw.csv, ekf_pass.py -> ekf_fair.csv,
dop853_check.py -> dop853_check.csv
