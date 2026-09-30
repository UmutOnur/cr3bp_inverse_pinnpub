"""
Non-PINN baselines for CR3BP mu estimation. Every estimator takes the
same (t, x, y, vx, vy) arrays used in train.py, accepts
use_velocity=False for the position-only setting the TUBITAK proposal
specifies, and returns a dict containing mu_pred.

Two families, from two literatures.

GRADIENT MATCHING (a.k.a. the two-step method, Varah 1982): smooth the
data, then fit mu algebraically to the smoothed derivatives. The
dynamics are never integrated. These differ only in the smoother:

    estimate_mu_spline()       interpolating spline (quintic on v,
                               septic on x/y differentiated twice).
                               The unregularised extreme; Varah's own
                               version used penalised splines.
    estimate_mu_smoothspline() penalised cubic spline (Reinsch/Wahba).
    estimate_mu_savgol()       Savitzky-Golay, the most widely
                               recognised noise-robust derivative filter.
    estimate_mu_tvdiff()       total-variation regularised
                               differentiation (Chartrand 2007), what
                               the sparse system-identification
                               literature uses for noisy derivatives.
    estimate_mu_fourier()      global Fourier least squares. NOT a
                               literature baseline for this problem;
                               kept as an extra, present it as such.
    classical_scan()           the original finite-difference grid scan.
                               UNUSABLE: linear mu grid that cannot
                               represent mu < 0.001, and no
                               position-only mode at all.

ORBIT DETERMINATION: integrate the dynamics and fit, never
differentiating the measurements. Batch least squares and sequential
filtering are the two established methods in this field (Tapley, Born &
Schutz, Statistical Orbit Determination, 2004):

    estimate_mu_bls()      batch least squares over the initial state
                           AND mu together. The standard baseline.
    estimate_mu_ekf()      extended Kalman filter with mu as an
                           augmented constant state.
    estimate_mu_shooting() ABLATION ONLY: as bls, but the initial state
                           is pinned to the first observation, so under
                           noise the whole arc is anchored to one noisy
                           point and the error comes out proportional
                           to sigma. Included to show why orbit
                           determination estimates the state rather
                           than reading it off.

Caveats to disclose in the paper:
  * Inverse crime: the dataset and bls/ekf/shooting all integrate the
    same CR3BP equations through solve_ivp, which flatters those three
    (not the gradient-matching family). Switching them to DOP853 is the
    open fix.
  * bls and ekf take their partial derivatives by finite differences;
    standard OD uses the state transition matrix from the variational
    equations (available in the cr3bp module). Same answers at
    atol/rtol=1e-12, but slower, and it is most of the sweep cost.
  * A cubic smoothing spline differentiated twice has a piecewise-
    linear second derivative. The right tool is a quintic penalised
    spline; SciPy provides none.
  * Four smoothers pick their own smoothing strength by whichever
    setting minimises the CR3BP residual (_select_by_residual), not by
    the conventional GCV / held-out CV, which is known to oversmooth
    for derivative estimation. Truth-free, but non-standard, and it
    makes the baselines stronger rather than weaker.
  * Residuals are unweighted. Identical to weighted here because every
    channel carries the same sigma by construction.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline, make_smoothing_spline
from scipy.signal import savgol_filter
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize_scalar

import cr3bp

DEFAULT_MU_BOUNDS = (1e-7, 0.5)
# method is part of this dict so a study can switch integrators by
# rebinding it; RK45 is what data_gen.py used, so inverting with RK45
# is an inverse crime and DOP853 is the control.
INTEGRATOR_TOL = dict(atol=1e-12, rtol=1e-12, method="RK45")


def initial_velocity_from_positions(t, x, y, window=None):
    """(vx0, vy0) at t[0] from the positions alone, via a local quadratic
    least-squares fit over the first `window` samples.

    Needed because position-only runs must not touch the observed
    velocities: seeding an estimator with vx[0], vy[0] leaks exactly the
    data the setting is meant to withhold. A one-sided difference would
    do the same job but amplifies noise by 1/dt; the local quadratic
    averages the window instead.
    """
    t = np.asarray(t, dtype=float).flatten()
    if window is None:
        # a fixed 7 samples spans 28% of the arc at n=25, where a local
        # quadratic is a poor model; cap it at ~8% of the record
        window = max(3, min(7, int(round(0.08 * len(t)))))
    m = min(window, len(t))
    tt = t[:m] - t[0]
    A = np.vstack([np.ones(m), tt, tt ** 2]).T
    vx0 = np.linalg.lstsq(A, np.asarray(x, dtype=float).flatten()[:m], rcond=None)[0][1]
    vy0 = np.linalg.lstsq(A, np.asarray(y, dtype=float).flatten()[:m], rcond=None)[0][1]
    return float(vx0), float(vy0)


# ---------------------------------------------------------------------------
# Gradient matching (grid scan)
# ---------------------------------------------------------------------------
def classical_scan(t, x, y, vx, vy, n_points=300):
    """Network-free baseline: scan mu against finite-difference accelerations."""
    t_f, x_f, y_f = t.flatten(), x.flatten(), y.flatten()
    vx_f, vy_f = vx.flatten(), vy.flatten()
    ax = np.gradient(vx_f, t_f)
    ay = np.gradient(vy_f, t_f)

    mu_grid = np.linspace(0.001, 0.499, n_points)
    losses = []
    for m in mu_grid:
        r1 = np.sqrt((x_f + m) ** 2 + y_f ** 2)
        r2 = np.sqrt((x_f - (1 - m)) ** 2 + y_f ** 2)
        pred_ax = x_f - (1 - m) * (x_f + m) / r1 ** 3 - m * (x_f - (1 - m)) / r2 ** 3 + 2 * vy_f
        pred_ay = y_f - (1 - m) * y_f / r1 ** 3 - m * y_f / r2 ** 3 - 2 * vx_f
        losses.append(np.mean((pred_ax - ax) ** 2) + np.mean((pred_ay - ay) ** 2))
    losses = np.array(losses)
    return float(mu_grid[losses.argmin()]), float(losses.max() / losses.min())


# ---------------------------------------------------------------------------
# Derivative estimators with a continuous log-mu fit (spline, Fourier)
# ---------------------------------------------------------------------------
def cr3bp_residual(mu, x, y, vx, vy, ax, ay):
    """Mean squared mismatch between given accelerations and the CR3BP
    equations at mu. Same formula as train.physics_residual."""
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    fx = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    fy = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return np.mean((ax - fx) ** 2) + np.mean((ay - fy) ** 2)


def fit_mu_continuous(x, y, vx, vy, ax, ay, n_grid=200, bounds=DEFAULT_MU_BOUNDS):
    """Coarse log-spaced scan, then a bounded 1-D minimisation of the
    residual in log10(mu) around the scan's best point. Returns (mu, contrast)."""
    f = lambda lm: cr3bp_residual(10 ** lm, x, y, vx, vy, ax, ay)
    lo, hi = np.log10(bounds[0]), np.log10(bounds[1])
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([f(g) for g in grid])
    i, step = int(vals.argmin()), grid[1] - grid[0]
    # clip to the declared range: without this the refinement window at an
    # edge cell escapes the bounds, and a flat landscape (whose argmin sits
    # at the first cell) returned mu = 9.25e-8, below the 1e-7 lower bound
    r = minimize_scalar(f, bounds=(max(lo, grid[i] - step), min(hi, grid[i] + step)),
                        method="bounded", options={"xatol": 1e-12})
    return float(10 ** r.x), float(vals.max() / vals.min())


class FourierFit:
    """Least squares on 1, t, t^2 and K harmonics of the 2*pi rotation; K
    chosen by 5-fold cross-validation. Derivatives are analytic. A global
    fit, so noise is averaged over the whole arc rather than per sample."""

    HARMONICS = (2, 4, 8, 16, 32, 64, 128)

    @classmethod
    def allowed(cls, n):
        """Harmonic counts that keep the fit overdetermined at this n."""
        return tuple(K for K in cls.HARMONICS if 2 * K + 3 <= n // 2) or (2,)

    def __init__(self, t, u, K):
        # K is chosen by the estimator, by CR3BP residual, the same way the
        # other four smoothers pick their smoothing strength. Cross-
        # validation on the data was tried and is the wrong criterion here:
        # interleaved folds leave every held-out sample bracketed by its
        # neighbours so K pinned to the maximum on every orbit, while
        # contiguous folds force extrapolation at the ends and pushed K
        # down to 2. Neither measures derivative quality, which is what
        # matters (see the _select_by_residual docstring).
        self.K = K
        self.c = np.linalg.lstsq(self.basis(t, K), u, rcond=None)[0]

    @staticmethod
    def basis(t, K, d=0):
        k = np.arange(1, K + 1)[None, :]
        tt = t[:, None]
        if d == 0:
            poly = np.stack([np.ones_like(t), t, t ** 2], 1)
            trig = [np.sin(k * tt), np.cos(k * tt)]
        elif d == 1:
            poly = np.stack([np.zeros_like(t), np.ones_like(t), 2 * t], 1)
            trig = [k * np.cos(k * tt), -k * np.sin(k * tt)]
        else:
            poly = np.stack([np.zeros_like(t), np.zeros_like(t), 2 * np.ones_like(t)], 1)
            trig = [-k ** 2 * np.sin(k * tt), -k ** 2 * np.cos(k * tt)]
        return np.hstack([poly] + trig)

    def __call__(self, t, d=0):
        return self.basis(t, self.K, d) @ self.c


def estimate_mu_spline(t, x, y, vx, vy, use_velocity=True):
    """Interpolating spline through the data, differentiated analytically
    (quintic on v; septic on x, y twice if positions only), then
    fit_mu_continuous. Exact on clean dense data, fragile under noise."""
    t0 = time.time()
    t_f, x_f, y_f = t.flatten(), x.flatten(), y.flatten()
    if use_velocity:
        vx_f, vy_f = vx.flatten(), vy.flatten()
        ax = make_interp_spline(t_f, vx_f, k=5).derivative()(t_f)
        ay = make_interp_spline(t_f, vy_f, k=5).derivative()(t_f)
    else:
        sx = make_interp_spline(t_f, x_f, k=7)
        sy = make_interp_spline(t_f, y_f, k=7)
        vx_f, vy_f = sx.derivative()(t_f), sy.derivative()(t_f)
        ax, ay = sx.derivative(2)(t_f), sy.derivative(2)(t_f)
    mu, contrast = fit_mu_continuous(x_f, y_f, vx_f, vy_f, ax, ay)
    return dict(mu_pred=mu, contrast=contrast,
                derivs=(x_f, y_f, vx_f, vy_f, ax, ay),
                residual=cr3bp_residual(mu, x_f, y_f, vx_f, vy_f, ax, ay),
                elapsed_s=time.time() - t0)


def estimate_mu_fourier(t, x, y, vx, vy, use_velocity=True, harmonics=None):
    """Gradient matching with a global Fourier least-squares fit,
    differentiated analytically. The harmonic count is chosen by
    residual unless given. Not a literature baseline for this problem;
    kept as the linear cousin of the PINN's Fourier input features."""
    t0 = time.time()
    n = len(np.asarray(t).flatten())
    Ks = (harmonics,) if harmonics is not None else FourierFit.allowed(n)
    cands = {}
    for K in Ks:
        cache = {}

        def fit(tt, u, K=K, cache=cache):
            key = (len(u), float(u[0]), float(u[-1]), float(u.sum()))
            if key not in cache:
                cache[key] = FourierFit(tt, u, K)
            return cache[key]

        cands[K] = (
            (lambda tt, u, fit=fit: fit(tt, u)(tt, 1)),
            (lambda tt, u, fit=fit: fit(tt, u)(tt, 2)),
        )
    mu, contrast, best, derivs, resid = _select_by_residual(
        t, x, y, vx, vy, use_velocity, cands)
    return dict(mu_pred=mu, contrast=contrast, harmonics=best, derivs=derivs,
                residual=resid, elapsed_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Shooting + nonlinear least squares
# ---------------------------------------------------------------------------
def _shoot(mu, t, ic6, **tol):
    """Integrate the CR3BP EOMs for a scalar mu, returning the solve_ivp result."""
    eoms = cr3bp.EOMConstructor(mu)
    return solve_ivp(eoms, (t[0], t[-1]), ic6, t_eval=t, **tol)


def _shooting_residual(log_mu, t, ic6, x, y, vx, vy, use_velocity, tol):
    mu = 10.0 ** log_mu[0]
    sol = _shoot(mu, t, ic6, **tol)
    n = len(t)
    if not sol.success or sol.y.shape[1] != n:
        # Penalize failed/truncated integrations (e.g. a candidate mu that
        # sends the third body into a primary) without crashing the optimizer.
        fill = 1e3
        return np.full(4 * n if use_velocity else 2 * n, fill)
    xp, yp, vxp, vyp = sol.y[0], sol.y[1], sol.y[3], sol.y[4]
    if use_velocity:
        return np.concatenate([xp - x, yp - y, vxp - vx, vyp - vy])
    return np.concatenate([xp - x, yp - y])


def estimate_mu_shooting(t, x, y, vx, vy, use_velocity=True,
                          mu_bounds=DEFAULT_MU_BOUNDS, n_coarse=25,
                          tol=None, verbose=False):
    """
    Recover mu by single shooting from the trajectory's own initial
    condition, refined with scipy.optimize.least_squares.

    Parameters
    ----------
    t, x, y, vx, vy : array-like
        Observed trajectory, same layout as train.load_trajectory.
    use_velocity : bool
        If True (default) the residual includes vx, vy as well as x, y.
        Set False for a position-only fit.
    mu_bounds : (lo, hi)
        Search range for mu, worked in log10(mu) space.
    n_coarse : int
        Log-spaced points used only to seed the optimizer's starting
        point; final precision comes from least_squares, not this grid.
    tol : dict or None
        Overrides for solve_ivp's atol/rtol (default 1e-12/1e-12).

    Returns
    -------
    dict with mu_pred, success, cost, nfev, coarse_log_mu0,
    coarse_costs, elapsed_s.
    """
    tol = tol or INTEGRATOR_TOL
    t = np.asarray(t, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    vx = np.asarray(vx, dtype=float).flatten()
    vy = np.asarray(vy, dtype=float).flatten()
    # position-only must not use the observed velocity, not even to pin the
    # start; derive it from the positions instead
    if use_velocity:
        v0 = (vx[0], vy[0])
    else:
        v0 = initial_velocity_from_positions(t, x, y)
    ic6 = np.array([x[0], y[0], 0.0, v0[0], v0[1], 0.0])

    t_start = time.time()
    log_lo, log_hi = np.log10(mu_bounds[0]), np.log10(mu_bounds[1])
    coarse_grid = np.linspace(log_lo, log_hi, n_coarse)
    coarse_costs = np.array([
        np.sum(_shooting_residual([lm], t, ic6, x, y, vx, vy, use_velocity, tol) ** 2)
        for lm in coarse_grid
    ])
    log_mu0 = float(coarse_grid[np.argmin(coarse_costs)])

    result = least_squares(
        _shooting_residual, x0=[log_mu0],
        args=(t, ic6, x, y, vx, vy, use_velocity, tol),
        bounds=([log_lo], [log_hi]), method="trf",
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    mu_pred = float(10.0 ** result.x[0])
    elapsed = time.time() - t_start

    if verbose:
        print(f"  shooting: coarse mu0={10**log_mu0:.6g} -> refined mu={mu_pred:.6g} "
              f"cost={result.cost:.3e} nfev={result.nfev} success={result.success} "
              f"({elapsed:.2f}s)")

    return {
        "mu_pred": mu_pred,
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "coarse_log_mu0": log_mu0,
        "coarse_costs": (coarse_grid, coarse_costs),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Savitzky-Golay and total-variation regularised differentiation
# ---------------------------------------------------------------------------
def savgol_derivatives(t, u, order, window=None, polyorder=4):
    """`order`-th derivative of u(t) by Savitzky-Golay: a least-squares
    polynomial fitted in a sliding window, differentiated analytically.

    Assumes uniform spacing (the sweep subsamples on a fixed stride, so
    it is). Window defaults to ~11% of the record, clamped odd and to at
    least polyorder+2; SG is known to be weakest near the ends of a
    record, which matters on short arcs, so mode="interp" is used rather
    than zero padding."""
    t = np.asarray(t, dtype=float).flatten()
    n = len(t)
    dt = float(np.median(np.diff(t)))
    if window is None:
        window = max(polyorder + 2, int(round(0.11 * n)))
    window = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
    polyorder = min(polyorder, window - 1)
    return savgol_filter(np.asarray(u, dtype=float).flatten(), window, polyorder,
                         deriv=order, delta=dt, mode="interp")


def _tv_operators(n, dt):
    """Matrix-free trapezoidal integration operator and its adjoint.

    A[i,j] = dt*[j<=i] - dt/2*[i==j] - dt/2*[j==0]. Building that
    explicitly costs O(n^2) nonzeros and makes A'A dense (82 s per orbit
    at n=1000); applying it with cumulative sums is O(n).
    """
    def A(v):
        return np.cumsum(v) * dt - dt * (v + v[0]) / 2.0

    def At(r):
        out = np.cumsum(r[::-1])[::-1] * dt - dt * r / 2.0
        out[0] -= dt * r.sum() / 2.0
        return out

    return A, At


def tv_regularised_derivative(t, u, alpha=1e-5, iters=20, eps=1e-8, cg_tol=1e-8):
    """First derivative of noisy u(t) by total-variation regularisation
    (Chartrand 2007).

    Finite differencing amplifies noise by 1/dt. Instead solve for the
    derivative v directly as the minimiser of

        0.5 * || A v - (u - u[0]) ||^2  +  alpha * || D v ||_1

    with A integrating and D differencing. The L1 penalty permits a
    derivative with kinks instead of forcing global smoothness, which is
    the point of the method for signals that are not smooth. Solved by
    lagged diffusivity: freeze the weights 1/sqrt((Dv)^2 + eps), solve
    the resulting linear system by conjugate gradients, repeat.
    """
    t = np.asarray(t, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()
    n = len(t)
    if n < 4:
        return np.gradient(u, t)
    dt = float(np.median(np.diff(t)))
    A, At = _tv_operators(n, dt)
    D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) / dt
    Dt = D.T.tocsr()
    rhs = At(u - u[0])

    v = np.gradient(u, t)
    for _ in range(iters):
        w = 1.0 / np.sqrt((D @ v) ** 2 + eps)

        def mv(z, w=w):
            return At(A(z)) + alpha * (Dt @ (w * (D @ z)))

        M = spla.LinearOperator((n, n), matvec=mv, dtype=float)
        # keep the last cg iterate even when the tolerance is not reached:
        # bailing out on info != 0 silently returned the finite-difference
        # starting guess, which made alpha have no effect (the same bug
        # was present in the second-derivative solve)
        v_new, info = spla.cg(M, rhs, x0=v, rtol=cg_tol, maxiter=800)
        if not np.all(np.isfinite(v_new)):
            break
        step = np.linalg.norm(v_new - v)
        v = v_new
        if step <= 1e-12 * max(np.linalg.norm(v), 1e-30):
            break
    return v


def tv_regularised_second_derivative(t, u, alpha=1e-5, iters=20, eps=1e-8,
                                     cg_tol=1e-7):
    """Velocity AND acceleration from noisy positions in one solve.

    Chartrand's method estimates a first derivative. Running it twice to
    reach a second derivative regularises twice and erases real signal
    along with the noise, which is not what the method is for. Here the
    acceleration `a` and the unknown initial velocity `v0` are solved
    for together, against the forward model

        u - u[0]  ~  A( v0 + A(a) )

    with A the trapezoidal integration operator: integrate the
    acceleration once for velocity, again for position. The L1 penalty
    acts on `a` only; `v0` is a single free scalar and is not penalised.
    Lagged diffusivity as before. Returns (v, a), consistent with each
    other by construction.
    """
    t = np.asarray(t, dtype=float).flatten()
    u = np.asarray(u, dtype=float).flatten()
    n = len(t)
    if n < 6:
        v = np.gradient(u, t)
        return v, np.gradient(v, t)
    dt = float(np.median(np.diff(t)))
    A, At = _tv_operators(n, dt)
    D = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) / dt
    Dt = D.T.tocsr()
    ones = np.ones(n)
    A_ones = A(ones)                      # the v0 column of the forward model
    target = u - u[0]

    def fwd(z):
        return A(z[n] * ones + A(z[:n]))

    def adj(r):
        s_ = At(r)
        out = np.empty(n + 1)
        out[:n] = At(s_)
        out[n] = float(A_ones @ r)
        return out

    rhs = adj(target)
    v_fd = np.gradient(u, t)
    z = np.append(np.gradient(v_fd, t), v_fd[0])
    for _ in range(iters):
        a = z[:n]
        w = 1.0 / np.sqrt((D @ a) ** 2 + eps)

        def mv(q, w=w):
            out = adj(fwd(q))
            out[:n] += alpha * (Dt @ (w * (D @ q[:n])))
            return out

        M = spla.LinearOperator((n + 1, n + 1), matvec=mv, dtype=float)
        # double integration is badly conditioned, so cg rarely reaches a
        # tight tolerance; its last iterate is still the useful answer.
        # Bailing out on info != 0 silently returned the finite-difference
        # starting guess and made alpha have no effect at all.
        z_new, info = spla.cg(M, rhs, x0=z, rtol=cg_tol, maxiter=2000)
        if not np.all(np.isfinite(z_new)):
            break
        step = np.linalg.norm(z_new - z)
        z = z_new
        if step <= 1e-12 * max(np.linalg.norm(z), 1e-30):
            break
    a = z[:n]
    v = z[n] * ones + A(a)
    return v, a


def _select_by_residual(t, x, y, vx, vy, use_velocity, candidates):
    """Pick a smoother's hyperparameter by the CR3BP residual it achieves.

    The spline and Fourier estimators choose their own flexibility from
    the data (interpolation order, cross-validated harmonic count), so a
    hardcoded smoothing strength for Savitzky-Golay or TV would be an
    unfair comparison: too wide a window smooths over real orbital
    curvature (12.9% error on traj 646 even with clean data).

    Selection is by the same physics residual the mu fit minimises,
    evaluated at each candidate's own best mu. It uses no truth, no
    velocity in position-only mode, and nothing outside the observations
    - the same grid-then-refine logic phase 2 uses for mu itself.

    `candidates` maps a label to (first_derivative_fn, second_derivative_fn).
    Returns (mu, contrast, best_label, derivs, residual), where derivs is
    (x, y, vx, vy, ax, ay) as the winning smoother saw them - kept so a
    caller can score the derivatives against truth without recomputing
    them (run7 uses this for the accel_rmse / vel_rmse columns).
    """
    best = None
    for label, (first, second) in candidates.items():
        try:
            mu, contrast, d = _gradient_match(t, x, y, vx, vy, use_velocity,
                                              first, second, return_derivs=True)
        except Exception:
            continue
        if not np.isfinite(mu):
            continue
        score = cr3bp_residual(mu, *d)
        if best is None or score < best[0]:
            best = (score, mu, contrast, label, d)
    if best is None:
        return float("nan"), float("nan"), None, None, float("nan")
    return best[1], best[2], best[3], best[4], best[0]


def _gradient_match(t, x, y, vx, vy, use_velocity, first, second,
                    return_derivs=False):
    """Shared tail for the gradient-matching estimators. `first` and
    `second` take (t, u) and return the 1st / 2nd derivative of u.
    With return_derivs, also hand back the smoothed states and
    accelerations so a caller scoring several candidates does not have
    to recompute them."""
    t_f, x_f, y_f = t.flatten(), x.flatten(), y.flatten()
    if use_velocity:
        vx_f, vy_f = vx.flatten(), vy.flatten()
        ax, ay = first(t_f, vx_f), first(t_f, vy_f)
    else:
        vx_f, vy_f = first(t_f, x_f), first(t_f, y_f)
        ax, ay = second(t_f, x_f), second(t_f, y_f)
    mu, contrast = fit_mu_continuous(x_f, y_f, vx_f, vy_f, ax, ay)
    if return_derivs:
        return mu, contrast, (x_f, y_f, vx_f, vy_f, ax, ay)
    return mu, contrast


SMOOTHSPLINE_LAMBDAS = (None, 1e-14, 1e-11, 1e-8, 1e-5)


def estimate_mu_smoothspline(t, x, y, vx, vy, use_velocity=True, lam=None):
    """Gradient matching with a cubic smoothing spline. The penalty is
    chosen by residual; `None` in the ladder means SciPy's automatic
    generalised cross-validation, which oversmooths badly here (0.75%
    median on clean dense data against the interpolating spline's
    millionth of a percent), so it is offered as one candidate rather
    than used unconditionally."""
    t0 = time.time()
    lams = (lam,) if lam is not None else SMOOTHSPLINE_LAMBDAS

    cache = {}

    def make(tt, u, l):
        key = (l, len(u), float(u[0]), float(u[-1]))
        if key not in cache:
            cache[key] = make_smoothing_spline(tt, u, lam=l)
        return cache[key]

    cands = {}
    for l in lams:
        cands[l] = (
            (lambda tt, u, l=l: make(tt, u, l).derivative()(tt)),
            (lambda tt, u, l=l: make(tt, u, l).derivative(2)(tt)),
        )
    mu, contrast, best, derivs, resid = _select_by_residual(
        t, x, y, vx, vy, use_velocity, cands)
    return dict(mu_pred=mu, contrast=contrast, lam=best, derivs=derivs,
                residual=resid, elapsed_s=time.time() - t0)


SAVGOL_WINDOW_FRACTIONS = (0.007, 0.015, 0.03, 0.06, 0.11, 0.2)
SAVGOL_POLYORDERS = (4, 6)


def estimate_mu_savgol(t, x, y, vx, vy, use_velocity=True,
                       window=None, polyorder=None):
    """Gradient matching with Savitzky-Golay derivatives. The window and
    polynomial order are chosen by residual (see _select_by_residual)
    unless given explicitly."""
    t0 = time.time()
    n = len(np.asarray(t).flatten())
    if window is not None and polyorder is not None:
        grid = {(window, polyorder): (window, polyorder)}
    else:
        grid = {}
        for frac in SAVGOL_WINDOW_FRACTIONS:
            w = max(7, int(round(frac * n)))
            w = w if w % 2 else w + 1
            if w > n:
                continue
            for po in SAVGOL_POLYORDERS:
                if po + 2 <= w:
                    grid[(w, po)] = (w, po)
    cands = {}
    for key, (w, po) in grid.items():
        cands[key] = (
            (lambda tt, u, w=w, po=po: savgol_derivatives(tt, u, 1, w, po)),
            (lambda tt, u, w=w, po=po: savgol_derivatives(tt, u, 2, w, po)),
        )
    mu, contrast, best, derivs, resid = _select_by_residual(
        t, x, y, vx, vy, use_velocity, cands)
    return dict(mu_pred=mu, contrast=contrast, window=None if best is None else best[0],
                polyorder=None if best is None else best[1], derivs=derivs,
                residual=resid, elapsed_s=time.time() - t0)


# Calibrated against a synthetic signal with known derivatives. The joint
# (position -> acceleration) solve needs a larger penalty than a single
# differentiation, because double integration flattens the data term.
# 1e-6 wins at every noise level tested; 1e-8 and 1e-5 bracket it. Values
# at 1e-4 and above were both slower (1-3 s, ill-conditioned weights) and
# never competitive, so they are excluded rather than left for the
# selector to reject.
TV_ALPHAS = (1e-8, 1e-7, 1e-6, 1e-5)


def estimate_mu_tvdiff(t, x, y, vx, vy, use_velocity=True,
                       alpha=None, iters=8):
    """Gradient matching with total-variation regularised derivatives.

    Full-state differentiates the observed velocity once. Position-only
    uses tv_regularised_second_derivative, which solves for velocity and
    acceleration in a single system rather than applying the method
    twice. alpha is chosen by residual unless given."""
    t0 = time.time()
    alphas = (alpha,) if alpha is not None else TV_ALPHAS
    cands = {}
    for al in alphas:
        if use_velocity:
            # one differentiation of the observed velocity
            cands[al] = (
                (lambda tt, u, al=al: tv_regularised_derivative(tt, u, al, iters)),
                None,
            )
        else:
            # positions only: one joint solve gives velocity and acceleration
            cache = {}

            def both(tt, u, al=al, cache=cache):
                key = (len(u), float(u[0]), float(u[-1]), float(u.sum()))
                if key not in cache:
                    cache[key] = tv_regularised_second_derivative(tt, u, al, iters)
                return cache[key]

            cands[al] = (
                (lambda tt, u, both=both: both(tt, u)[0]),
                (lambda tt, u, both=both: both(tt, u)[1]),
            )
    mu, contrast, best, derivs, resid = _select_by_residual(
        t, x, y, vx, vy, use_velocity, cands)
    return dict(mu_pred=mu, contrast=contrast, alpha=best, derivs=derivs,
                residual=resid, elapsed_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Batch least squares (initial state + mu estimated jointly)
# ---------------------------------------------------------------------------
def _bls_residual(params, t, x, y, vx, vy, use_velocity, tol):
    """Residual for [x0, y0, vx0, vy0, log10(mu)]. Unlike
    _shooting_residual, the initial state is a free parameter rather than
    the first observation, so observation noise averages over the whole
    arc instead of anchoring the trajectory to one noisy sample."""
    ic6 = np.array([params[0], params[1], 0.0, params[2], params[3], 0.0])
    mu = 10.0 ** params[4]
    sol = _shoot(mu, t, ic6, **tol)
    n = len(t)
    if not sol.success or sol.y.shape[1] != n:
        return np.full(4 * n if use_velocity else 2 * n, 1e3)
    xp, yp, vxp, vyp = sol.y[0], sol.y[1], sol.y[3], sol.y[4]
    if use_velocity:
        return np.concatenate([xp - x, yp - y, vxp - vx, vyp - vy])
    return np.concatenate([xp - x, yp - y])


def estimate_mu_bls(t, x, y, vx, vy, use_velocity=True,
                    mu_bounds=DEFAULT_MU_BOUNDS, n_coarse=25,
                    max_nfev=400, tol=None, verbose=False):
    """
    Recover mu by batch least squares: the classical orbit-determination
    baseline. Five parameters (x0, y0, vx0, vy0, log10 mu) are fitted to
    all observations at once with scipy.optimize.least_squares.

    The difference from estimate_mu_shooting() is that the initial state
    floats. With noisy data that matters a great deal: pinning the state
    to the first sample forces the optimizer to absorb that sample's
    error into mu. In position-only mode the initial velocity is
    unobserved and is recovered from the position history through the
    dynamics, which is the point of the method.

    max_nfev bounds the worst case (a few orbits otherwise spend minutes
    probing candidate mu that dive into a primary); `success` and `nfev`
    record whether that bound was reached.

    Returns
    -------
    dict with mu_pred, state0, success, cost, nfev, coarse_log_mu0,
    elapsed_s.
    """
    tol = tol or INTEGRATOR_TOL
    t = np.asarray(t, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    vx = np.asarray(vx, dtype=float).flatten()
    vy = np.asarray(vy, dtype=float).flatten()

    t_start = time.time()
    log_lo, log_hi = np.log10(mu_bounds[0]), np.log10(mu_bounds[1])

    # seed mu with the state held at the first observation; cheap, and only
    # decides where the joint fit starts
    if use_velocity:
        v0 = (vx[0], vy[0])
    else:
        v0 = initial_velocity_from_positions(t, x, y)
    state0 = np.array([x[0], y[0], v0[0], v0[1]])
    coarse_grid = np.linspace(log_lo, log_hi, n_coarse)
    coarse_costs = np.array([
        np.sum(_bls_residual(np.append(state0, lm), t, x, y, vx, vy,
                             use_velocity, tol) ** 2)
        for lm in coarse_grid
    ])
    log_mu0 = float(coarse_grid[int(coarse_costs.argmin())])

    p0 = np.append(state0, log_mu0)
    lo = np.array([-np.inf, -np.inf, -np.inf, -np.inf, log_lo])
    hi = np.array([np.inf, np.inf, np.inf, np.inf, log_hi])
    result = least_squares(_bls_residual, p0, bounds=(lo, hi),
                           x_scale="jac", max_nfev=max_nfev,
                           args=(t, x, y, vx, vy, use_velocity, tol),
                           verbose=2 if verbose else 0)

    mu_pred = float(10.0 ** result.x[4])
    return dict(mu_pred=mu_pred, state0=result.x[:4].copy(),
                success=bool(result.success), cost=float(result.cost),
                nfev=int(result.nfev), coarse_log_mu0=log_mu0,
                elapsed_s=time.time() - t_start)


# ---------------------------------------------------------------------------
# Extended Kalman filter, mu as an augmented constant parameter
# ---------------------------------------------------------------------------
def _augmented_dynamics(t, s):
    """CR3BP EOMs on state [x, y, vx, vy, mu], with d(mu)/dt = 0."""
    x, y, vx, vy, mu = s
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - (1 - mu)) ** 2 + y ** 2)
    ax = x - (1 - mu) * (x + mu) / r1 ** 3 - mu * (x - (1 - mu)) / r2 ** 3 + 2 * vy
    ay = y - (1 - mu) * y / r1 ** 3 - mu * y / r2 ** 3 - 2 * vx
    return np.array([vx, vy, ax, ay, 0.0])


def _augmented_jacobian(s, eps=1e-6):
    """Finite-difference Jacobian of _augmented_dynamics at state s (5x5)."""
    n = len(s)
    f0 = _augmented_dynamics(0.0, s)
    J = np.zeros((n, n))
    for i in range(n):
        step = eps * max(1.0, abs(s[i]))
        s_pert = s.copy()
        s_pert[i] += step
        J[:, i] = (_augmented_dynamics(0.0, s_pert) - f0) / step
    return J


def _ekf_single_pass(t, x, y, vx, vy, mu0=0.05, use_velocity=True,
                     state_var0=1e-10, vel_var0=None, mu_var0=0.25 ** 2,
                     measurement_std=1e-6, process_var_state=1e-14,
                     process_var_mu=0.0, mu_bounds=DEFAULT_MU_BOUNDS,
                     verbose=False):
    """
    One EKF pass from one starting guess. Public entry point is
    estimate_mu_ekf(), which sets the covariances from the noise level
    and runs this from several mu0.

    Treats mu as a constant, unobserved state variable alongside
    (x, y, vx, vy).

    Processes the trajectory once, sequentially: predicts the next state
    with the nonlinear CR3BP dynamics, propagates covariance by
    linearizing around the current estimate, then corrects against the
    next observed sample. No grid, no global search.

    Parameters mirror estimate_mu_shooting() where they overlap. Additional:
        mu0             -- initial mu guess
        state_var0      -- initial variance on x, y, vx, vy
        vel_var0        -- initial variance on vx, vy alone; defaults to
                           state_var0 with observed velocity, and to a
                           loose 1e-4 in position-only mode where the
                           initial velocity is only a positional estimate
        mu_var0         -- initial variance on mu
        measurement_std -- assumed observation noise std
        process_var_state / process_var_mu -- process noise per step

    Returns
    -------
    dict with mu_pred, mu_history, P_final, elapsed_s.
    """
    t = np.asarray(t, dtype=float).flatten()
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    vx = np.asarray(vx, dtype=float).flatten()
    vy = np.asarray(vy, dtype=float).flatten()
    n = len(t)

    t_start = time.time()
    if use_velocity:
        v0 = (vx[0], vy[0])
        vvar = state_var0 if vel_var0 is None else vel_var0
    else:
        # positions only: the initial velocity is a local fit to the
        # positions, so it must not be trusted like a measurement
        v0 = initial_velocity_from_positions(t, x, y)
        vvar = 1e-4 if vel_var0 is None else vel_var0
    s = np.array([x[0], y[0], v0[0], v0[1], mu0])
    P = np.diag([state_var0, state_var0, vvar, vvar, mu_var0])
    Q = np.diag([process_var_state] * 4 + [process_var_mu])

    dim_z = 4 if use_velocity else 2
    H = np.eye(5)[:dim_z]
    R = np.eye(dim_z) * measurement_std ** 2
    I5 = np.eye(5)

    mu_history = np.empty(n)
    mu_history[0] = s[4]
    sse_innov = 0.0          # sum of squared measurement residuals

    diverged = False
    steps_done = 0
    for k in range(1, n):
        dt = t[k] - t[k - 1]

        # a diverged filter feeds a non-finite state to the integrator,
        # which raises from inside scipy; stop cleanly instead
        if not np.all(np.isfinite(s)):
            diverged = True
            break

        sol = solve_ivp(_augmented_dynamics, (t[k - 1], t[k]), s,
                         t_eval=[t[k]], **INTEGRATOR_TOL)
        # on failure scipy leaves sol.y an empty list, so sol.y[:, -1]
        # raises TypeError rather than reporting the failed integration
        if not sol.success or np.size(sol.y) == 0:
            diverged = True
            break
        s_pred = sol.y[:, -1]

        F = _augmented_jacobian(s)
        Phi = expm(F * dt)
        P_pred = Phi @ P @ Phi.T + Q

        z = np.array([x[k], y[k], vx[k], vy[k]]) if use_velocity else np.array([x[k], y[k]])
        innovation = z - H @ s_pred
        sse_innov += float(innovation @ innovation)
        S = H @ P_pred @ H.T + R
        # solve rather than invert, and use the Joseph form below: the
        # short form (I - KH) P is not symmetry-preserving and can lose
        # positive-definiteness over 1000 steps. Joseph form is the
        # standard recommendation in the OD literature.
        try:
            K = np.linalg.solve(S.T, (P_pred @ H.T).T).T
        except np.linalg.LinAlgError:
            # singular innovation covariance: the filter has lost its
            # covariance, so report divergence rather than raising
            diverged = True
            break

        s = s_pred + K @ innovation
        s[4] = np.clip(s[4], mu_bounds[0], mu_bounds[1])
        IKH = I5 - K @ H
        P = IKH @ P_pred @ IKH.T + K @ R @ K.T
        P = 0.5 * (P + P.T)

        mu_history[k] = s[4]
        steps_done = k

    mu_history[steps_done + 1:] = mu_history[steps_done]
    mu_final = float(mu_history[steps_done])

    elapsed = time.time() - t_start
    if verbose:
        tail = f" DIVERGED at step {steps_done}/{n - 1}" if diverged else ""
        print(f"  ekf: mu0={mu0:.6g} -> mu={mu_final:.6g} ({elapsed:.2f}s){tail}")

    return {
        "mu_pred": mu_final,
        "mu_history": mu_history,
        "P_final": P,
        "diverged": diverged,
        "steps_used": steps_done,
        "n_steps": n - 1,
        # mean squared innovation: how well this run predicted the next
        # observation, averaged over the steps it completed. Needs no data
        # derivatives, so it ranks mu0 candidates without the weakness of
        # a finite-difference score.
        "mean_sq_innov": sse_innov / max(steps_done, 1),
        "elapsed_s": elapsed,
    }


# starting guesses for the global initialisation, spread over the five
# decades of mu the dataset covers
MU0_LADDER = (0.002, 0.02, 0.1, 0.25, 0.45)


def estimate_mu_ekf(t, x, y, vx, vy, mu0=None, use_velocity=True, sigma=None,
                    state_var0=None, measurement_std=None, verbose=False,
                    **kwargs):
    """
    Recover mu with an extended Kalman filter: mu as a constant,
    unobserved state variable alongside (x, y, vx, vy), the trajectory
    processed once, sequentially, with no grid and no data derivatives.

    Additional parameters beyond _ekf_single_pass():
        sigma -- standard deviation of the noise on the observations.
                 Given, it sets the measurement covariance R = sigma^2 I
                 and the initial state variance to match; left None, the
                 filter keeps the old hard-coded 1e-6 / 1e-10, i.e. it
                 assumes near-exact data. Standard orbit determination
                 sets R from the known measurement noise, so pass sigma
                 whenever the data is noisy - a filter told the data is
                 exact trusts every noisy sample and can diverge.
        mu0   -- a single starting guess, or None (the default) to run
                 MU0_LADDER and keep whichever answer has the lowest
                 CR3BP residual. A single start at 0.05 falls in the
                 wrong basin for mu >= 0.15 (the large-mu basin trap).
                 The ranking never consults the truth, so this is a
                 legitimate global initialisation - the same
                 grid-then-refine idea phase 2 and the shooting
                 baseline already use.

    Returns
    -------
    dict with mu_pred, mu_history, P_final, elapsed_s, and (when the
    ladder ran) mu0_best and mu0_score.
    """
    if sigma is not None:
        if measurement_std is None:
            measurement_std = max(float(sigma), 1e-8)
        if state_var0 is None:
            state_var0 = max(float(sigma) ** 2, 1e-12)
    if measurement_std is None:
        measurement_std = 1e-6
    if state_var0 is None:
        state_var0 = 1e-10

    common = dict(use_velocity=use_velocity, state_var0=state_var0,
                  measurement_std=measurement_std, **kwargs)
    if mu0 is not None:
        return _ekf_single_pass(t, x, y, vx, vy, mu0=mu0, verbose=verbose,
                                **common)

    t_start = time.time()
    # Rank candidates by mean squared innovation - the filter's own
    # prediction error against the observations. An earlier version scored
    # them with cr3bp_residual on np.gradient-twice derivatives, which is
    # the weakest estimator in the study: on clean position-only data it
    # picked the best of five candidates only 25% of the time and the pick
    # was ~1300x worse than the best available.
    best = None
    for cand in MU0_LADDER:
        try:
            res = _ekf_single_pass(t, x, y, vx, vy, mu0=cand, **common)
        except Exception:
            continue
        mu = res["mu_pred"]
        if not np.isfinite(mu) or mu <= 0:
            continue
        score = res["mean_sq_innov"]
        if not np.isfinite(score):
            continue
        # a run that stopped early saw fewer observations, so its mean is
        # not comparable; penalise by the fraction of the arc it covered
        score /= max(res["steps_used"] / max(res["n_steps"], 1), 1e-6)
        if best is None or score < best[0]:
            best = (score, cand, res)

    if best is None:
        return dict(mu_pred=float("nan"), mu0_best=float("nan"),
                    mu0_score=float("nan"), mu_history=None, P_final=None,
                    elapsed_s=time.time() - t_start)
    score, cand, res = best
    out = dict(res, mu0_best=cand, mu0_score=score,
               elapsed_s=time.time() - t_start)
    if verbose:
        print(f"  ekf: best of {len(MU0_LADDER)} starts, mu0={cand:.6g} "
              f"-> mu={out['mu_pred']:.6g} ({out['elapsed_s']:.2f}s)")
    return out


# ---------------------------------------------------------------------------
# Self-test on four representative trajectories
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from train import load_trajectory, min_primary_distance

    # easy, clean-but-far, close approach x2
    test_indices = [646, 821, 626, 740]
    METHODS = [
        ("spline", estimate_mu_spline),
        ("smoothspl", estimate_mu_smoothspline),
        ("fourier", estimate_mu_fourier),
        ("savgol", estimate_mu_savgol),
        ("tvdiff", estimate_mu_tvdiff),
        ("bls", estimate_mu_bls),
        ("ekf", estimate_mu_ekf),
        ("shoot*", estimate_mu_shooting),
    ]

    for use_velocity in (True, False):
        print(f"\n=== use_velocity={use_velocity} "
              f"({'full state' if use_velocity else 'positions only'})"
              "   error %, * = ablation")
        print(f"{'traj':>6} {'mu_true':>10} {'min_r':>7} " +
              " ".join(f"{n:>10}" for n, _ in METHODS))
        for idx in test_indices:
            t, x, y, vx, vy, mu_true = load_trajectory("cr3bp_dataset_final.npy", idx)
            min_r = min_primary_distance(x, y, mu_true)
            cells = []
            for _, fn in METHODS:
                try:
                    mu = fn(t, x, y, vx, vy, use_velocity=use_velocity)["mu_pred"]
                    cells.append(f"{abs(mu - mu_true) / mu_true * 100:10.4g}")
                except Exception as exc:
                    cells.append(f"{type(exc).__name__[:10]:>10}")
            print(f"{idx:>6} {mu_true:>10.6f} {min_r:>7.4f} " + " ".join(cells))

    print("\nclassical_scan is velocity-only and linear-grid limited; "
          "excluded from the comparison on purpose.")
