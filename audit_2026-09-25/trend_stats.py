"""
Vectorised trend statistics used by the recalculation audit (numpy only, NaN-aware).

mk_raw        : Mann-Kendall S / tau on a series exactly as the historical notebooks
                computed it (all pairs of the raw series, no tie correction, normal
                approximation with continuity correction). Reproduction only.
seasonal_kendall : Hirsch, Slack & Smith (1982) Seasonal Kendall test -- MK computed
                within each calendar month across years and summed; variance with the
                standard tie correction. Removes the seasonal cycle that invalidates MK on
                raw monthly series. tau_SK = S / sum_m C(n_m, 2).
seasonal_sen  : Hirsch et al. (1982) seasonal Sen slope -- median over all within-month
                pairwise slopes (units per year).
bh_fdr        : Benjamini-Hochberg (1995) adjusted q-values.
"""
import numpy as np
from scipy.special import erf


def _norm_p(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(np.abs(z) / np.sqrt(2.0))))


def mk_raw(x):
    """x: (T, N). Returns S, n, tau, p (historical definition, ties ignored)."""
    T = x.shape[0]
    S = np.zeros(x.shape[1], np.float64)
    for lag in range(1, T):
        d = x[lag:] - x[:-lag]
        S += np.nansum(np.sign(d), axis=0)
    n = np.sum(np.isfinite(x), axis=0).astype(np.float64)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = np.where(S > 0, (S - 1) / np.sqrt(np.maximum(var, 1e-9)),
                 np.where(S < 0, (S + 1) / np.sqrt(np.maximum(var, 1e-9)), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(n > 2, S / (n * (n - 1) / 2), np.nan)
    p = np.where(n >= 10, _norm_p(z), np.nan)
    return S, n, tau, p


def _tie_term(xm):
    """Sum over tie groups of t(t-1)(2t+5) per column; xm (Y, N) with NaNs."""
    out = np.zeros(xm.shape[1], np.float64)
    xs = np.sort(xm, axis=0)
    # run-length of equal consecutive finite values
    eq = (xs[1:] == xs[:-1]) & np.isfinite(xs[1:])
    run = np.zeros(xm.shape[1], np.float64)
    for k in range(eq.shape[0]):
        run = np.where(eq[k], run + 1, run)
        end = ~eq[k]
        t = run + 1
        out += np.where(end & (run > 0), t * (t - 1) * (2 * t + 5), 0.0)
        run = np.where(end, 0, run)
    t = run + 1
    out += np.where(run > 0, t * (t - 1) * (2 * t + 5), 0.0)
    return out


def seasonal_kendall(x, months):
    """x: (T, N) monthly series; months: (T,) calendar month 1..12.
    Returns S, tau_SK, z, p, n_pairs."""
    N = x.shape[1]
    S = np.zeros(N); V = np.zeros(N); NP = np.zeros(N)
    for m in range(1, 13):
        xm = x[months == m]
        Y = xm.shape[0]
        if Y < 2:
            continue
        Sm = np.zeros(N)
        for lag in range(1, Y):
            Sm += np.nansum(np.sign(xm[lag:] - xm[:-lag]), axis=0)
        n = np.sum(np.isfinite(xm), axis=0).astype(np.float64)
        Vm = (n * (n - 1) * (2 * n + 5) - _tie_term(xm)) / 18.0
        S += Sm; V += np.where(n >= 2, Vm, 0); NP += np.where(n >= 2, n * (n - 1) / 2, 0)
    z = np.where(S > 0, (S - 1) / np.sqrt(np.maximum(V, 1e-12)),
                 np.where(S < 0, (S + 1) / np.sqrt(np.maximum(V, 1e-12)), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(NP > 0, S / NP, np.nan)
    p = np.where(NP >= 30, _norm_p(z), np.nan)
    return S, tau, z, p, NP


def seasonal_sen(x, months, years, chunk=20000):
    """Median of all within-calendar-month pairwise slopes (per year)."""
    N = x.shape[1]
    out = np.full(N, np.nan)
    pairs = []
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                pairs.append((idx[i], idx[j]))
    pi = np.array([p[0] for p in pairs]); pj = np.array([p[1] for p in pairs])
    dy = (years[pj] - years[pi]).astype(np.float64)
    for s in range(0, N, chunk):
        sl = (x[pj, s:s + chunk] - x[pi, s:s + chunk]) / dy[:, None]
        out[s:s + chunk] = np.nanmedian(sl, axis=0)
    return out


def bh_fdr(p):
    """Benjamini-Hochberg q-values for a 1-D array with NaNs."""
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    o = np.argsort(pv)
    ranked = pv[o] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qq = np.empty(n); qq[o] = np.minimum(ranked, 1.0)
    q[ok] = qq
    return q
