"""
Recalculation R4 -- NDVI features F1-F9, from the RAW MOD13A3.061 GeoTIFFs (266 months,
2000-11 .. 2022-12) and their pixel-reliability layers. Run in wildfire_env (esda/libpysal).

Reproduces every historical definition (Step 2) and adds the audit variants:
  QA sensitivity   : {Good, Marginal} (historical) vs {Good} only
  Climatology      : 2001-2020 (historical) vs 2001-2022 (sensitivity)
  F3/F5 identities : anomaly-mean and residual-mean degeneracy, shown numerically
  F6               : MK on the 2x12-MA TREND series (historical) + lag-1 autocorrelation of
                     that series; v2 = Seasonal Kendall tau / p / BH-q + seasonal Sen slope on
                     the QA-filtered monthly NDVI
  F7               : CVSI mutual-information sweep k=1..12, 5 background draws, with
                     (a) historical half-pixel-shifted labels, (b) corrected labels,
                     (c) corrected labels restricted to the v2 training partition
  F8               : global Moran's I / LISA as historically computed (stride-8 subsample,
                     non-India cells row-mean filled) vs India-only cells, block-averaged
  F9               : two-regime logistic threshold theta*, historical labels vs corrected
                     labels vs training-only labels
"""
import glob
import os
import re
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import P, RES, provenance, dump, ndvi_grid, india_mask, read_tif, compare, save_tif  # noqa: E402
from trend_stats import mk_raw, seasonal_kendall, seasonal_sen, bh_fdr  # noqa: E402

OUT = os.path.join(RES, "recalculated", "R4_ndvi")
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)


def ym_of(fp):
    m = re.search(r"doy(\d{4})(\d{3})", os.path.basename(fp))
    d = datetime(int(m.group(1)), 1, 1) + timedelta(days=int(m.group(2)) - 1)
    return d.year, d.month


# ------------------------------------------------------------------ worker (per pixel chunk)
def _work(args):
    s, e, T, N = args
    x = np.load(os.path.join(TMP, "ndvi_qa01_v2.npy"), mmap_mode="r")[:, s:e].astype(np.float64)
    tr = np.load(os.path.join(TMP, "trend.npy"), mmap_mode="r")[:, s:e].astype(np.float64)
    months = np.load(os.path.join(TMP, "months.npy")); years = np.load(os.path.join(TMP, "years.npy"))
    _, n_tr, tau_tr, p_tr = mk_raw(tr)
    # lag-1 autocorrelation of the trend series and of the monthly anomaly series
    def lag1(a):
        a = a - np.nanmean(a, axis=0)
        num = np.nansum(a[1:] * a[:-1], axis=0); den = np.nansum(a * a, axis=0)
        return np.where(den > 0, num / den, np.nan)
    clim = np.load(os.path.join(TMP, "clim_v2.npy"), mmap_mode="r")[:, s:e]
    anom = x - clim[months - 1]
    S, tau_sk, z, p_sk, npairs = seasonal_kendall(x, months)
    sen = seasonal_sen(x, months, years)
    return s, e, tau_tr, p_tr, n_tr, lag1(tr), lag1(anom), tau_sk, p_sk, sen


def main():
    t0 = time.time()
    rep = dict(provenance=provenance(__file__))
    import pandas as pd
    import rasterio
    g = ndvi_grid(); H, W = g["shape"]
    imask = india_mask(g)
    nd = {ym_of(f): f for f in glob.glob(os.path.join(P["ndvi_raw"], "*_monthly_NDVI_doy*.tif"))}
    qa = {ym_of(f): f for f in glob.glob(os.path.join(P["ndvi_raw"], "*_monthly_pixel_reliability_doy*.tif"))}
    viq = {ym_of(f): f for f in glob.glob(os.path.join(P["ndvi_raw"], "*_monthly_VI_Quality_doy*.tif"))}
    keys = [(p.year, p.month) for p in pd.period_range("2000-11", "2022-12", freq="M")]
    rep["n_months"] = len(keys); rep["missing_ndvi"] = [k for k in keys if k not in nd]; rep["missing_qa"] = [k for k in keys if k not in qa]
    rep["reliability_files_outside_study_period"] = sorted(k for k in qa if k not in keys)
    rep["files_outside_study_period"] = len([k for k in nd if k not in keys])
    T = len(keys)
    idx = np.flatnonzero(imask.ravel())             # in-India pixels (state boundary)
    N = idx.size
    rep["india_mask_pixels"] = int(N)
    RESUME = os.environ.get("NDVI_RESUME") == "1" and os.path.exists(os.path.join(TMP, "trend.npy"))
    rep["resumed_from_cached_arrays"] = RESUME
    if RESUME:
        X01h = np.load(os.path.join(TMP, "ndvi_qa01_hist.npy"), mmap_mode="r")
        X01 = np.load(os.path.join(TMP, "ndvi_qa01_v2.npy"), mmap_mode="r")
        X0 = np.load(os.path.join(TMP, "ndvi_qa0.npy"), mmap_mode="r")
        keys_loop = []
    else:
        keys_loop = keys
    if not RESUME:
      X01h = np.lib.format.open_memmap(os.path.join(TMP, "ndvi_qa01_hist.npy"), "w+", np.float32, (T, N))
      X01 = np.lib.format.open_memmap(os.path.join(TMP, "ndvi_qa01_v2.npy"), "w+", np.float32, (T, N))
      X0 = np.lib.format.open_memmap(os.path.join(TMP, "ndvi_qa0.npy"), "w+", np.float32, (T, N))
    qa_counts = np.zeros(5, np.int64)
    for i, k in enumerate(keys_loop):
        with rasterio.open(nd[k]) as s:
            raw = s.read(1).ravel()[idx].astype(np.float32); nodv = s.nodata
        v = raw * 1e-4
        v[(raw == nodv) | (v < -0.2) | (v > 1.0)] = np.nan
        if k in qa:
            with rasterio.open(qa[k]) as s:
                q = s.read(1).ravel()[idx]
            for c in range(-1, 4):
                qa_counts[c + 1] += int((q == c).sum())
            X01h[i] = np.where(np.isin(q, [0, 1]), v, np.nan)
            X01[i] = X01h[i]
            X0[i] = np.where(q == 0, v, np.nan)
        else:
            # historical: qa_files_dict.get() -> None -> NO QA filter applied (reproduced as-is)
            X01h[i] = v
            with rasterio.open(viq[k]) as s:
                mq = s.read(1).ravel()[idx].astype(np.int64) & 0b11   # MODLAND QA bits 0-1
            X01[i] = np.where(np.isin(mq, [0, 1]), v, np.nan)
            X0[i] = np.where(mq == 0, v, np.nan)
            rep.setdefault("months_without_reliability_layer", []).append(dict(
                month=k, valid_unfiltered=int(np.isfinite(v).sum()), valid_after_viquality=int(np.isfinite(X01[i]).sum())))
    if not RESUME:
        X01h.flush(); X01.flush(); X0.flush()
    rep["qa_value_counts_minus1_to_3"] = qa_counts.tolist()
    years = np.array([k[0] for k in keys]); months = np.array([k[1] for k in keys])
    np.save(os.path.join(TMP, "years.npy"), years); np.save(os.path.join(TMP, "months.npy"), months)
    print("read", round(time.time() - t0), "s", flush=True)

    X = np.asarray(X01h)
    Xv2 = np.asarray(X01)
    valid_any = np.isfinite(X).any(axis=0)
    rep["pixels_with_any_valid_ndvi"] = int(valid_any.sum())
    rep["frac_missing_pixel_months_qa01"] = float(np.isnan(X[:, valid_any]).mean())
    rep["frac_missing_pixel_months_qa0"] = float(np.isnan(np.asarray(X0)[:, valid_any]).mean())
    base = (years >= 2001) & (years <= 2020)
    feats = {}
    feats["F1_mean"] = np.nanmean(X, axis=0)
    feats["v2_F1_mean"] = np.nanmean(Xv2, axis=0)
    rep["F1_hist_vs_v2_corr"] = float(np.corrcoef(np.nan_to_num(feats["F1_mean"]), np.nan_to_num(feats["v2_F1_mean"]))[0, 1])
    rep["F1_hist_vs_v2_max_abs_diff"] = float(np.nanmax(np.abs(feats["F1_mean"] - feats["v2_F1_mean"])))
    feats["F1_mean_qa0"] = np.nanmean(np.asarray(X0), axis=0)
    for tag, bmask in (("0120", base), ("0122", years >= 2001)):
        clim = np.full((12, N), np.nan, np.float32)
        for m in range(1, 13):
            clim[m - 1] = np.nanmean(X[bmask & (months == m)], axis=0)
        np.save(os.path.join(TMP, f"clim_{tag}.npy"), clim)
        feats[f"F2_clim_june_{tag}"] = clim[5]
        anom = X - clim[months - 1]
        feats[f"F3_anom_mean_{tag}"] = np.nanmean(anom, axis=0)
        if tag == "0120":
            nval = np.isfinite(anom).sum(0)
            outside = np.nansum(anom[~base], 0) / np.maximum(nval, 1)
            ok = np.isfinite(feats["F3_anom_mean_0120"])
            rep["F3_identity_max_abs_err"] = float(np.nanmax(np.abs(feats["F3_anom_mean_0120"][ok] - outside[ok])))
            np.save(os.path.join(TMP, "anom_0120.npy"), anom.astype(np.float32))
        del anom
    climv = np.full((12, N), np.nan, np.float32)
    for m in range(1, 13):
        climv[m - 1] = np.nanmean(Xv2[base & (months == m)], axis=0)
    np.save(os.path.join(TMP, "clim_v2.npy"), climv)
    feats["v2_F2_clim_june"] = climv[5]
    np.save(os.path.join(TMP, "anom_v2.npy"), (Xv2 - climv[months - 1]).astype(np.float32))
    del climv
    # 2x12 MA trend (historical: NaN-aware renormalised weights), seasonal, residual
    w = np.ones(13, np.float32); w[0] = w[-1] = 0.5
    if RESUME:
        TR = np.load(os.path.join(TMP, "trend.npy"), mmap_mode="r")
    else:
        TR = np.lib.format.open_memmap(os.path.join(TMP, "trend.npy"), "w+", np.float32, (T, N))
        TR[:] = np.nan
    for t in (range(6, T - 6) if not RESUME else []):
        win = X[t - 6:t + 7]
        ww = np.where(np.isnan(win), 0.0, w[:, None])
        TR[t] = np.where(ww.sum(0) > 0, np.nansum(win * w[:, None], 0) / np.maximum(ww.sum(0), 1e-12), np.nan)
    if not RESUME:
        TR.flush()
    trend = np.asarray(TR)
    det = X - trend
    seas = np.full((12, N), np.nan, np.float32)
    for m in range(1, 13):
        seas[m - 1] = np.nanmean(det[months == m], axis=0)
    resid = det - seas[months - 1]
    feats["F4_trend_mean"] = np.nanmean(trend, axis=0)
    feats["F5_resid_mean"] = np.nanmean(resid, axis=0)
    rep["F5_resid_mean_abs_max"] = float(np.nanmax(np.abs(feats["F5_resid_mean"])))
    rep["F5_resid_mean_std"] = float(np.nanstd(feats["F5_resid_mean"]))
    rep["F1_qa01_vs_qa0_corr"] = float(np.corrcoef(np.nan_to_num(feats["F1_mean"]), np.nan_to_num(feats["F1_mean_qa0"]))[0, 1])
    rep["F4_vs_F1_corr"] = float(np.corrcoef(np.nan_to_num(feats["F4_trend_mean"]), np.nan_to_num(feats["F1_mean"]))[0, 1])
    del det, resid, trend
    print("F1-F5", round(time.time() - t0), "s", flush=True)

    # --------------------------------------------------------------- trends in parallel
    chunks = [(s, min(s + 60000, N), T, N) for s in range(0, N, 60000)]
    arrs = {k: np.full(N, np.nan) for k in ("tau_tr", "p_tr", "n_tr", "ac1_trend", "ac1_anom", "sk_tau", "sk_p", "sen")}
    with Pool(22) as pool:
        for s, e, *vals in pool.imap_unordered(_work, chunks):
            for k, v in zip(arrs, vals):
                arrs[k][s:e] = v
    feats["F6_mk_tau_trendseries_hist"] = arrs["tau_tr"]
    feats["v2_sk_tau"] = arrs["sk_tau"]; feats["v2_sen_per_yr"] = arrs["sen"]
    sk_q = bh_fdr(arrs["sk_p"]); tr_q = bh_fdr(arrs["p_tr"])
    feats["v2_sk_q"] = sk_q
    rep["F6"] = dict(
        hist_sig_p05_browning=int(np.nansum((arrs["p_tr"] < 0.05) & (arrs["tau_tr"] < 0))),
        hist_sig_p05_greening=int(np.nansum((arrs["p_tr"] < 0.05) & (arrs["tau_tr"] > 0))),
        hist_sig_fdr_total=int(np.nansum(tr_q < 0.05)),
        median_lag1_autocorr_trend_series=float(np.nanmedian(arrs["ac1_trend"])),
        median_lag1_autocorr_monthly_anomaly=float(np.nanmedian(arrs["ac1_anom"])),
        sk_sig_fdr_greening=int(np.nansum((sk_q < 0.05) & (arrs["sk_tau"] > 0))),
        sk_sig_fdr_browning=int(np.nansum((sk_q < 0.05) & (arrs["sk_tau"] < 0))),
        sk_sig_p05_total=int(np.nansum(arrs["sk_p"] < 0.05)),
        corr_hist_tau_vs_sk_tau=float(pd.Series(arrs["tau_tr"]).corr(pd.Series(arrs["sk_tau"]))),
        sen_median_per_yr=float(np.nanmedian(arrs["sen"])))
    print("F6", rep["F6"], round(time.time() - t0), "s", flush=True)

    # --------------------------------------------------------------- compare to historical rasters
    hist = {"F1_mean": "F1_NDVI_QA_mean.tif", "F2_clim_june_0120": "F2_NDVI_climatological_June.tif",
            "F3_anom_mean_0120": "F3_NDVI_anomaly_mean.tif", "F4_trend_mean": "F4_NDVI_trend_2x12MA.tif",
            "F5_resid_mean": "F5_NDVI_residual_mean.tif", "F6_mk_tau_trendseries_hist": "F6_MannKendall_tau.tif"}
    rep["historical_comparison"] = []
    for k, fn in hist.items():
        old = read_tif(os.path.join(P["ndvi_out"], fn))[0].ravel()[idx]
        rep["historical_comparison"].append(compare(feats[k], old, name=f"{k} vs {fn}"))
        print(rep["historical_comparison"][-1]["name"], rep["historical_comparison"][-1].get("pearson_r"),
              rep["historical_comparison"][-1].get("max_abs_diff"), flush=True)

    # --------------------------------------------------------------- labels (historical shifted vs corrected)
    import pandas as pd  # noqa
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import KBinsDiscretizer
    ff = pd.read_parquet(os.path.join(RES, "recalculated", "R1_fire_labels", "forest_fire_points_recalc.parquet"))
    a_, b_, c_, d_, e_, f_ = g["transform"][:6]
    r_hist = np.round((ff.latitude.values - f_) / e_).astype(np.int64); c_hist = np.round((ff.longitude.values - c_) / a_).astype(np.int64)
    r_cor = np.floor((ff.latitude.values - f_) / e_).astype(np.int64); c_cor = np.floor((ff.longitude.values - c_) / a_).astype(np.int64)
    pos_of = np.full(H * W, -1, np.int64); pos_of[idx] = np.arange(N)
    kmap = {k: i for i, k in enumerate(keys)}
    t_of = np.array([kmap.get((y, m), -1) for y, m in zip(ff.year.values, ff.month.values)])
    lab = {}
    for name, rr, cc in (("hist", r_hist, c_hist), ("corr", r_cor, c_cor)):
        ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        pix = np.full(len(ff), -1); pix[ok] = pos_of[rr[ok] * W + cc[ok]]
        ever = np.zeros(N, bool); ever[pix[pix >= 0]] = True
        lab[name] = dict(pix=pix, ever=ever)
        rep[f"fire_ever_pixels_{name}"] = int(ever.sum())
    # v2 training partition (random 65/15/20 stratified by corrected label, seed 42) -- defined here, reused downstream
    from sklearn.model_selection import train_test_split
    valid_px = np.isfinite(feats["v2_F1_mean"])
    vid = np.flatnonzero(valid_px)
    y_v = lab["corr"]["ever"][vid].astype(int)
    trv, te = train_test_split(vid, test_size=0.20, stratify=y_v, random_state=42)
    tr_, va_ = train_test_split(trv, test_size=0.15 / 0.80, stratify=lab["corr"]["ever"][trv].astype(int), random_state=42)
    split = np.full(N, -1, np.int8); split[tr_] = 0; split[va_] = 1; split[te] = 2
    np.save(os.path.join(OUT, "v2_random_split_india_idx.npy"), split)
    np.save(os.path.join(OUT, "india_pixel_index.npy"), idx)

    anoms = {"hist": np.load(os.path.join(TMP, "anom_0120.npy"), mmap_mode="r"),
             "v2": np.load(os.path.join(TMP, "anom_v2.npy"), mmap_mode="r")}

    def cvsi_mean_and_pm(k, anom):
        cv = np.full((T, N), np.nan, np.float32)
        for t in range(k, T):
            win = np.asarray(anom[t - k:t])
            stress = np.where(win < 0, -win, 0.0)
            nv = np.isfinite(win).sum(0)
            cv[t] = np.where(nv >= max(1, k // 2), np.nansum(stress, 0), np.nan)
        return cv
    mi = {"hist": {}, "corr": {}, "corr_trainonly": {}}
    cv_means = {"hist": {}, "v2": {}}
    logged = {}
    if RESUME:
        import ast as _ast
        for line in open(os.path.join(RES, "logs", "recalc_04_ndvi_run1.log"), encoding="utf-8", errors="ignore"):
            if line.startswith("CVSI k"):
                kk = int(line.split()[2]); logged[kk] = _ast.literal_eval(line.split(" ", 3)[3].strip())
        for n in mi:
            mi[n] = {kk: (logged[kk][n], None) for kk in logged}
        rep["F7_mi_note"] = "MI means taken from run-1 log (identical code/seeds; run 1 crashed later at LISA); SDs not retained"
        for which in ("hist", "v2"):
            cv_means[which][8] = np.nanmean(cvsi_mean_and_pm(8, anoms[which]), axis=0)
    for k in (range(1, 13) if not RESUME else []):
      for which in ("hist", "v2"):
        cv = cvsi_mean_and_pm(k, anoms[which])
        cv_means[which][k] = np.nanmean(cv, axis=0)
        todo = (("hist", "hist", False),) if which == "hist" else (("corr", "corr", False), ("corr_trainonly", "corr", True))
        for name, labset, train_only in todo:
            pix = lab[labset]["pix"]; ever = lab[labset]["ever"]
            sel = (pix >= 0) & (t_of >= k)
            if train_only:
                sel &= split[np.maximum(pix, 0)] == 0
            pv = cv[t_of[sel], pix[sel]]; pv = pv[np.isfinite(pv)]
            neg_pool = np.flatnonzero(~ever & valid_px & ((split == 0) if train_only else True))
            vals = []
            for rep_i in range(5):
                rng = np.random.RandomState(42 + rep_i)
                pp = rng.choice(neg_pool, size=len(pv) * 3, replace=True)
                tt = rng.randint(k, T, size=len(pp))
                nv = cv[tt, pp]; nv = nv[np.isfinite(nv)][:len(pv)]
                x = np.concatenate([pv, nv]).reshape(-1, 1); y = np.r_[np.ones(len(pv)), np.zeros(len(nv))]
                xb = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile").fit_transform(x).ravel()
                vals.append(mutual_info_score(y, xb))
            mi[name][k] = (float(np.mean(vals)), float(np.std(vals)))
        del cv
      print("CVSI k", k, {n: round(mi[n][k][0], 5) for n in mi}, flush=True)
    rep["F7_mi_sweep"] = mi
    rep["F7_kstar"] = {n: int(max(mi[n], key=lambda kk: mi[n][kk][0])) for n in mi}
    old7 = read_tif(os.path.join(P["ndvi_out"], "F7_CVSI_k8.tif"))[0].ravel()[idx]
    rep["historical_comparison"].append(compare(cv_means["hist"][8], old7, name="F7 CVSI k=8 mean vs F7_CVSI_k8.tif"))
    feats["F7_cvsi_k8"] = cv_means["hist"][8]
    feats["v2_cvsi_kstar_trainonly"] = cv_means["v2"][rep["F7_kstar"]["corr_trainonly"]]

    # --------------------------------------------------------------- F9 theta* (historical algorithm)
    from scipy.optimize import minimize
    from scipy.special import expit
    def nll(pr, x, y):
        a1, b1, a2, b2, th = pr
        pp = np.clip(np.where(x <= th, expit(a1 + b1 * x), expit(a2 + b2 * x)), 1e-7, 1 - 1e-7)
        return -np.sum(y * np.log(pp) + (1 - y) * np.log(1 - pp))
    def theta_fit(xv, yv, seed=42, max_n=200000):
        ok = np.isfinite(xv); xv, yv = xv[ok], yv[ok]
        rng = np.random.RandomState(seed)
        pi, ni = np.where(yv == 1)[0], np.where(yv == 0)[0]
        sel = np.r_[rng.choice(pi, min(len(pi), max_n // 2), replace=False), rng.choice(ni, min(len(ni), max_n // 2), replace=False)]
        xs, ys = xv[sel], yv[sel]
        best = (np.inf, None)
        floor_n = max(1, int(0.01 * len(xs)))
        for th0 in np.linspace(np.percentile(xs, 10), np.percentile(xs, 90), 25):
            r = minimize(nll, [0., -1., 0., -0.5, th0], args=(xs, ys), method="Nelder-Mead",
                         bounds=[(None, None)] * 4 + [(-0.2, 1.0)], options={"maxiter": 500, "xatol": 1e-4})
            nb = int((xs <= r.x[4]).sum())
            if nb < floor_n or len(xs) - nb < floor_n:
                continue
            if r.fun < best[0]:
                best = (r.fun, float(r.x[4]))
        return best[1]
    # historical used the full grid (incl. NaN) flattened; restricting to India pixels is identical after NaN removal
    rep["F9_theta"] = dict(hist_labels=theta_fit(feats["F1_mean"], lab["hist"]["ever"].astype(int)),
                           corrected_labels=theta_fit(feats["v2_F1_mean"], lab["corr"]["ever"].astype(int)),
                           corrected_trainonly=theta_fit(np.where(split == 0, feats["v2_F1_mean"], np.nan), lab["corr"]["ever"].astype(int)))
    print("F9", rep["F9_theta"], flush=True)

    # --------------------------------------------------------------- F8 Moran / LISA
    import libpysal
    from esda.moran import Moran, Moran_Local
    F1g = np.full(H * W, np.nan, np.float32); F1g[idx] = feats["F1_mean"]; F1g = F1g.reshape(H, W)
    F1v = np.full(H * W, np.nan, np.float32); F1v[idx] = feats["v2_F1_mean"]; F1v = F1v.reshape(H, W)
    co = F1g[::8, ::8].astype(np.float64); Hc, Wc = co.shape; vm = np.isfinite(co)
    filled = co.copy()
    for r in range(Hc):
        rv = filled[r, vm[r]]; filled[r, ~vm[r]] = rv.mean() if rv.size else 0.0
    wl = libpysal.weights.lat2W(Hc, Wc, rook=False); wl.transform = "R"
    m_hist = Moran(filled.ravel(), wl, permutations=0)
    # v2: 8x8 block MEAN (not subsample) and India-only cells
    Hb, Wb = H // 8, W // 8
    blk = np.nanmean(F1v[:Hb * 8, :Wb * 8].reshape(Hb, 8, Wb, 8), axis=(1, 3))
    vb = np.isfinite(blk)
    wfull = libpysal.weights.lat2W(Hb, Wb, rook=False)
    keep = np.flatnonzero(vb.ravel())
    wsub = libpysal.weights.w_subset(wfull, keep.tolist())
    wsub.transform = "R"
    yb = blk.ravel()[keep].astype(np.float64)
    m_v2 = Moran(yb, wsub, permutations=999)
    rep["F8_moran"] = dict(hist_method_I=float(m_hist.I), hist_method_z=float(m_hist.z_norm),
                           hist_n_cells=int(Hc * Wc), hist_n_filled_nonindia=int((~vm).sum()),
                           v2_india_only_blockmean_I=float(m_v2.I), v2_z=float(m_v2.z_sim), v2_p_sim=float(m_v2.p_sim),
                           v2_n_cells=int(len(keep)), v2_islands=len(wsub.islands))
    lm = Moran_Local(yb, wsub, transformation="R", permutations=199, seed=42, n_jobs=1)
    q = np.where(lm.p_sim < 0.05, lm.q, 0).astype(np.float32)
    lisa_blk = np.full(Hb * Wb, np.nan, np.float32); lisa_blk[keep] = q
    lisa_full = np.repeat(np.repeat(lisa_blk.reshape(Hb, Wb), 8, 0), 8, 1)
    lisa_g = np.full((H, W), np.nan, np.float32); lisa_g[:Hb * 8, :Wb * 8] = lisa_full
    feats["v2_lisa_cluster"] = lisa_g.ravel()[idx]
    rep["F8_moran"]["v2_lisa_counts"] = {int(k): int((q == k).sum()) for k in range(5)}
    print("F8", rep["F8_moran"], flush=True)

    # --------------------------------------------------------------- save
    import pyarrow as pa, pyarrow.parquet as pqw
    tbl = {k: np.asarray(v, np.float32) for k, v in feats.items()}
    tbl["grid_index"] = idx.astype(np.int64)
    tbl["fire_ever_corrected"] = lab["corr"]["ever"].astype(np.int8)
    tbl["fire_ever_hist_shifted"] = lab["hist"]["ever"].astype(np.int8)
    tbl["v2_split"] = split
    pqw.write_table(pa.table(tbl), os.path.join(OUT, "ndvi_features_india_pixels.parquet"))
    rep["runtime_sec"] = time.time() - t0
    dump(rep, os.path.join(OUT, "R4_report.json"))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
