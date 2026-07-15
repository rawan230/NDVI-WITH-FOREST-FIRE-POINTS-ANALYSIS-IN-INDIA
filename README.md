# 🌿🔥 NDVI Fire-Susceptibility Feature Pipeline — India (Step 2)

**Notebook:** [`NDVI_ANALYSIS_WITH_FFP.ipynb`](NDVI_ANALYSIS_WITH_FFP.ipynb)

Derives 9 NDVI-based fire-susceptibility features plus a 10th feature that
rasterizes Step 1's real extracted fire points onto the NDVI grid — every
feature here is directly aligned, pixel and month, with Step 1's output.

**Reference methodology:** Biswas et al. (2025), *Environ. Sci. Pollut. Res.*, 32:4856–4878
**Data:** MOD13A3.061, monthly 1km NDVI, NASA AppEEARS
**Study period:** 1 Nov 2000 – 15 Dec 2022 — matches Step 1 exactly (266 months, zero gaps)

## What changed vs. the previous version of this notebook

| # | Problem in previous version | Fix in this version |
|---|---|---|
| 1 | Paths pointed to `C:\Users\...\Downloads\...` (wrong machine/folder) | Paths point to this project's actual local folders |
| 2 | `FIRE_CSV` and `BIO_ZONES` were referenced but **never defined** — those cells would crash on a fresh kernel | Both defined in Step 2 (Configuration), `FIRE_CSV` points at Step 1's real output |
| 3 | STL decomposition used a **per-pixel Python loop** calling `statsmodels.STL` — 12.7M pixels; the run crashed after 37 minutes at 6.7% complete (projected ≈9+ hours) | Replaced with a **GPU-vectorized classical 2×12 moving-average decomposition** applied to the whole `(T,H,W)` array at once — seconds, not hours |
| 4 | Mann-Kendall trend test used a **per-pixel `scipy.stats.kendalltau` double loop** (12.7M calls) — never even reached because STL crashed first | Replaced with a **GPU-vectorized Mann-Kendall S-statistic** computed via an O(T) lag-sweep across the whole grid simultaneously |
| 5 | CVSI's "optimal lag k\*" was chosen via a **proxy (temporal variance)** because fire data wasn't wired in — comment literally says *"using simulated fire mask for demo"* | k\* is now chosen by real **mutual information between CVSI and actual Step 1 fire occurrence** |
| 6 | NDVI–fire breakpoint threshold used a **synthetic proxy label** (`NDVI < 0.35`) instead of real fire data | Fit against **real fire/no-fire pixel labels** rasterized from Step 1's 541,545 extracted points, nationally and per biogeographic zone |
| 7 | QA/pixel-reliability files were discovered but **never actually applied** to mask unreliable pixels, despite Feature 1 being labelled "QA-Filtered NDVI" | QA masking is now genuinely applied (reliability ∈ {Good, Marginal} kept; Snow/Ice/Cloud/Fill dropped) |
| 8 | A dead, commented-out duplicate of the Moran's I cell was left in the notebook and errored on execution | Removed |
| 9 | No GPU usage anywhere — pure CPU NumPy for every array op | GPU-accelerated (CuPy, auto-detect with NumPy fallback) for anomaly/climatology, decomposition, Mann-Kendall, and CVSI — same auto-detect pattern as Step 1 |
| 10 | NDVI files spanned 2000-03 to 2025-12; processed regardless of Step 1's fire-data study period | Filtered to the exact Nov 2000 – Dec 2022 window (266 months, zero gaps) |

## Novel contributions delivered by this step

| # | Feature | Basis | What makes it novel here |
|---|---|---|---|
| F1 | QA-filtered NDVI mean | pixel_reliability SDS | Genuinely QA-masked (not just claimed) |
| F2 | Climatological monthly mean | seasonal baseline, 2001–2020 | — |
| F3 | Monthly anomaly δ | δ = NDVI − μ̄⁽ᵐ⁾ | — |
| F4 / F5 | Trend + Residual decomposition | classical 2×12-MA, GPU-vectorized | Full-resolution, full-country decomposition that was **computationally infeasible** with the original per-pixel approach |
| F6 | Mann-Kendall τ (trend significance) | GPU lag-sweep S-statistic | Exact statistic at full 1km resolution nationwide, not a per-pixel serial loop |
| F7 | CVSI (pre-fire stress index) | optimal lag k\* | k\* chosen via **real mutual information with Step 1 fire points**, not a proxy |
| F8 | LISA cluster map | Local Moran's I | — |
| F9 | NDVI–fire breakpoint θ\* | piecewise logistic, national + zone-stratified | Fit on **real fire/no-fire labels** from Step 1, case-control balanced sampling |
| F10 | Fire occurrence raster | Step 1 points → NDVI grid | Makes the Step 1 → Step 2 link tangible and exportable |

## Notebook structure

| Step | What it does |
|---|---|
| 0 | Install required packages (checks each, only installs what's missing) |
| 1 | Imports & GPU detection (CuPy auto-detect, NumPy fallback) |
| 2 | Configuration — paths, study period, QA codes, biogeographic zones |
| 3 | File discovery & date parsing, filtered to the Step 1 study window |
| 4 | Read, QA-mask & stack NDVI into a `(T,H,W)` array |
| 4B | Quick visual validation (mean NDVI map, coverage map, national time series) |
| 5 | Feature 1 & 3 — climatological mean & monthly anomaly (GPU) |
| 6 | Features 4, 5 & 6 — GPU-vectorized trend/seasonal/residual decomposition + Mann-Kendall |
| 7 | Rasterize Step 1's real fire points onto the NDVI grid |
| 8 | Feature 7 — CVSI with a real, data-driven optimal lag k\* |
| 9 | Feature 8 — Global Moran's I + Local LISA cluster map |
| 10 | Feature 9 — NDVI–fire breakpoint threshold (real labels, national + zone-stratified) |
| 11 | Feature compilation & GeoTIFF export |
| 12 | Summary: novel feature comparison table |

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace "NDVI_ANALYSIS_WITH_FFP.ipynb"
# or open it in Jupyter/VS Code and run all cells top to bottom
```

Requires Step 1 to have already been run (`FIRE_CSV` points at its output:
`../Forest fire Extraction in INDIA(2000-2022)/Forest_Fire_Outputs/all_forest_fires_2000_2022.csv`).

### Data sources

| Data | Source | Included in repo? |
|---|---|---|
| MOD13A3.061 monthly NDVI, 1km, India | [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/) | ❌ (~4.5 GB — download separately) |
| Step 1 forest-fire points | `../Forest fire Extraction in INDIA(2000-2022)/Forest_Fire_Outputs/` | ❌ (regenerate via Step 1) |

## Results (2000-11-01 → 2022-12-15, 266 months)

- **NDVI stack**: 266 × 3641 × 3504 = 12,758,064 pixels/month, 13.57 GB, QA-masked (264/266 months had a QA layer). NaN fraction 69.9% (ocean + persistently cloud/snow-masked land).
- **Anomaly**: range [-1.181, 1.167], mean ≈ 0.0036 (correctly centered near zero).
- **Trend/seasonal/residual decomposition**: 254-step GPU moving-average loop in **2 seconds** (replacing a per-pixel loop that projected ≈9+ hours and crashed).
- **Mann-Kendall trend significance**: 265-step GPU lag-sweep in **3m37s** across the full national grid.
  - Significant browning pixels (p<0.05): **150,108**
  - Significant greening pixels (p<0.05): **3,748,043**
- **Step 1 → Step 2 link**: all 541,545 Step 1 fire points fell inside the NDVI grid bounds and matched an NDVI month (100%) — confirms the two steps' periods and geographic clips are consistent. **270,655** distinct pixels (2.12% of the grid) had ≥1 fire detection.
- **CVSI optimal lag** (chosen by mutual information with real fire occurrence, not a proxy):

  | k (months) | MI score |
  |---:|---:|
  | 1 | 0.00151 |
  | 2 | 0.00050 |
  | 3 | 0.00101 |
  | 4 | 0.00339 |
  | 5 | 0.00676 |
  | **6** | **0.00988** ← selected |

- **Global Moran's I** (mean NDVI, stride-8 coarsened grid): **I = 0.8925**, z = 795.86, p ≈ 0 — strong positive spatial autocorrelation (forest patches cluster). LISA (199 permutations): 13,566 High-High (dense forest core), 9,637 Low-Low (degraded/sparse forest), 228 Low-High, 77 High-Low (fragment/WUI-risk pixels).
- **NDVI–fire breakpoint θ\*** (piecewise logistic, real fire/no-fire labels, balanced case-control sampling):

  | Zone | θ\* | Sample (fire / no-fire) |
  |---|---:|---|
  | All India | **0.529** | 100,000 / 100,000 |
  | Western Ghats | 0.482 | 16,335 / 100,000 |
  | Northeast | 0.668 | 100,000 / 100,000 |
  | Central India | 0.504 | 81,298 / 100,000 |
  | Deccan | 0.497 | 18,954 / 100,000 |
  | Himalayan | −0.613 ⚠️ | 27,030 / 100,000 |

  **Caveat**: the Himalayan zone's fitted θ\* falls outside NDVI's valid
  range (−0.2 to 1.0), indicating the optimizer hit a degenerate boundary
  solution for that zone (likely driven by snow/rock cover and a different
  fire-NDVI relationship than the other zones). Treat the national and
  other zone thresholds as reliable; the Himalayan figure needs a follow-up
  look (e.g. a bounded optimizer, or dropping permanently snow-covered
  pixels before fitting) before it's used downstream.

### Outputs (`NDVI_Fire_Susceptibility_Outputs/`)

10 features exported as GeoTIFF (F1–F10, `.tif`, not tracked in git — see
`.gitignore`) plus 7 summary plots (`.png`, tracked):

```
NDVI_Fire_Susceptibility_Outputs/
├── F1_NDVI_QA_mean.tif                    F6_MannKendall_tau.tif
├── F2_NDVI_climatological_June.tif        F7_CVSI_k6.tif
├── F3_NDVI_anomaly_mean.tif                F8_LISA_cluster.tif
├── F4_NDVI_trend_2x12MA.tif                F9_NDVI_below_threshold_0.529.tif
├── F5_NDVI_residual_mean.tif               F10_fire_count_Step1.tif
└── *.png                                   (validation, F1/F3, F4/F6, F7, F8, F9, F10 plots)
```

## Citation

- Biswas, S. et al. (2025). *[see notebook header for full reference]*

## License

No license has been chosen yet for this repository's code. MOD13A3.061 NDVI
data is subject to NASA/USGS data-use terms.
