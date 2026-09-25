# 🌿🔥 NDVI Fire-Susceptibility Feature Pipeline — India (Step 2)

<!-- AUDIT-UPDATE-2026-09-25 -->
> ### Audit update (2026-09-25)
> This repository's step was recalculated independently from the raw data in a full end-to-end audit.
> **Corrected results, reproduction checks and audit code: [`AUDIT_2026-09-25.md`](AUDIT_2026-09-25.md)** and `audit_2026-09-25/`.
> Earlier text below is kept for the record (it also remains in the git history). Statements superseded by the audit:
>
> - **Moran's I 0.8322**: reproduced, but 67% of the cells were row-mean-filled non-India cells. India-only (8×8 block means): **I = 0.9456**.
> - **Mann–Kendall significance counts**: reproduce, but the tests are invalid (MK on a smoothed or seasonal series). Seasonal Kendall + FDR gives NDVI 3,552,278 greening / 72,305 browning; LST day 2,435,163 cooling; night 2,080,747 warming; DTR 3,273,301 narrowing.
> - **Anomaly-mean features** (climate, LST, NDVI) are degenerate: with a 2001–2020 baseline they equal the residue of the 26 out-of-baseline months. v2 replaces them with climatological levels.
<!-- AUDIT-UPDATE-2026-09-25 -->


**Notebook:** [`NDVI_ANALYSIS_WITH_FFP.ipynb`](NDVI_ANALYSIS_WITH_FFP.ipynb)

Derives 9 NDVI-based fire-susceptibility features plus a 10th feature that
rasterizes Step 1's real extracted fire points onto the NDVI grid — every
feature here is directly aligned, pixel and month, with Step 1's output.

**Reference methodology:** Biswas et al. (2025), *Environ. Sci. Pollut. Res.*, 32:4856–4878
**Data:** MOD13A3.061, monthly 1km NDVI, NASA AppEEARS
**Study period:** 1 Nov 2000 – 15 Dec 2022 — matches Step 1 exactly (266 months, zero gaps)

## Why this step, and how

NDVI is used here as a proxy for vegetation moisture/fuel condition: healthy,
water-rich canopy has high NDVI and burns less readily, while browning/drying
vegetation (falling NDVI, rising cumulative moisture deficit) is a well-established
precursor signal for fire risk in the remote-sensing fire-danger literature (e.g.
the NDVI-based fuel-moisture and fire-danger indices reviewed in Chuvieco et al.,
*Remote Sensing of Environment*, 2004, and the vegetation-condition components of
national fire-danger rating systems). This step also does double duty as the
pipeline's geometric foundation: because NDVI is the first full-resolution national
raster product processed (1km, 3641×3504, EPSG:4326), its pixel grid becomes the
common reprojection target for every later step (LST in Step 3, FLDAS climatic
variables and land cover in Step 4, terrain/accessibility in Step 5, and the
integrated stack in Step 6) — one grid defined once, rather than each step
re-deriving its own and risking misalignment. Each of the 9 features exists for a
distinct conceptual reason: **climatology** (F2) establishes what "normal" NDVI
looks like per month/pixel; **anomaly** (F3) measures deviation from that normal,
i.e. abnormal stress; **trend** (F4) and **residual** (F5) separate a slow-moving
vegetation-health trajectory from short-term noise; **Mann-Kendall τ** (F6) tests
whether that trend is statistically real rather than random drift; **CVSI** (F7)
accumulates recent negative anomalies into a single pre-fire stress score, since
fire risk depends on antecedent dryness, not one month's snapshot; **LISA** (F8)
identifies spatial clusters of degraded vegetation that a purely per-pixel feature
would miss; and the **breakpoint θ\*** (F9) converts continuous NDVI into a single,
empirically fire-relevant threshold usable as a binary risk indicator. Downstream,
Steps 3–6 reproject their own rasters onto this grid and Step 6 joins all nine
NDVI-derived columns (plus the Step 1 fire-count raster) into the integrated
feature table that Steps 7–8 train models on.

## Comparison against Biswas et al. (2025)

Biswas et al. use a single monthly NDVI value from **MODIS-Terra MOD13C2 v006 at
0.05° (~5.5 km)** resolution as one input among their 15 predictors — in their
MaxEnt model it is the single most important predictor (22.3% permutation
importance / 28.4% percent contribution, their Table 3), but it enters the model
as one raw number per pixel-month, with no further temporal or spatial decomposition.
This step differs in both resolution and depth:

- **Resolution**: MOD13A3.061 at 1 km native grid here vs. MOD13C2 v006 at 0.05°
  (~5.5 km) in Biswas et al. — roughly 30× finer per-pixel area.
- **Temporal decomposition**: this step separates the one raw NDVI signal into
  climatology, anomaly, a GPU-vectorized classical 2×12-month moving-average
  trend/residual decomposition, and a GPU-vectorized Mann-Kendall τ significance
  test — none of which appear anywhere in Biswas et al.'s MaxEnt-only treatment.
- **A fire-data-driven antecedent-stress index (CVSI)**: an optimal cumulative-lag
  vegetation-stress index whose lag k\* (=8 months) is chosen by real mutual
  information against Step 1's actual fire/no-fire occurrence, not assumed from
  the literature or fit as a free MaxEnt parameter — a genuinely novel construct
  absent from Biswas et al. entirely.
- **Spatial clustering (LISA)**: Local Moran's I identifies High-High/Low-Low/
  High-Low/Low-High vegetation clusters, capturing neighborhood-scale
  fragmentation/degradation patterns a per-pixel-only feature set cannot.
- **An empirically fit fire-relevant threshold (θ\*)**: a piecewise-logistic
  breakpoint fit directly on real fire/no-fire pixel labels (nationally and per
  biogeographic zone), rather than an arbitrary or literature-assumed NDVI cutoff.

None of the last three (CVSI's data-driven lag, LISA, and the fitted breakpoint)
exist in Biswas et al.'s methodology — they are this step's genuine methodological
contributions beyond replicating a finer-resolution version of their single NDVI
predictor.

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
  - Significant browning pixels (p<0.05): ~~150,108 (pre-mask)~~ **147,206** (current, post-boundary-mask — see 2026-08-21 update below)
  - Significant greening pixels (p<0.05): ~~3,748,043 (pre-mask)~~ **3,731,210** (current, post-boundary-mask)
- **Step 1 → Step 2 link**: all 541,545 Step 1 fire points fell inside the NDVI grid bounds and matched an NDVI month (100%) — confirms the two steps' periods and geographic clips are consistent. **270,655** distinct pixels (2.12% of the grid) had ≥1 fire detection. (Unaffected by the boundary-mask fix below — Step 1's points were already India-forest-filtered.)
- **CVSI optimal lag** (chosen by mutual information with real fire occurrence, not a proxy).
  The sweep range was extended from k=1..6 to k=1..12 after the original run selected k=6 —
  the edge of the tested range — with MI still rising, making it an unverified boundary
  result. The extended sweep confirms a genuine **interior** optimum at **k\*=8**: MI rises
  through k=8 then falls off for k=9..12, so k\*=8 is not itself a boundary artifact.
  Table below is the **current, post-boundary-mask** sweep (see 2026-08-21 update) — MI
  values shifted marginally from the pre-mask run (e.g. k=8 was 0.01246, now 0.01257) because
  the background/fire pixel population used for MI no longer includes neighboring-country
  pixels; the selected lag itself (k\*=8, interior optimum) is unchanged either way.

  | k (months) | MI score |
  |---:|---:|
  | 1 | 0.00152 |
  | 2 | 0.00049 |
  | 3 | 0.00100 |
  | 4 | 0.00333 |
  | 5 | 0.00690 |
  | 6 | 0.00948 |
  | 7 | 0.01157 |
  | **8** | **0.01257** ← selected (interior optimum) |
  | 9 | 0.01036 |
  | 10 | 0.00936 |
  | 11 | 0.00831 |
  | 12 | 0.00739 |

- **Global Moran's I** (mean NDVI, stride-8 coarsened grid, current post-boundary-mask run):
  **I = 0.8322**, z = 742.105, p ≈ 0 — strong positive spatial autocorrelation (forest patches
  cluster). LISA (199 permutations, p<0.05): 13,412 High-High (dense forest core), 9,254
  Low-Low (degraded/sparse forest), 197 Low-High, 77 High-Low (fragment/WUI-risk pixels).
  (Pre-mask run reported I = 0.8925, z = 795.86, HH=13,566/LL=9,637/LH=228/HL=77 — superseded,
  see 2026-08-21 update below; kept here only for change-tracking.)
- **NDVI–fire breakpoint θ\*** (piecewise logistic, real fire/no-fire labels, balanced case-control sampling).
  The optimizer (`scipy.optimize.minimize`, Nelder-Mead) bounds θ to NDVI's physically valid
  range ([-0.2, 1.0]) on every multi-start run, and any winning solution where one regime
  (x≤θ or x>θ) holds fewer than 1% of the sample is treated as a non-identifiable
  boundary/regime-collapse solution and excluded rather than reported as a numeric θ\*.
  **Table below is the current, post-boundary-mask run** (2026-08-21, commit `b7c7d3c`) —
  all six values, national and all five zones, come from that single re-execution
  (notebook cell execution counts 6→15 run sequentially in one kernel session, confirmed
  from the saved `.ipynb` outputs):

  | Zone | θ\* | Sample (fire / no-fire) | Notes |
  |---|---:|---|---|
  | All India | **0.535** | 100,000 / 100,000 | post-mask (was 0.529 pre-mask) |
  | Western Ghats | 0.484 | 16,335 / 100,000 | post-mask (was 0.482; 11/25 starts degenerate, correctly excluded — was 10/25 pre-mask) |
  | Northeast | 0.643 | 100,000 / 100,000 | post-mask (was 0.668 pre-mask) |
  | Central India | 0.506 | 81,298 / 100,000 | post-mask (was 0.504 pre-mask) |
  | Deccan | 0.498 | 18,954 / 100,000 | post-mask (was 0.497; 5/25 starts degenerate, correctly excluded — was 6/25 pre-mask) |
  | Himalayan | **0.530** | 27,007 / 100,000 | post-mask — supersedes the pre-mask −0.001 value below; sample size also shifted (27,007 vs 27,030 pre-mask), consistent with this zone bordering Nepal/Bhutan/China/Pakistan and being the most affected by the boundary fix |

  **Historical note — resolved pre-mask caveat (superseded)**: before the boundary mask was
  added, an *unbounded* fit put the Himalayan θ\* at −0.613, outside NDVI's valid range —
  diagnosed as a boundary regime-collapse (n_below=0, NLL surface provably flat across
  θ∈[-2.0, -0.1605]) that an unconstrained Nelder-Mead could land anywhere on. Bounding the
  optimizer (`bounds=[...,(NDVI_VALID_MIN, NDVI_VALID_MAX)]`) resolved that specific
  degeneracy and moved the (still pre-boundary-mask) Himalayan solution to θ\*=−0.001. That
  −0.001 value is **not** the current number — once the India-boundary mask was added (below),
  the same zone's real fire/no-fire label population changed enough (removing Nepal/Bhutan
  border pixels) that the Himalayan θ\* moved again, to **0.530** (see table above). Both
  fixes (optimizer bounding, then boundary masking) were genuine, sequential corrections to
  the same zone — not a discarded/wrong intermediate result, but also not the final number.

  **2026-08-21 update — India boundary masking added** (commit `b7c7d3c`): this was the one
  step in the whole pipeline with no India-boundary clipping at all — the raw NDVI grid's
  rectangular raster bounds silently included valid NDVI signal from Pakistan, Nepal,
  Bangladesh, Myanmar, Sri Lanka, and Bhutan in every downstream feature (climatology,
  anomaly, trend/residual decomposition, Mann-Kendall, CVSI, LISA, and the breakpoint fit
  above). Fixed with the same set-CRS(3857)/reproject(4326)/dissolve/rasterize convention
  used by every other step. Verified via full re-execution (notebook cell 12's printed
  output): **8,573,393 px (67.20% of the 12,758,064-px grid) fall outside the India
  boundary and are excluded; 4,184,671 px (32.80%) are geometrically inside India** — these
  two numbers are complementary and sum to the full grid. A separate, stricter **"valid
  land" filter** applied later in the pipeline (cell 24, ahead of the LISA coarsening step)
  additionally excludes persistent no-data/ocean pixels still present within the India
  boundary, bringing the usable count down to **4,161,009 valid land pixels** — it is *this*
  narrower, stricter count (not the raw geometric 4,184,671) that matches Step 6's
  independently-built in-India pixel count exactly. (An earlier version of this README
  conflated these two different pixel counts as if they summed with 8,573,393 to the full
  grid, which they do not — 4,184,671 does; 4,161,009 is a further-filtered subset. Corrected
  2026-09-23.)

  Effect on every feature above: **All-India θ\* shifted 0.529 → 0.535** (current output file
  is `F9_NDVI_below_threshold_0.535.tif`, not `_0.529.tif`); **CVSI optimal lag k\*=8
  unchanged** (only the underlying MI values shifted marginally, table above); **Mann-Kendall
  significant-pixel counts and Global Moran's I / LISA cluster counts also shifted** (both
  updated above) — these were not previously called out as boundary-mask-affected in this
  README, which was an omission fixed 2026-09-23, not a new re-run. **All five regional
  breakpoint zones were, in fact, re-run in the same 2026-08-21 execution** (contrary to what
  an earlier version of this README stated) — their updated values are in the table above;
  the Himalayan zone in particular moved substantially (−0.001 → 0.530), which is the single
  most consequential number this boundary-mask fix changed, since that zone borders the most
  countries (Nepal, Bhutan, China, Pakistan) excluded by the mask.

### Outputs (`NDVI_Fire_Susceptibility_Outputs/`)

10 features exported as GeoTIFF (F1–F10, `.tif`, not tracked in git — see
`.gitignore`) plus 7 summary plots (`.png`, tracked):

```
NDVI_Fire_Susceptibility_Outputs/
├── F1_NDVI_QA_mean.tif                    F6_MannKendall_tau.tif
├── F2_NDVI_climatological_June.tif        F7_CVSI_k8.tif
├── F3_NDVI_anomaly_mean.tif                F8_LISA_cluster.tif
├── F4_NDVI_trend_2x12MA.tif                F9_NDVI_below_threshold_0.535.tif
├── F5_NDVI_residual_mean.tif               F10_fire_count_Step1.tif
└── *.png                                   (validation, F1/F3, F4/F6, F7, F8, F9, F10 plots)
```

> **Downstream note (2026-08-15 fix, closed):** the CVSI GeoTIFF filename changed from
> `F7_CVSI_k6.tif` to `F7_CVSI_k8.tif` (k\* moved from 6 to 8 after extending the
> mutual-information sweep to a confirmed interior optimum — see Results above). Step 6
> (`Integrated_Analysis/Step6_Integrated_FireRisk_Analysis.ipynb`, renumbered 2026-08-19
> from Step 5 — see that repo's README) originally still hardcoded
> `'ndvi_cvsi_k6': 'F7_CVSI_k6.tif'` in its `ndvi_feature_files` dict, which would have
> **silently skipped the CVSI feature** (print a `WARNING: missing ... F7_CVSI_k6.tif` and
> drop the column) rather than error — that dict entry has since been updated to
> `'ndvi_cvsi_k8': 'F7_CVSI_k8.tif'`, confirmed present in the current notebook. The F9
> breakpoint file was never affected — Step 6 globs for `F9_NDVI_below_threshold_*.tif`
> rather than hardcoding the fitted threshold value in the filename.

## Citation

- Biswas, S. et al. (2025). *[see notebook header for full reference]*

## License

No license has been chosen yet for this repository's code. MOD13A3.061 NDVI
data is subject to NASA/USGS data-use terms.
