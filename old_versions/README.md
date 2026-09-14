# Old / Superseded Files

Kept for history — not current. The canonical notebook is `NDVI_ANALYSIS_WITH_FFP.ipynb`
(built by `build_ndvi_notebook.py`); canonical outputs live in `NDVI_Fire_Susceptibility_Outputs/`
as documented in the repo's own `README.md`.

| File / folder | Why it's here |
|---|---|
| `F7_CVSI_k6.tif` | Pre-optimal-lag CVSI raster (k=6). Superseded by k\*=8, found via a fuller lag sweep — k6 was an unverified boundary artifact of a sweep that stopped too early (per the repo's own `.gitignore` comment). The k8 file is what every downstream step consumes. |
| `NDVI_Novel_Analysis_FINAL_15.ipynb` (root copy) | An unexecuted duplicate of the canonical notebook (identical cell source, no outputs) — the literal, now-abandoned output filename of `build_ndvi_notebook.py` from an earlier naming convention. |
| `NDVI_NOVEL_METHODOLOGY/` | Contains two more, older notebook generations (`NDVI_Novel_Analysis_FINAL_15.ipynb` dated 2026-07-13, and the earliest prototype `NDVI_Novel_Analysis_Pipeline.ipynb`), plus a third, undocumented parallel methodology-writeup draft (`NDVI_Novel_Methodology_MATH_1.tex`/`.pdf`, `NDVI_Numerical_Calculations_1.tex`/`.pdf`). |
| `NDVI_ANALYSIS_100/` + `.zip` | An early LaTeX methodology draft (`NDVI_Novel_Methodology_MATH.tex`), one of three parallel, undocumented draft tracks found for the same writeup. Not referenced anywhere in `README.md`. |
| `NDVI_NOVEL_METHOD/` + `.zip` | A second, differently-named LaTeX methodology draft (`NDVI_Novel_Methodology_STANDALONE.tex`). Same situation as above. |
| `Cumulative Pre-Fire Vegetation Stress Index.png`, `Local Moran.png`, `Monthly NDVI Anomaly.png` | Standalone plot exports (2026-06-18) predating the 2026-08-21 India-boundary-masking fix. Superseded by the corrected, equivalent plots inside `NDVI_Fire_Susceptibility_Outputs/` (`F7_CVSI_map.png`, `F8_LISA_cluster_map.png`, `F1_F3_NDVI_Anomaly.png`). |

**Note on `README.md` itself** (not moved, out of scope for archiving, but worth fixing before
writing the methodology section): its "Results" section still reports the pre-fix breakpoint
θ\*=0.529 for All India and never mentions India-boundary masking, even though the latest
commit shifted this to θ\*=0.535 (`F9_NDVI_below_threshold_0.535.tif`).

Moved 2026-09-14.
