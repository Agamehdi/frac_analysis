# FRAC HYPERION V2.3 — Entire Monte Carlo Cloud → KAPPA Benchmark

Updated from FRAC HYPERION V2.2. Previous Monte Carlo, 2D Grid, 3D Fracture, SPE References and editable uncertainty inputs are preserved.

## Installation / deployment

Copy **all** these files to the root of the existing GitHub repository (replace old versions of files with identical names):

- `streamlit_app.py` — Streamlit UI, one-click matching and comparison plots.
- `frac_physics.py` — grid-based productivity and Monte Carlo, including the per-well calibration.
- `kappa_benchmark.py` — anchor validation, fitting and permeability interpolation.
- `requirements.txt` — dependency versions for Streamlit Cloud.

Cloud entry point is `streamlit_app.py`. Locally:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

`test_kappa_benchmark.py` and `test_cloud_match.py` are optional tests (requires `pytest`).

## The reported bug: disabled checkbox

In V2.2, **Apply KAPPA-matched calibration** was disabled until a match had been saved on the separate KAPPA tab **and** every modal input, baseline rate mode and grid resolution agreed with the saved match signature. This was confusing when Monte Carlo used the default `Measured rate anchor`, whereas KAPPA Benchmark used the default `Mechanistic radial Darcy PI`.

**V2.3 fix:** The checkbox is never disabled because of a missing/stale saved match. On request, the program fits the editable KAPPA points **using the actual Monte Carlo inputs**. No pre-saving on the KAPPA tab, matching mode manually or grid synchronization is required.

## One-click usage

1. Open **KAPPA Benchmark & Match** if you want to edit/import reference data. Default points are shown below. Default interpretation is *incremental rate*, not total post-frac rate.
2. Open **Monte Carlo Studio** and edit uncertainty parameters as before. Use **Mechanistic radial Darcy PI** for a permeability-dependent pre-frac PI; **Measured rate anchor** is still available and unchanged.
3. Optionally tick **Include full KAPPA permeability span** to make a *temporary* 1–100 mD `Log-triangular` permeability sampling distribution for both scenarios, leaving the user's uncertainty input table untouched. If unticked, the existing k distribution is unchanged (default 1–20 mD: the 50 and 100 mD anchors are outside its population).
4. Click **⚡ Match ENTIRE Monte Carlo Cloud to KAPPA**. This button automatically checks the calibration checkbox, fits the latest points and runs **two** models with identical seeds/input draws: original and KAPPA-matched. The standard **Run Monte Carlo** button also respects the manual checkbox.
5. Inspect the **KAPPA | Entire Cloud — Before vs After** charts, percentile comparison table, matched cloud, individual well scatter, and downloadable results.
6. Use **Show results in main charts → Original cloud** to switch back without deleting the matched results, or untick **Apply...** and click **Run Monte Carlo** to run original only.

### Default KAPPA single-zone inputs

| Permeability (mD) | Incremental oil (stb/d) |
| ---: | ---: |
| 1 | 70 |
| 10 | 380 |
| 50 | 700 |
| 100 | 720 |

The user can edit or upload a CSV in the **KAPPA Benchmark & Match** page. `Permeability-dependent (PCHIP)` fits these four *modal-condition* targets exactly, by construction. `Single global multiplier` is available but cannot generally fit every point.

## What "entire cloud" means technically

- Each Monte Carlo realization samples a different number of wells and their rock, fluid, reservoir and fracture inputs.
- Every **successful, physically productive fractured well** gets its permeability-specific calibrated correction during every forecast time step. Technical failures remain uncorrected.
- **Incremental-rate benchmarks** multiply the modeled *uplift* over the comparable unfractured rate, rather than multiplying the entire rate; **post-frac rate benchmarks** multiply the post-frac rate.
- With the corrected rate, the simulator **recalculates pressure depletion month by month**, total and incremental oil rates, cumulative oil, and all P90/P50/P10 statistics. This is **not** a cosmetic scaling of Plotly curves.
- The original and matched scenarios use exactly the same sampled inputs and random seed, so before/after comparisons are paired. Every realization contributes to percentile calculations, though only a subset of curve traces is plotted to keep the dashboard responsive.
- Correction is **disabled outside the reference permeability interval**, rather than extrapolating without evidence.
- The primary before/after overlay displays matched trajectories and original trajectories, plus all three percentile lines for both cases.
- Export includes matched rates, run-level outcomes, individual well results, the original forecast and before/after table, calibration details and benchmark points.

## Key limitations / QC

- The four KAPPA values are **simulator benchmark points for one zone**. They are not measured field history and have no decline/time-profile data.
- Matching **initial oil rate versus permeability** cannot calibrate the **time-dependent decline profile**, pressure support, fluid mobility / relative permeability, fracture propagation, conductivity retention, water-cut evolution, cleanup, VLP constraints or unsuccessful frac-job performance. Those are driven by the existing assumptions.
- Thus *all* forecast paths change due to a rate correction and recalculated depletion, but **it does not mean every point of every forecast path has been independently matched to KAPPA**. To match forecast shapes, supply KAPPA oil-rate-versus-time profiles at relevant permeabilities.
- Preserve consistency of net pay, reservoir pressure, Pwf, PVT, drawdown, pre/post skin, fracture geometry and rate definition. Use **Mechanistic radial Darcy PI** to benchmark relative permeability sensitivity when a fixed pre-frac measured rate across all k values would be inappropriate.
- The 700→720 stb/d flattening over 50→100 mD suggests a rate ceiling or efficiency decline, but four data points cannot identify its physical cause. Check KAPPA sensitivity and other constraints.
- This remains a single-phase fracture-productivity *screening* model, not a hydraulic fracture-propagation or multiphase transient simulator, nor a reserves certification tool.

## Validation

From this folder with `pytest` installed:

```bash
python -m pytest -q test_kappa_benchmark.py test_cloud_match.py
```

Nine tests cover anchor matching, interpretation as incremental vs post-frac, global/local matching, stale signature, no extrapolation, failure behavior, and entire Monte Carlo array / forecast recalculation. The code is syntax-checked. Full Streamlit front-end runtime testing was unavailable in the authoring environment because Streamlit could not be installed offline.
