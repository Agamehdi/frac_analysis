# FRAC HYPERION — V1

**GitHub-ready Streamlit Monte Carlo Fracture Productivity Screening App**

A transparent, educational engineering screening tool for probabilistic oil-production uplift from a **single bi-wing hydraulic fracture in each of multiple vertical wells**. Supports variable well counts, distributions for every input, 1,000+ runs, SPE-linked methodology, a computed 2D pressure grid, a 3D conceptual fracture view, and full CSV exports. **Not a commercial REVEAL simulator, a fluid-treatment fracture-design simulator, or a reserves booking package.**

## Files (upload ALL to the root of a GitHub repository)

| File | Purpose |
|---|---|
| `streamlit_app.py` | Full Streamlit UI and four pages |
| `frac_physics.py` | Numerical grid solver and Monte Carlo core |
| `requirements.txt` | Packages for Streamlit Cloud |
| `sample_ranges.csv` | User-editable range template |
| `test_frac_physics.py` | Numerical tests (optional in production) |
| `README.md` | Instructions / methods |
| `LICENSE` | MIT license |
| `preview_2D_grid.png` | Computed grid before/after frac preview |
| `preview_MC_cloud.png` | Synthetic 1,000-run cloud preview |

No nested folder structure is required for deployment. `.gitignore` is optional.

## Fastest deployment: GitHub → Streamlit Community Cloud

1. Create a **new public GitHub repository**, e.g. `frac-hyperion`.
2. Choose **Add file → Upload files**; drag the **individual files** from the extracted ZIP into the GitHub upload page (not the ZIP itself). Commit to the `main` branch.
3. Open [share.streamlit.io](https://share.streamlit.io/) and select **Create app** → **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and app entry point **`streamlit_app.py`**.
5. Deploy. Streamlit Cloud will install `requirements.txt` automatically.
6. In the app, edit all Low / Most likely / High ranges, set well count range, and press **Run Monte Carlo Forecast**.

For local development:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

For testing:

```bash
python -m pip install pytest
python -m pytest -q test_frac_physics.py
```

## Pages

1. **Monte Carlo Studio** — editable table for every parameter, one-click run, 100–3,000 MC runs, adjustable well-count range, forecasts, all-path clouds, P90/P50/P10 tables, S-curve, Spearman descriptive sensitivity, zip export.
2. **2D Fracture Grid** — independent numerical solve of a **Cartesian Darcy transmissibility grid** with a central well and one finite-conductivity bi-wing fracture; heatmaps before and after frac.
3. **3D Fracture View** — conceptual 3D well, reservoir grid, and vertical fracture plane. Geometry visualization only, not a 3D flow solve.
4. **Guidelines & SPE References** — classical equations, assumptions, QA checklist, literature links, and limitations.

## Important units

- k: **mD**; h, half-length, well radius: **ft**; area: **acres**.
- P, Pwf: **psia**; viscosity: **cP**; Bo: **reservoir bbl / stock-tank bbl**.
- Fracture conductivity: **mD·ft = kf × wf**; not the fracture permeability itself.
- Oil rates: **bopd**; cumulative oil: **stb**.
- All suggested inputs are **fictional illustrative starting values**, not actual wells.

## What the model computes

**Grid-derived geometric improvement (steady-state):**

For a homogeneous isotropic square grid, normalized matrix link transmissibility is 1; when a horizontal connection is intersected by a planar fracture, the extra normalized conductance is:

```
extra dimensionless link conductance = (Cf / (k * dy)) * (hf / h)
```

with a partial-cell overlap correction. Boundary cells are held at `p=1`, the well cell at `p=0`. The sparse finite-volume Laplacian calculates the well's total dimensionless Darcy inflow for (a) unfractured and (b) fractured cases. Their quotient is the geometric PI gain. The MC engine precomputes/interpolates responses over `(half-length/domain-size, log10(extra link conductance))`; an exact 2D solve can be explored on the grid page. Only horizontal x-aligned fracture orientation and isotropic matrix are included.

**Combining fracture gain with skin:**

Let `Dclean = ln(re/rw) - 0.75`. The baseline denominator is `Dclean + Spre`. The modeled fracture denominator is `Dclean/Ggrid + Spost-residual`. Therefore:

```
frac PI / prefrac PI = (Dclean + Spre) / (Dclean/Ggrid + Spost-residual)
```

This bridge between a coarse-grid response and the radial pressure-drop formula is **approximate, not a published exact Cinco-Ley solution**. Use PTA calibration and measured frac conductivity when possible.

**Base-case PI:**

```
q0 = 0.00708 * k_md * h_ft * (Pr - Pwf) / (mu_cp * Bo * [ln(re/rw)-0.75+Spre])
```

and multiplied by `(1 - wc0)` when the optional mechanistic q0 mode is selected. Default mode uses the **measured q0 oil-rate anchor**, then scales it by the pressure ratio for future months.

**Time-dependent assumptions:**

- Conductivity is reduced by `retention_5y ** (time_months / 60)`.
- Pressure storage per well = `PV_rb × ct`, where `PV_rb = 7758 × Area_ac × h_ft × phi`.
- Each month's reservoir-fluid pressure depletion is proportional to `oil_stb × Bo / (PV × ct)` and adjusted by `(1 - pressure_support)`. No explicit injected-water material-balance network.
- Water cut is an empirical oil-cut curve, not a multiphase flow simulation.
- Treatment downtime means baseline wells still produce while the frac scenario is offline; even failed treatments can produce negative incremental oil.
- Every well has its own independent drainage tank. **There is no well interference or fracture overlap across wells.**

**Paired Monte Carlo methodology:** sample each scenario's uncertain properties once and use exactly the same sample for baseline and frac paths. The output `incremental = frac − baseline` is calculated **per realization**. Then calculate percentiles from the incremental values, not from the subtraction of independently calculated case percentiles.

`P90`: 10th percentile (downside). `P50`: median. `P10`: 90th percentile (upside). These are probability-of-exceedance style labels, not 1P/2P/3P proved reserves.

## Scope and limitations

- 2D grid is a *stabilized pressure-response* solution, not a pressure-vs-time reservoir simulator. Cumulative forecast uses a lumped compressible tank.
- A grid-like representation is conceptually related to finite-volume / EDFM approaches, but does **not replicate proprietary Petroleum Experts REVEAL, DFN, local grid refinement, or 3D fracture propagation**.
- No oil-water/gas relative permeability, injector/producer coupling, solution gas evolution, fluid cleanup, mechanical fracture growth, acid transport, or proppant design.
- No facility constraints, economic optimization, or simultaneous frac-job scheduling.
- Sampling assumes independent wells and independent sampled rock/fluid inputs other than shared well-count logic. Shared geological uncertainties and correlations should be added for field development forecasts.
- For reservoir-pressure crossing bubble point, geomechanically complex reservoirs, or aggressive waterfloods, run a fully coupled calibrated numerical simulator.
- Beware coarse-grid half-length aliasing; confirm grid-size convergence. Results should be subjected to field-level engineering validation.

## Academic references (methods inspiration, not reproduction)

- Prats, M. (1961), *Effect of Vertical Fractures on Reservoir Behavior—Incompressible-Fluid Case*, SPE-1575-G, https://doi.org/10.2118/1575-G
- Cinco-Ley and Samaniego (1981), *Transient Pressure Analysis for Fractured Wells*, SPE-7490-PA, https://doi.org/10.2118/7490-PA
- *Stabilized Productivity of a Hydraulically Fractured Well Producing at Constant Pressure*, SPE-88960-PA, https://doi.org/10.2118/88960-PA
- *Productivity Simulation of Hydraulically Fractured Wells Based on Hybrid Local Grid Refinement and Embedded Discrete Fracture Model*, https://www.sciencedirect.com/science/article/pii/S1876380420600532
- Petroleum Experts (2018) REVEAL feature document (high-level comparison only): https://www.petex.com/media/2452/feb2018_ipm11_ivm8_ifm5_enhancements.pdf

## Suggested next versions

- Well-specific CSV overrides, staggered frac dates, common reservoir-level correlated uncertainties.
- Oil/water relative permeability; two-phase finite-volume dynamic solve with injection and water breakthrough.
- Analytical fracture skin correlations calibrated to PTA and treatment/closure constraints.
- Ensemble history matching and Bayesian priors, separate aleatory and epistemic uncertainties.
- Economic breakeven/EV, discounting, capex, risk-weighted well selection, diagnostics.

## Verification status

The Python numerical engine has been syntax-checked and run through 10 unit tests. The Streamlit graphical application could not be executed inside the authoring environment because Streamlit was not installed and the package registry was inaccessible. Live Streamlit/Cloud deployment should be verified before relying on results.
