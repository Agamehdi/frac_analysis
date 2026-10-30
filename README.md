# FRAC HYPERION V2.2 | KAPPA Benchmark Matching

Updated from the user's **October 17, 2026 Frac Hyperion V2.1** Streamlit app, with its existing Monte Carlo / 2D / 3D / SPE pages preserved.

## Deploy to GitHub + Streamlit Cloud

Upload `streamlit_app.py`, `frac_physics.py`, `kappa_benchmark.py`, `requirements.txt` to the repository root. Streamlit Cloud entry point: **`streamlit_app.py`**. `test_kappa_benchmark.py` and this README are optional for deployment.

For local execution: `pip install -r requirements.txt` and `streamlit run streamlit_app.py`.

## New tab: KAPPA Benchmark & Match

The supplied KAPPA **single-zone benchmark** is loaded automatically and can be edited or replaced with a two-column CSV:

| Perm (mD) | KAPPA Oil Rate (stb/d) |
| ---: | ---: |
| 1 | 70 |
| 10 | 380 |
| 50 | 700 |
| 100 | 720 |

**Default interpretation:** KAPPA numbers are **incremental oil rates (post-frac minus pre-frac)**, as stated by the user. Switch to **Post-frac oil rate** if these numbers are actually *total* post-frac rates.

The comparison holds modal inputs constant except permeability and computes a **stabilized 1-zone flow rate** with the same radial-PI + grid-derived frac geometry formula as the Monte Carlo engine. It intentionally excludes uptime/downtime, depletion, frac technical-failure probability and well count. The default baseline mode in the new tab is **Mechanistic radial Darcy PI**, to make permeability sensitivities meaningful.

Display: KAPPA points versus the raw Hyperion curve, a matched curve, table of relative errors, MAPE, RMSE, maximum error, and the applied correction factor.

### Two matching options

- **Permeability-dependent PCHIP:** positive factor based on `KAPPA rate / raw model rate` at each anchor. Shape-preserving interpolation in `log(k)` and `log(factor)`, with **no extrapolation**. It necessarily matches the four points (in-sample); this does **not** prove physical accuracy.
- **Single global multiplier:** nonnegative least-squares factor through the origin, which reduces squared error but does not necessarily match every anchor.

Press **Save match for Monte Carlo** then go to **Monte Carlo Studio**, use the **same baseline model and grid resolution** as the KAPPA tab, select **Apply KAPPA-matched calibration**, and rerun. The change is OFF by default, affects *successful fracture* rates only, and leaves baseline production unchanged. The resulting forecast / cumulative increments can change; original uncalibrated solver remains available. A saved match cannot be used after changing the modal physics reference inputs until it is rebuilt. Calibration factors apply **only inside the submitted permeability interval**, otherwise the factor is 1.0.

### Engineering QC warnings

- KAPPA simulations are an engineering reference, not independent field observations.
- Four points alone do not identify the physical reason for saturation at 50–100 mD: vertical lift, rate caps, skin, limited fracture conductivity, etc. should be tested separately.
- Ensure **one fractured zone**, **net pay**, **reservoir pressure**, **Pwf**, **Bo**, **viscosity**, **pre- / post-frac skin**, **fracture length / conductivity** are aligned between both models.
- This empirical start-rate correction is not a rigorous history match of fracture geometry, multiphase fluid flow, WC, wellhead/choke constraints, pressure support, frac degradation or decline. **Forecast and reserves still require independent verification.**

## Python numerical tests

Run `python -m pytest -q test_kappa_benchmark.py` (requires pytest). The tests check the 4 anchor points, both rate definitions, MC propagation, no extrapolation and stale-calibration protection.
