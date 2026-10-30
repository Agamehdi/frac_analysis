"""Single-zone KAPPA rate benchmark and transparent empirical calibration.

KAPPA points here are user-supplied *model* outputs, not field measurements.
A shape-preserving, log-permeability correction reproduces the anchor points
but is not a new reservoir/fracture-physics formulation. No extrapolation.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from frac_physics import GridLookup, cached_grid_lookup

DEFAULT_BENCHMARK = pd.DataFrame({
    'Perm (mD)': [1.0, 10.0, 50.0, 100.0],
    'KAPPA Oil Rate (stb/d)': [70.0, 380.0, 700.0, 720.0],
})
POST = 'Post-frac oil rate'
INCREMENTAL = 'Incremental oil rate'
GLOBAL = 'Single global multiplier'
LOCAL = 'Permeability-dependent (PCHIP)'


def check_benchmark(table: pd.DataFrame) -> pd.DataFrame:
    needed = ['Perm (mD)', 'KAPPA Oil Rate (stb/d)']
    if not set(needed).issubset(table):
        raise ValueError('Benchmark CSV requires columns: ' + ', '.join(needed))
    data = table[needed].copy()
    for name in needed:
        data[name] = pd.to_numeric(data[name], errors='coerce')
    if data.isna().any().any() or not np.isfinite(data[needed].to_numpy(float)).all():
        raise ValueError('All benchmark permeability and rate values must be numeric and finite.')
    if len(data) < 2 or len(data) > 50:
        raise ValueError('Provide 2 to 50 benchmark points.')
    if (data['Perm (mD)'] <= 0).any() or (data['KAPPA Oil Rate (stb/d)'] <= 0).any():
        raise ValueError('Permeability and target oil rate must be greater than zero.')
    data = data.sort_values('Perm (mD)').reset_index(drop=True)
    if data['Perm (mD)'].duplicated().any():
        raise ValueError('Duplicate permeability points are not allowed.')
    return data


def _modal(table: pd.DataFrame) -> dict[str, float]:
    return {str(row['Key']):float(row['Most likely']) for _, row in table.iterrows()}


def reference_signature(table: pd.DataFrame, baseline_mode: str, grid_size: int) -> str:
    """A calibration must not silently survive input or reference-mode changes."""
    d = table[['Key', 'Most likely']].sort_values('Key')
    inputs = repr((tuple((str(r['Key']), float(r['Most likely'])) for _,r in d.iterrows()),
                  str(baseline_mode), int(grid_size)))
    return hashlib.sha256(inputs.encode()).hexdigest()


def reference_rates(table: pd.DataFrame, perms: Any, baseline_mode: str,
                    grid_size: int = 21, lookup: GridLookup | None = None) -> pd.DataFrame:
    """Stabilized single-zone theoretical start rates with no uptime/downtime/depletion.

    Shares the same PI denominator, lookup gain and initial q0 modes as run_mc.
    All modal parameters except permeability remain fixed as k varies.
    """
    x = _modal(table)
    k = np.atleast_1d(np.asarray(perms, dtype=float))
    if np.any(~np.isfinite(k)) or np.any(k <= 0):
        raise ValueError('Permeability must be finite and >0.')
    if x['pr_psi'] <= x['pwf_psi']:
        raise ValueError('Reservoir pressure must exceed flowing pressure to benchmark a flowing well.')
    n = int(grid_size)
    re = np.sqrt(x['area_ac'] * 43560 / np.pi)
    side = np.sqrt(x['area_ac'] * 43560)
    dy = side/n
    clean_den = max(np.log(re / x['rw_ft']) - .75, 1.)
    pre_den = clean_den + x['skin_pre']
    if pre_den <= .5:
        raise ValueError('Nonphysical baseline radial PI denominator.')
    if baseline_mode == 'Mechanistic radial Darcy PI':
        baseline = (.00708 * k * x['h_ft'] * (x['pr_psi'] - x['pwf_psi']) /
                    (x['mu_cp'] * x['bo'] * pre_den) * (1. - x['wc0']))
    elif baseline_mode == 'Measured rate anchor':
        baseline = np.full_like(k, x['q0_bopd'])
    else:
        raise ValueError('Unknown baseline mode.')
    lookup = lookup if lookup is not None else cached_grid_lookup(n)
    frac_length = min(x['xf_ft']/side, .49)
    conductance = (x['cf_mdft']/(k * dy)) * x['frac_height_frac']
    geo_gain = lookup.gain(np.full_like(k, frac_length), conductance)
    post_den = np.maximum(clean_den/geo_gain + x['skin_post_residual'], .5)
    gain = (pre_den/post_den) if (x['xf_ft'] > 0 and x['cf_mdft'] > 0) else np.ones_like(k)
    frac = baseline * gain
    return pd.DataFrame({'Perm (mD)': k, 'Base rate (stb/d)': baseline,
                         'Post-frac rate (stb/d)': frac,
                         'Incremental rate (stb/d)': frac-baseline,
                         'Frac PI multiplier': gain})


def calibration_factor(k: Any, cal: dict[str, Any]) -> np.ndarray:
    """Use factor=1 outside the provided k interval; never extrapolate blindly."""
    permeability = np.asarray(k, dtype=float)
    knots = np.asarray(cal['perms_md'], dtype=float)
    factors = np.asarray(cal['factors'], dtype=float)
    flat = permeability.ravel()
    answer = np.ones_like(flat, dtype=float)
    within = np.isfinite(flat) & (flat >= knots[0]) & (flat <= knots[-1])
    if cal['method'] == GLOBAL:
        answer[within] = float(cal['global_factor'])
    elif cal['method'] == LOCAL:
        interp = PchipInterpolator(np.log(knots), np.log(factors), extrapolate=False)
        answer[within] = np.exp(interp(np.log(flat[within])))
    else:
        raise ValueError('Unknown calibration method.')
    return answer.reshape(permeability.shape)


def fit_benchmark(reference: pd.DataFrame, raw: pd.DataFrame, definition: str,
                  method: str, signature: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if definition not in (POST, INCREMENTAL):
        raise ValueError('Unknown target definition')
    benchmark = check_benchmark(reference)
    src = 'Post-frac rate (stb/d)' if definition == POST else 'Incremental rate (stb/d)'
    model = np.asarray(raw[src], dtype=float)
    target = benchmark['KAPPA Oil Rate (stb/d)'].to_numpy(float)
    if len(model) != len(benchmark) or np.any(~np.isfinite(model)) or np.any(model <= 1e-10):
        raise ValueError('Reference model response must be positive at each KAPPA permeability.')
    scalar = float(np.dot(model, target) / np.dot(model, model))
    if scalar <= 0 or not np.isfinite(scalar):
        raise ValueError('Global calibration factor invalid')
    knots = benchmark['Perm (mD)'].to_numpy(float)
    ratios = target/model
    cal = {
        'perms_md': knots.tolist(), 'factors': ratios.tolist(), 'method': method,
        'global_factor': scalar, 'definition': definition, 'signature': signature,
    }
    calibrated_response = model * calibration_factor(knots, cal)
    out = benchmark.copy()
    out['Hyperion Raw (stb/d)'] = model
    out['Matched (stb/d)'] = calibrated_response
    out['Raw Error (%)'] = 100*(model - target)/target
    out['Matched Error (%)'] = 100*(calibrated_response - target)/target
    out['Applied Factor'] = calibration_factor(knots, cal)
    return cal, out


def performance(rows: pd.DataFrame, col: str) -> dict[str,float]:
    target = rows['KAPPA Oil Rate (stb/d)'].to_numpy(float)
    values = rows[col].to_numpy(float)
    return {'MAPE (%)':float(np.mean(np.abs(values-target)/target)*100),
            'RMSE (stb/d)':float(np.sqrt(np.mean((values-target)**2))),
            'Max APE (%)':float(np.max(np.abs(values-target)/target)*100)}
