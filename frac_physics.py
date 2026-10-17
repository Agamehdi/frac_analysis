"""Frac Hyperion v1: transparent 2-D transmissibility screening + MC forecasts.

NOT a hydraulic-fracture propagation, compositional, multiphase, or certified
reserves simulator. Models one vertical, bi-wing finite-conductivity fracture.

Grid: pressure 1 at outer cells, 0 in center well cell. Between neighboring
cells, normalized Darcy transmissibility is 1; fracture increases horizontal
transmissibility along its row by Cf/(k * dy) * (hf/h). Grid response PI ratio is
mapped into a radial productivity denominator. Forecast uses separate per-well
lumped fluid-expansion tanks for baseline and frac; no well interference.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

DAYS_PER_MONTH = 365.25 / 12.0

# GROUP, KEY, NAME, UNIT, LOW, MODE, HIGH, DIST, PURPOSE
PARAMETERS = [
    ('Rock', 'k_md', 'Matrix permeability', 'mD', 1., 6., 20., 'Log-triangular', 'Isotropic matrix permeability'),
    ('Rock', 'h_ft', 'Net pay', 'ft', 20., 50., 90., 'Triangular', 'Connected reservoir thickness'),
    ('Rock', 'phi', 'Porosity', 'fraction', .10, .17, .24, 'Triangular', 'Pore-volume fraction'),
    ('Rock', 'ct_psi', 'Effective total compressibility', '1/psi', .000008, .000015, .00003, 'Log-triangular', 'Lumped pressure depletion storage'),
    ('Rock', 'area_ac', 'Independent drainage area per well', 'acres', 80., 200., 360., 'Triangular', 'Independent tanks assumed; no interference'),
    ('Pressure', 'pr_psi', 'Initial average pressure', 'psia', 2600., 3200., 3800., 'Triangular', 'Average pressure at frac date'),
    ('Pressure', 'pwf_psi', 'Producing bottomhole pressure', 'psia', 1500., 2100., 2400., 'Triangular', 'Held constant for forecast'),
    ('Pressure', 'pressure_support', 'External pressure support fraction', '0-1', 0., .25, .60, 'Triangular', 'Fraction of produced withdrawal replaced in lumped tank'),
    ('PVT', 'mu_cp', 'Oil viscosity', 'cP', 1., 1.5, 2.5, 'Triangular', 'Used in mechanistic q0 calculation'),
    ('PVT', 'bo', 'Oil FVF', 'rb/stb', 1.08, 1.22, 1.40, 'Triangular', 'Used in storage depletion'),
    ('Well', 'rw_ft', 'Wellbore radius', 'ft', .25, .33, .50, 'Triangular', 'Used in radial PI denominator'),
    ('Well', 'skin_pre', 'Pre-frac skin', 'dimensionless', 0., 4., 8., 'Triangular', 'Pre-frac damage + skin'),
    ('Well', 'skin_post_residual', 'Post-frac residual skin', 'dimensionless', 0., .3, 1., 'Triangular', 'Additional residual skin beyond grid geometric enhancement'),
    ('Well', 'q0_bopd', 'Measured current oil rate', 'bopd', 60., 110., 170., 'Triangular', 'Initial baseline anchor (history-matched mode)'),
    ('Well', 'uptime', 'Operating uptime', '0-1', .75, .90, .98, 'Triangular', 'Operational uptime applied to forecast'),
    ('Water', 'wc0', 'Current water cut', 'fraction', .05, .15, .35, 'Triangular', 'Starting produced water fraction'),
    ('Water', 'wc_growth_ppyr', 'Baseline WC growth', 'fraction/year', 0., .02, .05, 'Triangular', 'Simple empirical rise in water cut'),
    ('Water', 'frac_extra_wc_pp', 'Frac-related additional WC', 'fraction', 0., .04, .15, 'Triangular', 'Gradual post-frac extra water-cut penalty'),
    ('Water', 'frac_water_delay_m', 'Water penalty onset', 'months', 3., 9., 18., 'Triangular', 'Months until extra water cut develops'),
    ('Frac', 'xf_ft', 'Fracture half-length', 'ft', 80., 200., 400., 'Triangular', 'Single planar bi-wing fracture'),
    ('Frac', 'cf_mdft', 'Initial fracture conductivity', 'mD.ft', 100., 900., 4500., 'Log-triangular', 'kf*w, NOT fracture permeability alone'),
    ('Frac', 'frac_height_frac', 'Fraction of pay contacted', '0-1', .50, .85, 1.0, 'Triangular', 'Fraction of thickness effectively connected to fracture'),
    ('Frac', 'retention_5y', 'Conductivity retained at 5 years', '0-1', .4, .75, .95, 'Triangular', 'Exponential conductivity decline factor'),
    ('Frac', 'frac_success_prob', 'Technical success probability', '0-1', .60, .82, .95, 'Triangular', 'Sampled per-well conditional success probability'),
    ('Frac', 'downtime_days', 'Frac job and cleanup downtime', 'days', 5., 12., 25., 'Triangular', 'Lost production days at start of forecast'),
]
COLS = ['Group', 'Key', 'Parameter', 'Unit', 'Low', 'Most likely', 'High', 'Distribution', 'Guidance']


def default_parameter_table() -> pd.DataFrame:
    return pd.DataFrame(PARAMETERS, columns=COLS)


def validate_parameters(df: pd.DataFrame) -> list[str]:
    problems = []
    expected = set(default_parameter_table()['Key'])
    have = set(df['Key'])
    for key in sorted(expected - have):
        problems.append(f'Missing parameter: {key}')
    for _, row in df.iterrows():
        key = str(row['Key'])
        try:
            lo, mid, hi = [float(row[x]) for x in ('Low', 'Most likely', 'High')]
            if not all(np.isfinite([lo, mid, hi])):
                raise ValueError('not finite')
            if not lo <= mid <= hi:
                problems.append(f'{key}: Low <= Most likely <= High is required')
            if row['Distribution'] not in ('Triangular', 'Log-triangular', 'Uniform', 'Fixed'):
                problems.append(f'{key}: invalid distribution')
            if row['Distribution'] == 'Log-triangular' and lo <= 0:
                problems.append(f'{key}: log distribution requires Low > 0')
            if key in ('phi', 'pressure_support', 'uptime', 'wc0', 'frac_extra_wc_pp', 'frac_height_frac',
                       'retention_5y', 'frac_success_prob') and (lo < 0 or hi > 1):
                problems.append(f'{key}: should be within [0,1]')
            if key in ('ct_psi', 'h_ft', 'area_ac', 'bo', 'mu_cp', 'rw_ft', 'pr_psi', 'pwf_psi', 'k_md') and lo <= 0:
                problems.append(f'{key}: must be > 0')
            if key in ('xf_ft', 'cf_mdft', 'downtime_days', 'frac_water_delay_m', 'wc_growth_ppyr') and lo < 0:
                problems.append(f'{key}: cannot be negative')
        except (ValueError, TypeError):
            problems.append(f'{key}: all three range values must be numeric')
    return problems


def sample_range(row: pd.Series, rng: np.random.Generator, shape):
    lo, mid, hi = float(row['Low']), float(row['Most likely']), float(row['High'])
    name = row['Distribution']
    if np.isclose(lo, hi, rtol=0, atol=1.e-15) or name == 'Fixed':
        return np.full(shape, mid, dtype=float)
    if name == 'Uniform':
        return rng.uniform(lo, hi, size=shape)
    if name == 'Log-triangular':
        return np.exp(rng.triangular(np.log(lo), np.log(mid), np.log(hi), size=shape))
    return rng.triangular(lo, mid, hi, size=shape)


@lru_cache(maxsize=10)
def _grid_geometry(n: int):
    if n < 9 or n % 2 != 1:
        raise ValueError('Grid dimension must be odd and at least 9')
    c = n // 2
    coords = [(r, x) for r in range(1, n - 1) for x in range(1, n - 1) if (r, x) != (c, c)]
    idmap = {pair: i for i, pair in enumerate(coords)}
    a = lil_matrix((len(coords), len(coords)), dtype=float)
    b = np.zeros(len(coords))
    for index, (r, x) in enumerate(coords):
        for rr, xx in ((r - 1, x), (r + 1, x), (r, x - 1), (r, x + 1)):
            a[index, index] += 1.
            if (rr, xx) in idmap:
                a[index, idmap[(rr, xx)]] -= 1.
            elif rr == 0 or rr == n - 1 or xx == 0 or xx == n - 1:
                b[index] += 1.  # fixed boundary p=1
    return a.tocsr(), b, idmap


def solve_grid(n: int, xf_fraction: float, enhanced_link_ratio: float, return_map=False):
    """Solve normalized steady grid; enhanced ratio=Cf/(k*dy)*(hf/h).

    Baseline wells modeled as a fixed-pressure center cell (not a Peaceman WI).
    Facies assumed isotropic and constant; no no-flow edges (boundary p=1).
    """
    a0, b, idmap = _grid_geometry(n)
    a = a0.tolil()
    c = n // 2
    frac = float(np.clip(xf_fraction, 0, .495))
    link_ratio = max(float(enhanced_link_ratio), 0.)
    # Segment between x-cell centers in coordinates normalized by square side.
    for left in range(1, n - 2):
        xleft, xright = (left - c) / n, (left + 1 - c) / n
        overlap = max(0., min(xright, frac) - max(xleft, -frac))
        coverage = np.clip(overlap * n, 0., 1.)
        added = link_ratio * coverage
        if added < 1e-14:
            continue
        u, v = (c, left), (c, left + 1)
        if u in idmap:
            i = idmap[u]
            a[i, i] += added
        if v in idmap:
            j = idmap[v]
            a[j, j] += added
        if u in idmap and v in idmap:
            a[i, j] -= added
            a[j, i] -= added
    psol = spsolve(a.tocsr(), b)
    p = np.ones((n, n), dtype=float)
    p[c, c] = 0.
    for coord, idx in idmap.items():
        p[coord] = psol[idx]
    # Four well-interface connections, including enhanced fracture x-links.
    q = 0.
    for r, x in ((c - 1, c), (c + 1, c), (c, c - 1), (c, c + 1)):
        g = 1.
        if r == c:
            halfstep_overlap = max(0., min(frac, 1. / n) - 0.)
            g += link_ratio * np.clip(halfstep_overlap * n, 0., 1.)
        q += g * p[r, x]
    if return_map:
        return q, p
    return q


@dataclass
class GridLookup:
    n: int = 21
    length_bins: int = 22
    conductivity_bins: int = 20

    def __post_init__(self):
        self.length_axis = np.linspace(0., .49, self.length_bins)
        self.log_link_axis = np.linspace(-4., 5., self.conductivity_bins)
        self.q_base = solve_grid(self.n, 0, 0.)
        ratios = np.ones((self.length_bins, self.conductivity_bins))
        for i, lf in enumerate(self.length_axis):
            for j, lc in enumerate(self.log_link_axis):
                if lf > 0:
                    ratios[i, j] = solve_grid(self.n, lf, 10 ** lc) / self.q_base
        self.ratios = ratios
        self.interp = RegularGridInterpolator((self.length_axis, self.log_link_axis), ratios,
                                               bounds_error=False, fill_value=None)

    def gain(self, length_fraction, conductance_ratio):
        lf = np.asarray(length_fraction, dtype=float)
        cr = np.asarray(conductance_ratio, dtype=float)
        lf, cr = np.broadcast_arrays(lf, cr)
        pair = np.column_stack((np.clip(lf.ravel(), self.length_axis[0], self.length_axis[-1]),
                                np.clip(np.log10(np.maximum(cr.ravel(), 1e-10)),
                                        self.log_link_axis[0], self.log_link_axis[-1])))
        v = self.interp(pair).reshape(lf.shape)
        return np.where((lf <= 0) | (cr <= 0), 1., np.maximum(v, 1.))


@dataclass
class Result:
    months: np.ndarray
    base_rate: np.ndarray
    frac_rate: np.ndarray
    incremental_rate: np.ndarray
    base_cum: np.ndarray
    frac_cum: np.ndarray
    incremental_cum: np.ndarray
    well_count: np.ndarray
    success_count: np.ndarray
    diagnostic: pd.DataFrame
    lookup: GridLookup


def summarize_curves(result: Result) -> pd.DataFrame:
    d = {'Month': result.months, 'Year': result.months / 12.}
    for name, values in [('Base Oil Rate (bopd)', result.base_rate),
                         ('Frac Oil Rate (bopd)', result.frac_rate),
                         ('Incremental Oil Rate (bopd)', result.incremental_rate),
                         ('Base Cum Oil (stb)', result.base_cum),
                         ('Frac Cum Oil (stb)', result.frac_cum),
                         ('Incremental Cum Oil (stb)', result.incremental_cum)]:
        d[name + ' P90'] = np.percentile(values, 10, axis=0)
        d[name + ' P50'] = np.percentile(values, 50, axis=0)
        d[name + ' P10'] = np.percentile(values, 90, axis=0)
        d[name + ' Mean'] = np.mean(values, axis=0)
    return pd.DataFrame(d)


def run_mc(table: pd.DataFrame, n_low: int, n_mode: int, n_high: int,
           n_runs: int = 1000, years: int = 5, seed: int = 42,
           q0_mode: str = 'Measured rate anchor', grid_size: int = 21,
           max_work: int = 300_000) -> Result:
    errs = validate_parameters(table)
    if errs:
        raise ValueError('; '.join(errs[:7]))
    if not (1 <= n_low <= n_mode <= n_high <= 250):
        raise ValueError('Well count Low <= Most Likely <= High in [1,250] required')
    if n_runs < 10 or years < 1 or years > 20:
        raise ValueError('At least 10 runs; forecast years 1-20')
    if n_runs * n_high > max_work:
        raise ValueError(f'Workload too large for Streamlit Cloud: runs × max wells must be <= {max_work:,}')
    rng = np.random.default_rng(seed)
    count = np.rint(rng.triangular(n_low, n_mode, n_high, size=n_runs)).astype(int) if n_low != n_high else np.full(n_runs, n_low)
    count = np.clip(count, n_low, n_high)
    shape = (n_runs, n_high)
    mask = np.arange(n_high)[None, :] < count[:, None]
    rows = table.set_index('Key')
    x: Dict[str, np.ndarray] = {key: sample_range(rows.loc[key], rng, shape) for key in rows.index}
    faildraw = rng.random(shape)
    success = (faildraw < x['frac_success_prob']) & mask
    productive_frac = success & (x['xf_ft'] > 0) & (x['cf_mdft'] > 0)
    k, h, phi = x['k_md'], x['h_ft'], x['phi']
    A, ct, pr0 = x['area_ac'], x['ct_psi'], x['pr_psi']
    pwf = x['pwf_psi']
    rw = x['rw_ft']
    mu, bo = x['mu_cp'], x['bo']
    wc = x['wc0']
    re = np.sqrt(A * 43560. / np.pi)
    domain_side = np.sqrt(A * 43560.)
    dy = domain_side / grid_size
    clean_den = np.maximum(np.log(re / rw) - .75, 1.0)
    pre_den = clean_den + x['skin_pre']
    if np.any(pre_den <= .5):
        raise ValueError('Non-physical radial denominator (check well radius, drainage area and pre-frac skin)')
    if q0_mode == 'Mechanistic radial Darcy PI':
        q0 = .00708 * k * h * np.maximum(pr0 - pwf, 0) / (mu * bo * pre_den) * (1. - wc)
    else:
        q0 = x['q0_bopd']
    # Baseline measured q0 anchors oil PI; draws in k/viscosity affect frac gain,
    # tank storage, or mechanistic q0 only. Both cases share identical draws.
    support = x['pressure_support']
    pv_rb = 7758. * A * h * phi
    storage = np.maximum(pv_rb * ct, 1e-6)  # reservoir bbl per psi
    base_pressure = pr0.copy()
    frac_pressure = pr0.copy()
    lookup = GridLookup(n=grid_size)
    M = 12 * years
    pre_rate = np.zeros((n_runs, M))
    frac_rate = np.zeros((n_runs, M))
    base_cum = np.zeros((n_runs, M))
    frac_cum = np.zeros((n_runs, M))
    cb = np.zeros((n_runs, n_high))
    cf = np.zeros((n_runs, n_high))
    init_dp = np.maximum(pr0 - pwf, 1e-5)
    xfrac_fraction = x['xf_ft'] / domain_side
    length_max = xfrac_fraction > .49
    if np.any(length_max & mask):
        # Geometry outside grid domain is truncated, NOT an infinite fracture.
        xfrac_fraction = np.minimum(xfrac_fraction, .49)
    init_link = x['cf_mdft'] / np.maximum(k * dy, 1e-7) * x['frac_height_frac']
    frac_cut0 = x['wc0']
    for m in range(M):
        years_mid = (m + .5) / 12.
        wc_base = np.clip(wc + x['wc_growth_ppyr'] * years_mid, 0, .975)
        delay = x['frac_water_delay_m']
        extra_ramp = np.clip(((m + .5) - delay) / 12., 0., 1.)
        wc_frac = np.clip(wc_base + x['frac_extra_wc_pp'] * extra_ramp * productive_frac, 0., .975)
        b_oilcut = (1. - wc_base) / np.maximum(1. - frac_cut0, .025)
        f_oilcut = (1. - wc_frac) / np.maximum(1. - frac_cut0, .025)
        retain = np.power(x['retention_5y'], (m + .5) / 60.)
        geo_gain = lookup.gain(xfrac_fraction, init_link * retain)
        # Normalize geometry to equivalent radial clean-well resistance then
        # incorporate measured skin / explicitly residual post-frac skin.
        post_den = np.maximum(clean_den / geo_gain + x['skin_post_residual'], .5)
        frac_gain = np.where(productive_frac, pre_den / post_den, 1.)
        q_base = q0 * np.maximum(base_pressure - pwf, 0.) / init_dp * b_oilcut * x['uptime']
        q_frac = q0 * np.maximum(frac_pressure - pwf, 0.) / init_dp * f_oilcut * x['uptime'] * frac_gain
        # Lost first days in frac case including frac failures. Start all frac jobs at t0.
        month_start = m * DAYS_PER_MONTH
        month_end = (m + 1) * DAYS_PER_MONTH
        fraction_online = np.clip((month_end - np.maximum(month_start, x['downtime_days'])) / DAYS_PER_MONTH, 0., 1.)
        q_frac *= fraction_online
        q_base *= mask
        q_frac *= mask
        pre_rate[:, m] = np.sum(q_base, axis=1)
        frac_rate[:, m] = np.sum(q_frac, axis=1)
        cb += q_base * DAYS_PER_MONTH
        cf += q_frac * DAYS_PER_MONTH
        base_cum[:, m] = np.sum(cb, axis=1)
        frac_cum[:, m] = np.sum(cf, axis=1)
        base_pressure = np.maximum(pwf, base_pressure - q_base * DAYS_PER_MONTH * bo / storage * (1. - support))
        frac_pressure = np.maximum(pwf, frac_pressure - q_frac * DAYS_PER_MONTH * bo / storage * (1. - support))
    diag = pd.DataFrame({
        'Run': np.arange(1, n_runs + 1),
        'Well count': count,
        'Technical successes': np.sum(success, axis=1),
        'Baseline cum oil stb': base_cum[:, -1],
        'Frac cum oil stb': frac_cum[:, -1],
        'Incremental oil stb': frac_cum[:, -1] - base_cum[:, -1],
        'Base end oil rate bopd': pre_rate[:, -1],
        'Frac end oil rate bopd': frac_rate[:, -1],
    })
    # Mean inputs per realization for meaningful (descriptive) Spearman tornado.
    for key in ('k_md', 'xf_ft', 'cf_mdft', 'skin_pre', 'area_ac', 'pr_psi', 'q0_bopd',
                'frac_success_prob', 'wc0', 'retention_5y', 'pressure_support'):
        diag['Avg ' + key] = np.sum(x[key] * mask, axis=1) / count
    return Result(months=np.arange(1, M + 1), base_rate=pre_rate, frac_rate=frac_rate,
                  incremental_rate=frac_rate - pre_rate, base_cum=base_cum,
                  frac_cum=frac_cum, incremental_cum=frac_cum - base_cum,
                  well_count=count, success_count=np.sum(success, axis=1),
                  diagnostic=diag, lookup=lookup)
