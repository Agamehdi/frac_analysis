"""KAPPA benchmark matching regression tests (no Streamlit UI required)."""
import numpy as np
import pandas as pd
import pytest

from frac_physics import default_parameter_table, run_mc
from kappa_benchmark import (DEFAULT_BENCHMARK, GLOBAL, INCREMENTAL, LOCAL, POST,
                             calibration_factor, check_benchmark, fit_benchmark,
                             performance, reference_rates, reference_signature)


def set_fixed(table, overrides=None):
    df = table.copy()
    overrides = overrides or {}
    for idx, row in df.iterrows():
        value = float(overrides.get(row['Key'], row['Most likely']))
        for col in ['Low', 'Most likely', 'High']:
            df.at[idx, col] = value
        df.at[idx, 'Distribution'] = 'Fixed'
    return df


@pytest.fixture(scope='module')
def kappa_setup():
    params = default_parameter_table()
    benchmark = check_benchmark(DEFAULT_BENCHMARK)
    raw = reference_rates(params, benchmark['Perm (mD)'].to_numpy(), 'Mechanistic radial Darcy PI')
    signature = reference_signature(params, 'Mechanistic radial Darcy PI', 21)
    return params, benchmark, raw, signature


def test_all_four_anchor_points_match_for_incremental_and_post(kappa_setup):
    _, bench, raw, sig = kappa_setup
    for definition in [INCREMENTAL, POST]:
        cal, comparison = fit_benchmark(bench, raw, definition, LOCAL, sig)
        assert cal['definition'] == definition
        assert np.allclose(comparison['Matched (stb/d)'], [70,380,700,720], atol=1e-9)
        assert performance(comparison,'Matched (stb/d)')['MAPE (%)'] < 1e-10


def test_global_multiplier_and_inactive_extrapolation(kappa_setup):
    _, bench, raw, sig = kappa_setup
    cal, comp = fit_benchmark(bench, raw, INCREMENTAL, GLOBAL, sig)
    assert len(comp)==4
    f = calibration_factor(np.array([0.01,1,10,100,1000]), cal)
    assert f[0] == 1 and f[-1] == 1
    assert np.all(np.isfinite(f)) and np.all(f > 0)


def test_invalid_benchmark_rejected():
    with pytest.raises(ValueError,match='Duplicate'):
        check_benchmark(pd.DataFrame({'Perm (mD)':[10,10], 'KAPPA Oil Rate (stb/d)':[20,30]}))
    with pytest.raises(ValueError,match='greater than zero'):
        check_benchmark(pd.DataFrame({'Perm (mD)':[1,10], 'KAPPA Oil Rate (stb/d)':[0,30]}))


@pytest.mark.parametrize('definition', [INCREMENTAL, POST])
def test_calibrated_mc_one_zone_anchor(definition):
    params = set_fixed(default_parameter_table(), {
        'k_md': 10, 'uptime':1, 'downtime_days':0,
        'wc_growth_ppyr':0, 'frac_extra_wc_pp':0,
        'retention_5y':1, 'frac_success_prob':1,
        'pressure_support':1,
    })
    bench = check_benchmark(DEFAULT_BENCHMARK)
    raw = reference_rates(params, bench['Perm (mD)'].values, 'Mechanistic radial Darcy PI')
    sig = reference_signature(params, 'Mechanistic radial Darcy PI', 21)
    cal,_ = fit_benchmark(bench,raw,definition,LOCAL,sig)
    model = run_mc(params,1,1,1,n_runs=10,years=1,seed=12,
                   q0_mode='Mechanistic radial Darcy PI',calibration=cal)
    # One zone/well, 10 mD. At forecast month 1 before depletion, no WC/downtime impact.
    result = model.frac_rate[:,0] if definition == POST else model.incremental_rate[:,0]
    assert np.allclose(result,380,rtol=1e-9)
    assert model.calibration_method == LOCAL


def test_unmatched_mc_unchanged_and_stale_match_rejected(kappa_setup):
    params, bench, raw, sig = kappa_setup
    no_fit = run_mc(params,1,1,1,n_runs=10,years=1,seed=123)
    assert no_fit.calibration_method is None
    cal,_ = fit_benchmark(bench,raw,INCREMENTAL,LOCAL,sig)
    modified=params.copy()
    modified.loc[modified['Key']=='xf_ft','Most likely'] = 240.
    with pytest.raises(ValueError, match='stale'):
        run_mc(modified,1,1,1,n_runs=10,years=1,
               q0_mode='Mechanistic radial Darcy PI',calibration=cal)
