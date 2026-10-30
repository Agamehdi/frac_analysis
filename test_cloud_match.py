"""Tests for full-ensemble recalculation and original/matched consistency."""
import numpy as np
from frac_physics import default_parameter_table, run_mc
from kappa_benchmark import (DEFAULT_BENCHMARK, INCREMENTAL, LOCAL,
                             check_benchmark, fit_benchmark,
                             reference_rates, reference_signature)


def calibration_for(params, mode='Measured rate anchor'):
    b = check_benchmark(DEFAULT_BENCHMARK)
    raw = reference_rates(params, b['Perm (mD)'].to_numpy(), mode, 21)
    cal, _ = fit_benchmark(b, raw, INCREMENTAL, LOCAL,
                           reference_signature(params, mode, 21))
    return cal


def test_matched_whole_cloud_is_paired_and_recomputed():
    params = default_parameter_table()
    cal = calibration_for(params)
    original = run_mc(params, 3, 5, 8, n_runs=64, years=2, seed=1234)
    matched = run_mc(params, 3, 5, 8, n_runs=64, years=2, seed=1234, calibration=cal)
    assert np.array_equal(original.well_count, matched.well_count)
    assert np.array_equal(original.success_count, matched.success_count)
    assert np.array_equal(original.well_active_mask, matched.well_active_mask)
    assert np.array_equal(original.base_rate, matched.base_rate)
    assert np.array_equal(original.base_cum, matched.base_cum)
    assert np.allclose(original.diagnostic['Avg k_md'], matched.diagnostic['Avg k_md'])
    assert matched.calibration_method == LOCAL
    assert matched.frac_rate.shape == original.frac_rate.shape == (64, 24)
    assert not np.array_equal(matched.frac_rate, original.frac_rate)
    assert not np.array_equal(matched.incremental_rate, original.incremental_rate)
    assert not np.array_equal(matched.incremental_cum, original.incremental_cum)
    assert np.allclose(matched.incremental_cum, matched.frac_cum - matched.base_cum)
    assert np.allclose(matched.incremental_rate, matched.frac_rate - matched.base_rate)
    assert not np.array_equal(matched.frac_rate[:, 12], original.frac_rate[:, 12])


def test_outside_benchmark_range_remains_uncalibrated():
    params = default_parameter_table()
    cal = calibration_for(params)
    params.loc[params['Key'] == 'k_md', ['Low', 'Most likely', 'High']] = 200.
    params.loc[params['Key'] == 'k_md', 'Distribution'] = 'Fixed'
    # Exact current modal signature with modified fixed permeability
    cal = calibration_for(params)
    orig = run_mc(params, 1, 1, 1, n_runs=10, years=1, seed=23)
    match = run_mc(params, 1, 1, 1, n_runs=10, years=1, seed=23, calibration=cal)
    assert np.array_equal(orig.frac_rate, match.frac_rate)
    assert np.array_equal(orig.incremental_cum, match.incremental_cum)


def test_failure_scenarios_are_unmodified_by_matching():
    params = default_parameter_table()
    params.loc[params['Key'] == 'frac_success_prob', ['Low','Most likely','High']] = 0.
    params.loc[params['Key'] == 'frac_success_prob', 'Distribution'] = 'Fixed'
    cal = calibration_for(params)
    orig = run_mc(params, 1, 3, 5, n_runs=16, years=1, seed=45)
    match = run_mc(params, 1, 3, 5, n_runs=16, years=1, seed=45, calibration=cal)
    assert np.array_equal(orig.frac_rate, match.frac_rate)
    assert np.array_equal(orig.incremental_cum, match.incremental_cum)
