import numpy as np
import pytest
from frac_physics import (GridLookup, default_parameter_table, run_mc, solve_grid,
                          summarize_curves, validate_parameters)


def test_grid_pressure_and_gain():
    q_base, p0 = solve_grid(21,0,0,True)
    q_frac, pf = solve_grid(21,.25,400.,True)
    assert q_base > 0
    assert q_frac > q_base
    assert np.min(pf) >= -1e-9
    assert np.max(pf) <= 1+1e-9
    assert np.isclose(pf[10,10],0)
    assert np.allclose(pf[0,:],1)


def test_zero_fracture_response():
    base=solve_grid(19,0,0)
    assert np.isclose(base,solve_grid(19,0,100))
    assert np.isclose(base,solve_grid(19,.3,0))


def test_bigger_fracture_more_gain():
    short=solve_grid(21,.08,300)
    long=solve_grid(21,.3,300)
    assert long > short
    assert solve_grid(21,.2,1000) > solve_grid(21,.2,1)


def test_lookup_matches_exact_on_nodes():
    lk=GridLookup(n=15,length_bins=9,conductivity_bins=9)
    fi=4; ci=5
    expected=solve_grid(15,lk.length_axis[fi],10**lk.log_link_axis[ci])/lk.q_base
    assert np.isclose(lk.gain(lk.length_axis[fi],10**lk.log_link_axis[ci]),expected)


def test_mc_has_consistent_shapes_and_cumulative():
    p=default_parameter_table()
    r=run_mc(p,5,10,15,100,years=3,grid_size=15)
    assert r.frac_rate.shape==(100,36)
    assert np.all((r.well_count>=5)&(r.well_count<=15))
    assert np.all(r.success_count<=r.well_count)
    assert np.allclose(r.incremental_cum,r.frac_cum-r.base_cum)
    assert np.all(np.diff(r.frac_cum,axis=1)>=-1e-8)
    assert len(summarize_curves(r))==36


def test_identical_seed_reproducibility():
    p=default_parameter_table()
    x=run_mc(p,3,4,5,30,years=1,seed=123,grid_size=15)
    y=run_mc(p,3,4,5,30,years=1,seed=123,grid_size=15)
    assert np.array_equal(x.incremental_cum,y.incremental_cum)


def test_all_fracs_fail_production_lost_for_downtime():
    p=default_parameter_table()
    p.loc[p['Key']=='frac_success_prob',['Low','Most likely','High']]=0.
    r=run_mc(p,4,4,4,60,years=1,grid_size=15)
    assert np.all(r.success_count==0)
    assert np.all(r.incremental_cum[:,-1]<0)  # frac wells lose first-month production


def test_no_frac_conductivity_no_gain_after_cleanup():
    p=default_parameter_table()
    p.loc[p['Key']=='cf_mdft',['Low','Most likely','High']]=0.
    p.loc[p['Key']=='cf_mdft','Distribution']='Fixed'
    p.loc[p['Key']=='skin_post_residual',['Low','Most likely','High']]=0.
    p.loc[p['Key']=='skin_pre',['Low','Most likely','High']]=0.
    p.loc[p['Key']=='frac_extra_wc_pp',['Low','Most likely','High']]=0.
    p.loc[p['Key']=='downtime_days',['Low','Most likely','High']]=0.
    r=run_mc(p,2,2,2,30,years=1,grid_size=15)
    assert np.allclose(r.frac_rate,r.base_rate)


def test_bad_range_caught():
    p=default_parameter_table()
    p.loc[p['Key']=='k_md','Low']=1000
    assert validate_parameters(p)


def test_probability_logic_per_realization():
    p=default_parameter_table()
    r=run_mc(p,2,4,7,200,years=2,grid_size=15)
    point=np.percentile(r.incremental_cum[:,-1],[10,50,90])
    assert point[0]<=point[1]<=point[2]
