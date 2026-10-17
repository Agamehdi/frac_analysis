"""Frac Hyperion v1 – GitHub/Streamlit Cloud-ready demonstration application.
Copyright (c) 2026. Open source MIT. NOT a commercial REVEAL replica.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from frac_physics import (GridLookup, default_parameter_table, run_mc, solve_grid,
                          summarize_curves, validate_parameters)

st.set_page_config(page_title='Frac Hyperion | Monte Carlo', page_icon='🛢️',
                   layout='wide', initial_sidebar_state='expanded')

CUSTOM = """
<style>
.block-container { padding-top:1.5rem; max-width:1450px; }
[data-testid="stMetricValue"] { font-size:1.8rem; }
.hero {padding:1.0rem 1.4rem; border-radius:14px; border:1px solid #3a5261;
       background:linear-gradient(100deg,#142c3d,#1a4358); color:white; margin-bottom:1rem; }
.hero h1 { margin:0 0 .25rem 0; font-size:2rem; }
.hero p {margin:0;color:#dce8ee;}
.note { color:#64748b; font-size:.92rem; }
</style>
"""
st.markdown(CUSTOM, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>FRAC HYPERION <small style="font-size:0.42em">V1</small></h1>'
            '<p>Probabilistic fracture productivity • Monte Carlo forecast • 2D grid flow model</p></div>',
            unsafe_allow_html=True)

if 'params' not in st.session_state:
    st.session_state['params'] = default_parameter_table()
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

PAGES = ['Monte Carlo Studio', '2D Fracture Grid', '3D Fracture View', 'Guidelines & SPE References']
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select workspace', PAGES, label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.markdown('**Model scope**')
st.sidebar.caption('Single-phase, isotropic matrix; one finite-conductivity bi-wing fracture per vertical well. 2D grid-derived productivity gain + lumped pressure depletion.')
st.sidebar.warning('SCREENING ONLY — not REVEAL, not a transient multiphase reservoir simulator, and not a reserves booking tool.')


def mode(key):
    return float(st.session_state['params'].set_index('Key').loc[key, 'Most likely'])


def render_probability_cloud(res, kind, title, y_title, max_lines=70):
    t = res.months / 12.
    cases = {
        'incremental_cum': (res.incremental_cum, None),
        'incremental_rate': (res.incremental_rate, None),
        'frac_rate': (res.frac_rate, res.base_rate),
        'frac_cum': (res.frac_cum, res.base_cum),
    }
    values, baseline = cases[kind]
    p90, p50, p10 = np.percentile(values, (10, 50, 90), axis=0)
    fig = go.Figure()
    for i in np.linspace(0, len(values) - 1, min(max_lines, len(values)), dtype=int):
        fig.add_trace(go.Scatter(x=t, y=values[i], mode='lines',
                                 line=dict(color='rgba(103,139,167,0.12)', width=1),
                                 hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=p90, name='P90 | downside', mode='lines',
                             line=dict(color='#df8e46', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=p10, name='P10 | upside', mode='lines',
                             fill='tonexty', fillcolor='rgba(43,150,172,0.13)',
                             line=dict(color='#34b2cd', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=p50, name='P50 | median', mode='lines',
                             line=dict(color='#1c90ca', width=3)))
    if baseline is not None:
        fig.add_trace(go.Scatter(x=t, y=np.median(baseline, axis=0),
                                 name='No frac | median', mode='lines',
                                 line=dict(color='#9a9a9a', width=2.5, dash='dash')))
    fig.update_layout(title=title, xaxis_title='Years from frac date',
                      yaxis_title=y_title, hovermode='x unified', template='plotly_white',
                      height=450, margin=dict(l=15,r=10,b=20,t=55), legend=dict(orientation='h',y=-.22))
    return fig


def display_table_editor():
    st.subheader('Uncertainty parameter ranges')
    st.caption('Edit Low / Most likely / High and the distribution for EVERY input. '
               'Numbers displayed below are fictional starting assumptions — replace with reservoir data.')
    df = st.session_state['params']
    edited = st.data_editor(
        df, hide_index=True, num_rows='fixed', use_container_width=True, height=570,
        disabled=['Group', 'Key', 'Parameter', 'Unit', 'Guidance'],
        column_config={
            'Group': st.column_config.TextColumn('Section', width='small'),
            'Parameter': st.column_config.TextColumn('Parameter', width='medium'),
            'Key': st.column_config.TextColumn('Code', width='small'),
            'Unit': st.column_config.TextColumn('Unit', width='small'),
            'Low': st.column_config.NumberColumn('Low', format='%.6g', required=True),
            'Most likely': st.column_config.NumberColumn('Most likely', format='%.6g', required=True),
            'High': st.column_config.NumberColumn('High', format='%.6g', required=True),
            'Distribution': st.column_config.SelectboxColumn('Distribution', options=['Triangular','Log-triangular','Uniform','Fixed'], required=True),
            'Guidance': st.column_config.TextColumn('Meaning', width='large'),
        }, key='parameter_grid')
    st.session_state['params'] = edited.copy()
    errs = validate_parameters(edited)
    if errs:
        st.error('Invalid inputs: ' + ' | '.join(errs[:6]))
    c1,c2 = st.columns(2)
    with c1:
        st.download_button('⬇ Export current ranges (CSV)', edited.to_csv(index=False).encode('utf-8'),
                           'frac_uncertainty_ranges.csv','text/csv',use_container_width=True)
    with c2:
        upload = st.file_uploader('Import ranges CSV', type=['csv'], key='upload_ranges',
                                  help='Use a CSV exported from this application, preserving all parameter keys.')
        if upload is not None:
            try:
                imported = pd.read_csv(upload)
                if (set(imported['Key']) != set(df['Key'])) or set(df.columns) != set(imported.columns):
                    st.error('Import must have exact keys and columns as exported template.')
                elif validate_parameters(imported):
                    st.error('Imported CSV has invalid ranges: ' + '; '.join(validate_parameters(imported)[:5]))
                elif st.button('Apply imported ranges'):
                    st.session_state['params'] = imported[df.columns].copy()
                    st.session_state['last_result'] = None
                    st.rerun()
            except Exception as exc:
                st.error(f'Could not read CSV: {exc}')
    if st.button('Reset all range defaults'):
        st.session_state['params'] = default_parameter_table()
        st.session_state['last_result'] = None
        st.rerun()
    return errs


def result_archive(res) -> bytes:
    curves = summarize_curves(res)
    monthly = {'Run':np.repeat(np.arange(1, len(res.well_count) + 1), len(res.months)),
               'Month':np.tile(res.months, len(res.well_count))}
    for name, ar in [('Baseline rate bopd', res.base_rate), ('Post frac rate bopd', res.frac_rate),
                     ('Incremental rate bopd',res.incremental_rate),
                     ('Baseline cum stb',res.base_cum), ('Post frac cum stb',res.frac_cum),
                     ('Incremental cum stb',res.incremental_cum)]:
        monthly[name] = ar.ravel()
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zz:
        zz.writestr('monthly_percentiles.csv', curves.to_csv(index=False))
        zz.writestr('run_level_results.csv', res.diagnostic.to_csv(index=False))
        zz.writestr('all_realizations_monthly.csv', pd.DataFrame(monthly).to_csv(index=False))
        zz.writestr('input_ranges.csv', st.session_state['params'].to_csv(index=False))
    return bio.getvalue()


if page == 'Monte Carlo Studio':
    st.markdown('### Monte Carlo Forecast Engine')
    a, b, c, d, e = st.columns([1,1,1,1,1])
    with a:
        nlo = int(st.number_input('Wells — Low', min_value=1, max_value=250, value=5, step=1))
    with b:
        nmid = int(st.number_input('Wells — Most likely', min_value=1, max_value=250, value=10, step=1))
    with c:
        nhi = int(st.number_input('Wells — High', min_value=1, max_value=250, value=20, step=1))
    with d:
        simulations = int(st.select_slider('Monte Carlo runs', options=[100,250,500,1000,2000,3000], value=1000))
    with e:
        years = int(st.number_input('Forecast years', min_value=1, max_value=10, value=5, step=1))
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        initial_rate = st.radio('Pre-frac baseline initialization',
                                ['Measured rate anchor','Mechanistic radial Darcy PI'],
                                help='Measured anchor keeps supplied q0; mechanistic computes baseline PI from k,h,μ,B,skin,pressure.',
                                horizontal=True)
    with c2:
        grid_size = st.select_slider('Grid lookup resolution (odd cells per side)',
                                     options=[15,19,21,25,29], value=21,
                                     help='Higher resolution = more faithful geometry, slightly longer initialization')
    with c3:
        seed = int(st.number_input('Random seed',min_value=0,max_value=10**8,value=42))
    st.info('**How well counts work:** sample a total well count for each realization from Low/Most likely/High. '
            'Then sample independent reservoir/fracture ranges for each selected well. '
            'All wells start frac work at forecast month 0; each has its own drainage volume. '
            'No interference, scheduling constraints, or waterflood cross-flow are simulated.')
    errors = display_table_editor()
    st.divider()
    if not (nlo <= nmid <= nhi):
        st.error('Well counts must satisfy Low <= Most likely <= High.')
    can_run = not errors and (nlo <= nmid <= nhi) and simulations * nhi <= 300000
    if simulations * nhi > 300000:
        st.error('Runs × maximum wells must be <= 300,000 for Streamlit Cloud memory safety.')
    if st.button('▶ Run Monte Carlo Forecast', type='primary', use_container_width=True, disabled=not can_run):
        try:
            with st.spinner('Computing 2D grid response lookup, paired depletion cases and percentile clouds...'):
                st.session_state['last_result'] = run_mc(st.session_state['params'], nlo,nmid,nhi,
                    n_runs=simulations, years=years, seed=seed, q0_mode=initial_rate,
                    grid_size=int(grid_size))
            st.success('Monte Carlo completed — scroll down for results.')
        except Exception as exc:
            st.session_state['last_result'] = None
            st.error(f'Cannot run: {exc}')
    res = st.session_state['last_result']
    if res is not None:
        st.markdown('## Probabilistic results')
        final = res.incremental_cum[:, -1]
        p90, p50, p10 = np.percentile(final, [10, 50, 90])
        z1,z2,z3,z4,z5 = st.columns(5)
        z1.metric('Incremental P90',f'{p90/1000:,.0f} kstb')
        z2.metric('Incremental P50',f'{p50/1000:,.0f} kstb')
        z3.metric('Incremental P10',f'{p10/1000:,.0f} kstb')
        z4.metric('Mean frac wells', f'{np.mean(res.well_count):.1f}')
        z5.metric('Probability > 0 incremental', f'{np.mean(final>0)*100:.1f}%')
        st.caption('P90 is the 10th percentile (downside); P10 is the 90th percentile (upside). '
                   'Cumulative increment = paired frac cumulative oil minus paired baseline cumulative oil.')
        t1,t2,t3,t4 = st.tabs(['Cumulative increment cloud', 'Oil-rate clouds','Probability / S-curve','Sensitivity'])
        with t1:
            st.plotly_chart(render_probability_cloud(res,'incremental_cum',
                'Incremental cumulative oil — 1,000-path style cloud', 'Incremental oil (stb)'),
                use_container_width=True)
            st.plotly_chart(render_probability_cloud(res,'frac_cum',
                'Total cumulative oil: with vs without frac', 'Cumulative oil (stb)'),
                use_container_width=True)
        with t2:
            st.plotly_chart(render_probability_cloud(res,'frac_rate',
                'Post-frac oil rate compared with untreated baseline','Oil rate (bopd)'),
                use_container_width=True)
            st.plotly_chart(render_probability_cloud(res,'incremental_rate',
                'Incremental oil-rate probability cloud', 'Incremental rate (bopd)'),
                use_container_width=True)
        with t3:
            cplot, ctrg = st.columns([3,1])
            with ctrg:
                goal_kstb = st.number_input('Incremental target (kstb)', min_value=0., value=300.,step=25.)
                st.metric('Probability of exceeding target',
                          f'{np.mean(final >= goal_kstb*1000)*100:.1f}%')
                st.metric('Prob. of frac technical success per well',
                          f'{np.sum(res.success_count)/np.sum(res.well_count)*100:.1f}%')
            with cplot:
                xvals = np.sort(final / 1000.)
                exceedance = 100 * (1 - np.arange(1, len(xvals)+1)/len(xvals))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xvals,y=exceedance,mode='lines',line=dict(color='#1889ba',width=3),name='Exceedance'))
                fig.add_vline(x=goal_kstb,line_dash='dash',line_color='#dc7935')
                fig.update_layout(title='S-curve: probability of exceeding incremental oil',xaxis_title='Incremental oil after forecast (kstb)',
                                  yaxis_title='Exceedance probability (%)', template='plotly_white',height=420)
                st.plotly_chart(fig,use_container_width=True)
            st.plotly_chart(go.Figure(go.Histogram(x=final/1000.,nbinsx=50,marker_color='#4396bb')).update_layout(
                title='Final incremental oil distribution',xaxis_title='Incremental oil (kstb)',
                yaxis_title='MC realizations',template='plotly_white'), use_container_width=True)
        with t4:
            cand = ['Well count','Technical successes'] + [col for col in res.diagnostic if col.startswith('Avg ')]
            sensitivity = res.diagnostic[cand+['Incremental oil stb']].corr(method='spearman')['Incremental oil stb'].drop('Incremental oil stb').dropna()
            sensitivity = sensitivity.loc[sensitivity.abs().sort_values(ascending=False).index].head(13)
            fig = go.Figure(go.Bar(x=sensitivity.values[::-1],y=sensitivity.index[::-1],orientation='h',
                                   marker_color=['#cc7d55' if v<0 else '#2499b9' for v in sensitivity.values[::-1]]))
            fig.update_layout(template='plotly_white',height=480,title='Spearman rank correlation (descriptive, not causal)',
                              xaxis_title='Correlation with final incremental oil (−1 to +1)',margin=dict(l=25,r=15))
            st.plotly_chart(fig,use_container_width=True)
            st.caption('Counts and success totals strongly drive field results. Correlation is not Sobol sensitivity or causal attribution.')
        forecast = summarize_curves(res)
        st.subheader('Month-by-month P90 / P50 / P10')
        st.dataframe(forecast, use_container_width=True, hide_index=True, height=275)
        horizon_rows = []
        for yy in [1,3,5,10]:
            if yy <= years:
                col = res.incremental_cum[:, 12*yy-1]
                horizon_rows.append({'Year':yy, 'P90 (kstb)':np.percentile(col,10)/1000,
                                     'P50 (kstb)':np.percentile(col,50)/1000,
                                     'P10 (kstb)':np.percentile(col,90)/1000,
                                     'Mean (kstb)':np.mean(col)/1000})
        st.dataframe(pd.DataFrame(horizon_rows),hide_index=True,use_container_width=True)
        st.download_button('⬇ Export complete MC results + all monthly paths (ZIP)', result_archive(res),
                           file_name='frac_hyperion_mc_results.zip',mime='application/zip',use_container_width=True)
        st.warning('Forecast assumes one independent compressible tank per fractured well. '
                   'Pressure-support fraction and empirical water-cut penalty are scenario assumptions. '
                   'For waterflood interaction, multiphase flow, below-bubble-point flow, fracture propagation '
                   'or reserve booking, validate in a full reservoir simulator.')
    else:
        st.caption('Adjust the ranges and click **Run Monte Carlo Forecast** to generate actual results.')

elif page == '2D Fracture Grid':
    st.markdown('## 2D Finite-Conductivity Fracture — Numerical Grid Explorer')
    st.write('A reservoir-like Cartesian grid with fixed outer pressure and a central producing well. '
             'An explicit horizontal fracture increases Darcy transmissibility only along grid connections it intersects. '
             'The 2D pressure map is *calculated*, not drawn arbitrarily.')
    aa,bb,cc = st.columns(3)
    k = aa.number_input('Matrix permeability k (mD)',min_value=.01,value=max(.01,mode('k_md')))
    A = bb.number_input('Drainage area per well (acres)',min_value=1.,value=max(1.,mode('area_ac')))
    hf = cc.number_input('Net pay thickness (ft)',min_value=1.,value=max(1.,mode('h_ft')))
    dd,ee,ff = st.columns(3)
    xf = dd.number_input('Fracture half-length (ft)',min_value=0.,value=max(0.,mode('xf_ft')))
    Cf = ee.number_input('Fracture conductivity (mD.ft)',min_value=0.,value=max(0.,mode('cf_mdft')))
    coverage = ff.slider('Fracture connected pay fraction',min_value=.1,max_value=1.,value=float(np.clip(mode('frac_height_frac'),.1,1)),step=.05)
    ni = st.select_slider('2D grid: cells per side',options=[15,21,31,41,51],value=41)
    side = np.sqrt(A*43560.)
    dy = side/ni
    lf = min(xf/side,.49)
    factor = (Cf/(k*dy))*coverage
    if xf/side > .49:
        st.warning('Fracture exceeds grid boundary; plotted length is truncated to 0.49 × reservoir side.')
    qb, before = solve_grid(ni,0,0,return_map=True)
    qf, after = solve_grid(ni,lf,factor,return_map=True)
    j1,j2,j3,j4 = st.columns(4)
    j1.metric('Grid PI multiplier',f'{qf/qb:.2f} ×')
    j2.metric('Dimensionless FCD',f'{Cf/(k*max(xf,.001)):.2f}')
    j3.metric('Grid cell size',f'{dy:.1f} ft')
    j4.metric('Enhanced link ratio',f'{factor:.1f}')
    xgrid = (np.arange(ni)-ni//2)*dy
    def heatmap(pressure, title, show_frac=False):
        fig=go.Figure(go.Heatmap(z=pressure,x=xgrid,y=xgrid,colorscale='Viridis',zmin=0,zmax=1,
                                 colorbar=dict(title='Normalized P')))
        if show_frac:
            fig.add_trace(go.Scatter(x=[-lf*side,lf*side],y=[0,0],mode='lines',
                                     line=dict(color='#ff6d46',width=7),name='Fracture'))
        fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers',marker=dict(size=11,symbol='x',color='white',line=dict(width=2,color='#101a23')),
                                 name='Well'))
        fig.update_layout(title=title,xaxis_title='x (ft)',yaxis_title='y (ft)',yaxis_scaleanchor='x',
                          template='plotly_white',height=450,margin=dict(l=5,r=5,b=25,t=50))
        return fig
    left,right=st.columns(2)
    with left: st.plotly_chart(heatmap(before,'Without fracture | Darcy pressure field'),use_container_width=True)
    with right: st.plotly_chart(heatmap(after,'With finite-conductivity fracture | pressure field',True),use_container_width=True)
    st.info('**Grid mechanics:** matrix x/y transmissibility is normalized to 1. '
            'An intersected horizontal face receives extra conductance '
            '`(kf × wf)/(k × Δy) × (hf/h)` adjusted by the fraction of that face intersected. '
            'Sparse finite-volume linear equations solve the pressure on every cell. '
            'Rate into fixed-pressure well grid cell yields the fracture/no-frac PI ratio.')
    st.caption('Caution: fixed-pressure outer boundary for stabilized productivity, not time-dependent pressure diffusion; '
               'nearwell geometry represented by well cell rather than Peaceman WI; interpolation and coarse-grid sensitivity apply. '
               'Compare multiple grid sizes before accepting PI multipliers.')

elif page == '3D Fracture View':
    st.markdown('## Conceptual 3D Grid + Planar Fracture')
    a,b,c = st.columns(3)
    side = float(a.number_input('Reservoir square side (ft)', value=1500.,min_value=200.))
    h = float(b.number_input('Pay thickness (ft)',value=80.,min_value=10.))
    xf = float(c.number_input('Frac half-length (ft)',value=300.,min_value=1.))
    contact = st.slider('Fracture height / pay height',.1,1.,.85,.05)
    xf = min(xf,.49*side)
    ztop=h/2; zbase=-h/2; hf=h*contact
    fig = go.Figure()
    for zz in [zbase,ztop]:
        for xx in np.linspace(-side/2,side/2,9):
            fig.add_trace(go.Scatter3d(x=[xx,xx],y=[-side/2,side/2],z=[zz,zz],
                mode='lines',line=dict(color='rgba(124,156,181,0.25)',width=1),showlegend=False,hoverinfo='skip'))
        for yy in np.linspace(-side/2,side/2,9):
            fig.add_trace(go.Scatter3d(x=[-side/2,side/2],y=[yy,yy],z=[zz,zz],
                mode='lines',line=dict(color='rgba(124,156,181,0.25)',width=1),showlegend=False,hoverinfo='skip'))
    fracx=[-xf,xf,xf,-xf]
    fracy=[0,0,0,0]
    fracz=[-hf/2,-hf/2,hf/2,hf/2]
    fig.add_trace(go.Mesh3d(x=fracx,y=fracy,z=fracz,i=[0,0],j=[1,2],k=[2,3],
                             color='#ec774b',opacity=.55,name='Planar fracture'))
    fig.add_trace(go.Scatter3d(x=[-xf,xf,xf,-xf,-xf],y=[0]*5,
                               z=[-hf/2,-hf/2,hf/2,hf/2,-hf/2],
                               mode='lines',line=dict(color='#e75a36',width=7),name='Frac outline'))
    fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[zbase-15,ztop+15],
                               mode='lines',line=dict(color='#ffe386',width=10),name='Wellbore'))
    fig.update_layout(scene=dict(xaxis_title='X (ft)',yaxis_title='Y (ft)',zaxis_title='Z (ft)',
                                 aspectmode='manual',aspectratio=dict(x=1.5,y=1.5,z=.55)),
                      height=590,margin=dict(l=0,r=0,b=0,t=15),legend=dict(orientation='h'))
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Conceptual geometry only; the flow calculation is the *2D* horizontal grid on the other page. '
               'This is not a 3D propagating fracture mesh and does not solve fracture mechanics.')

else:
    st.markdown('## User Guidelines & Technical Basis')
    st.markdown('''
### 1. Workflow
1. **Monte Carlo Studio:** enter a Low / Most likely / High value and distribution for every available rock, fluid, pressure, well and fracture parameter.
2. **Well count:** set Low / Most likely / High for **the number of fractured wells**. Each Monte Carlo case independently selects a count; well properties are resampled for each well.
3. **Run 1,000 cases:** each case shares input uncertainty across *paired* baseline and fracture forecasts, so incremental oil is calculated as the difference of two matched cases.
4. **Review:** compare P90/P50/P10 oil rate and incremental cumulative oil through time, S-curves, success probability and sensitivity.
5. **Validate:** check a representative fracture on **2D Fracture Grid**, compare grid resolutions, and export full run-level and monthly CSV files.

### 2. Classical fractured-well analysis
For matrix flow, the starting point is the stabilized Darcy radial productivity equation (field units):

$$q_o=\\frac{0.00708\\, k h (\\bar p-p_{wf})}{\\mu_o B_o [\\ln(r_e/r_w)-0.75+s]}$$

The classical finite-conductivity fracture parameter is:

$$F_{CD}=\\frac{k_f w_f}{k x_f}=\\frac{C_f}{k x_f}$$

A large half-length or high conductivity can lower effective resistance, but conductivity is *finite*: a low-conductivity fracture does not behave as an infinitely conducting line source. This app **does not implement exact Cinco-Ley transient type curves**.

### 3. What the 2D grid actually solves
For every interior cell, the steady, single-phase Darcy finite-volume equation is:

$$\\sum_{j} T_{ij}(p_i-p_j)=0$$

For square cells, matrix transmissibility in the x/y directions is proportional to $k h$. If a horizontal cell-to-cell connection is intersected by the fracture, the x-direction transmissibility additionally gets:

$$\\Delta T_x \\propto \\frac{C_f h_f}{\\Delta x}$$

The relative additional conductivity for a fully intersected edge is:

$$\\frac{\\Delta T_x}{T_{x,matrix}}=\\frac{C_f}{k\\Delta y}\\frac{h_f}{h}$$

The app solves the resulting sparse pressure grid twice (fractured/unfractured), then calculates the ratio of cell-interface well inflows. The Monte Carlo engine uses an interpolated table of these **numerically computed grid ratios** as a fast productivity response. This is not a full transient EDFM or REVEAL reproduction.

The ratio is combined with a radial denominator to account for measured pre-frac skin and specified residual post-frac skin. This combination is a **screening approximation**; calibrate against PTA and treatment diagnostics.

### 4. Forecast and uncertainty
- Initial rate: either measured oil rate (history-match anchor) or the radial Darcy PI equation.
- Frac conductivity declines continuously to the sampled five-year retained fraction.
- For each well, **successful frac** is a Bernoulli sample; if it fails there is no productivity uplift but treatment downtime still applies.
- Every well has a separate compressible pore-volume tank with effective storage $PV\\,c_t$; produced oil causes depletion reduced by an empirical pressure-support factor.
- Water cut is an **empirical oil-cut multiplier**, not Buckley–Leverett or oil/water relative-permeability physics.
- Water-cut penalties may make incremental oil negative. Additional recoverable reserves are **not** guaranteed by early incremental cumulative oil.
- For fractional scenarios, P90 is the 10th percentile (conservative), P50 the median, and P10 the 90th percentile (upside), including any failed treatments.

### 5. Inputs and QA checklist
- **PTA:** permeability, skin, reservoir pressure, boundary response.
- **PVT:** viscosity and formation volume factor, and check whether pressure crosses bubble point.
- **Petrophysics:** thickness, porosity, saturation, water contact / gas contact proximity.
- **Frac design:** fracture half-length, height, conductivity, closure stress, stress contrast, proppant/acid performance.
- **Production:** baseline rate, pressures, water cut, artificial-lift/facility constraints, uptime.
- **Critical validation:** compare grids, investigate water injection connectivity and water breakthrough, use well-by-well history matching and economic cutoffs.

### 6. Scope / excluded physics
This is a **v1 engineering screening tool for a single vertical well with one planar bi-wing fracture**. It is not a geomechanical fracture propagation simulator, hydraulic fracture treatment simulator, DFN/EDFM full 3D reservoir simulator, or REVEAL source-code implementation. It does not simulate full three-phase flow, pressure-dependent PVT, injector-producer interference, fracture-fluid cleanup, water coning, fracturing-fluid chemistry, or multi-stage horizontal wells. Physical inputs should be checked before commercial use.

### 7. Verified research references
1. **Prats, M. (1961).** *Effect of Vertical Fractures on Reservoir Behavior—Incompressible-Fluid Case*. SPE Journal, SPE-1575-G. [DOI: 10.2118/1575-G](https://doi.org/10.2118/1575-G). Classic finite-capacity fracture / equivalent radius work.
2. **Cinco-Ley, H. and Samaniego-V., F. (1981).** *Transient Pressure Analysis for Fractured Wells*. Journal of Petroleum Technology, **SPE-7490-PA**. [DOI: 10.2118/7490-PA](https://doi.org/10.2118/7490-PA). Finite-conductivity fractured well transient analysis.
3. **SPE-88960-PA (2006).** *Stabilized Productivity of a Hydraulically Fractured Well Producing at Constant Pressure*. SPE Journal. [DOI: 10.2118/88960-PA](https://doi.org/10.2118/88960-PA). Simple numerical approach linking length/conductivity and productivity.
4. **2020 study.** *Productivity Simulation of Hydraulically Fractured Wells Based on Hybrid Local Grid Refinement and Embedded Discrete Fracture Model*. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1876380420600532). More advanced grid refinement and fracture modeling.
5. **Petroleum Experts (2018).** [REVEAL feature / enhancement publication](https://www.petex.com/media/2452/feb2018_ipm11_ivm8_ifm5_enhancements.pdf). Referenced for high-level context; **this app is not endorsed by or based on REVEAL's proprietary equations**.
''')
    st.info('Recommended next development: well-by-well catalogue with shared correlated uncertainty, '
            'measured history match, numerical transient grid simulation and explicit multiphase relative permeability.')

st.sidebar.divider()
st.sidebar.caption('Frac Hyperion • V1.0 • GitHub-ready • October 2026')
