"""Frac Hyperion v2 – GitHub/Streamlit Cloud-ready demonstration application."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from frac_physics import GridLookup, default_parameter_table, run_mc, solve_grid, summarize_curves, validate_parameters

st.set_page_config(page_title='Frac Hyperion | Monte Carlo', page_icon='🛢️', layout='wide', initial_sidebar_state='expanded')

CUSTOM = """
<style>
.block-container { padding-top:1.2rem; max-width:1500px; }
[data-testid="stMetricValue"] { font-size:1.7rem; }
.hero {padding:1.0rem 1.4rem; border-radius:14px; border:1px solid #3a5261;
       background:linear-gradient(100deg,#142c3d,#1a4358); color:white; margin-bottom:1rem; }
.hero h1 { margin:0 0 .25rem 0; font-size:2rem; }
.hero p {margin:0;color:#dce8ee;}
.note { color:#64748b; font-size:.92rem; }
.smallcaps {font-size:.85rem; letter-spacing:0.04em; color:#64748b; text-transform:uppercase;}
</style>
"""
st.markdown(CUSTOM, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>FRAC HYPERION <small style="font-size:0.42em">V2</small></h1>'
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
st.sidebar.caption('SCREENING ONLY — not REVEAL, not a transient multiphase simulator, and not a reserves booking tool.')


@dataclass(frozen=True)
class UnitPrefs:
    system: str
    rate_scale: str
    cum_scale: str


FIELD_TO_METRIC = {
    'h_ft': 0.3048,
    'rw_ft': 0.3048,
    'xf_ft': 0.3048,
    'area_ac': 0.40468564224,
    'pr_psi': 0.0689475729,
    'pwf_psi': 0.0689475729,
    'ct_psi': 14.5037738,
    'q0_bopd': 0.1589872949,
    'cf_mdft': 0.3048,
}
METRIC_UNITS = {
    'h_ft': 'm', 'rw_ft': 'm', 'xf_ft': 'm', 'area_ac': 'hectares',
    'pr_psi': 'bar', 'pwf_psi': 'bar', 'ct_psi': '1/bar', 'q0_bopd': 'm³/d', 'cf_mdft': 'mD·m'
}


def get_prefs() -> UnitPrefs:
    st.sidebar.divider()
    st.sidebar.markdown('**Display units**')
    system = st.sidebar.radio('Unit system', ['Field', 'Metric'], horizontal=True)
    if system == 'Field':
        rate_scale = st.sidebar.radio('Rate unit', ['bopd', 'mbd'], horizontal=True,
                                      help='mbd here means thousand barrels of oil per day.')
        cum_scale = st.sidebar.radio('Cumulative unit', ['stb', 'mmstb'], horizontal=True)
    else:
        rate_scale = st.sidebar.radio('Rate unit', ['m³/d', 'km³/d'], horizontal=True,
                                      help='km³/d here means thousand cubic metres per day.')
        cum_scale = st.sidebar.radio('Cumulative unit', ['m³', 'Mm³'], horizontal=True)
    return UnitPrefs(system, rate_scale, cum_scale)


PREFS = get_prefs()


def rate_factor_and_label(unit: str):
    if unit == 'bopd':
        return 1.0, 'bopd'
    if unit == 'mbd':
        return 1e-3, 'Mstb/d'
    if unit == 'm³/d':
        return 0.1589872949, 'm³/d'
    if unit == 'km³/d':
        return 0.1589872949e-3, '10³ m³/d'
    return 1.0, unit


def cum_factor_and_label(unit: str):
    if unit == 'stb':
        return 1.0, 'stb'
    if unit == 'mmstb':
        return 1e-6, 'MMstb'
    if unit == 'm³':
        return 0.1589872949, 'm³'
    if unit == 'Mm³':
        return 0.1589872949e-6, 'Mm³'
    return 1.0, unit


def display_table_for_editor(df_internal: pd.DataFrame, system: str) -> pd.DataFrame:
    df = df_internal.copy()
    units = []
    lows, mids, highs = [], [], []
    for _, row in df.iterrows():
        key = row['Key']
        factor = 1.0
        unit = row['Unit']
        if system == 'Metric' and key in FIELD_TO_METRIC:
            factor = FIELD_TO_METRIC[key]
            unit = METRIC_UNITS[key]
        units.append(unit)
        lows.append(float(row['Low']) * factor)
        mids.append(float(row['Most likely']) * factor)
        highs.append(float(row['High']) * factor)
    df['Unit'] = units
    df['Low'] = lows
    df['Most likely'] = mids
    df['High'] = highs
    return df


def editor_to_internal(df_display: pd.DataFrame, template_internal: pd.DataFrame, system: str) -> pd.DataFrame:
    df = df_display.copy()
    out = template_internal.copy()
    for i, row in df.iterrows():
        key = row['Key']
        factor = 1.0
        if system == 'Metric' and key in FIELD_TO_METRIC:
            factor = FIELD_TO_METRIC[key]
        out.at[i, 'Low'] = float(row['Low']) / factor
        out.at[i, 'Most likely'] = float(row['Most likely']) / factor
        out.at[i, 'High'] = float(row['High']) / factor
        out.at[i, 'Distribution'] = row['Distribution']
    return out


def mode(key):
    return float(st.session_state['params'].set_index('Key').loc[key, 'Most likely'])


def percentile_cloud(values, months, title, y_title, baseline=None, y_factor=1.0, annotation_mode='peak'):
    t = months / 12.0
    val = values * y_factor
    p90 = np.percentile(val, 10, axis=0)
    p50 = np.percentile(val, 50, axis=0)
    p10 = np.percentile(val, 90, axis=0)
    fig = go.Figure()
    draw_idx = np.linspace(0, len(val)-1, min(120, len(val)), dtype=int)
    for i in draw_idx:
        fig.add_trace(go.Scatter(x=t, y=val[i], mode='lines', hoverinfo='skip', showlegend=False,
                                 line=dict(color='rgba(0,255,140,0.12)', width=1.0)))
    fig.add_trace(go.Scatter(x=t, y=p90, name='P90', mode='lines', line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=t, y=p50, name='P50', mode='lines', line=dict(color='black', width=3.2)))
    fig.add_trace(go.Scatter(x=t, y=p10, name='P10', mode='lines', line=dict(color='purple', width=3)))
    if baseline is not None:
        base = np.percentile(baseline * y_factor, 50, axis=0)
        fig.add_trace(go.Scatter(x=t, y=base, name='Base P50', mode='lines',
                                 line=dict(color='gray', width=2, dash='dash')))
    # annotate values
    if annotation_mode == 'peak':
        idx_p90 = int(np.argmax(p90)); idx_p50 = int(np.argmax(p50)); idx_p10 = int(np.argmax(p10))
    else:
        idx_p90 = idx_p50 = idx_p10 = len(t)-1
    annos = [
        (p90[idx_p90], t[idx_p90], 'red'),
        (p50[idx_p50], t[idx_p50], 'black'),
        (p10[idx_p10], t[idx_p10], 'purple'),
    ]
    for yv, xv, color in annos:
        fig.add_annotation(x=xv, y=yv, text=f'{yv:,.2f}', showarrow=False,
                           bgcolor='white', bordercolor=color, borderwidth=2,
                           font=dict(color=color, size=13), xshift=28, yshift=12)
    fig.update_layout(title=title, template='plotly_white', height=460,
                      paper_bgcolor='#e9e9e9', plot_bgcolor='#e9e9e9',
                      xaxis_title='Time (years)', yaxis_title=y_title,
                      legend=dict(x=0.98, y=0.98, xanchor='right', bgcolor='rgba(255,255,255,0.8)'),
                      margin=dict(l=20, r=20, t=55, b=20))
    fig.update_xaxes(showgrid=True, gridcolor='#a7a7a7', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#a7a7a7', zeroline=False)
    return fig


def distribution_scatter(values, title, y_title, count=None, color='royalblue'):
    v = np.asarray(values)
    x = np.arange(1, len(v)+1)
    p90, p50, p10 = np.percentile(v, [10, 50, 90])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=v, mode='markers', name='MC runs',
                             marker=dict(color='rgba(30,200,120,0.35)', size=7),
                             text=[f'Well count: {c}' if count is not None else '' for c in (count if count is not None else x)],
                             hovertemplate='Run %{x}<br>Value %{y:,.2f}<br>%{text}<extra></extra>'))
    for label, yv, col in [('P90', p90, 'red'), ('P50', p50, 'black'), ('P10', p10, 'purple')]:
        fig.add_hline(y=yv, line_color=col, line_width=3)
        fig.add_annotation(x=max(x), y=yv, text=f'{label}: {yv:,.2f}', showarrow=False,
                           xshift=35, bgcolor='white', bordercolor=col, borderwidth=2,
                           font=dict(color=col, size=12))
    fig.update_layout(title=title, template='plotly_white', height=420,
                      paper_bgcolor='#e9e9e9', plot_bgcolor='#e9e9e9',
                      xaxis_title='Monte Carlo realization', yaxis_title=y_title,
                      margin=dict(l=20, r=20, t=55, b=20))
    fig.update_xaxes(showgrid=True, gridcolor='#b2b2b2')
    fig.update_yaxes(showgrid=True, gridcolor='#b2b2b2')
    return fig


def well_count_chart(counts):
    p90, p50, p10 = np.percentile(counts, [10, 50, 90])
    hist = px.histogram(x=counts, nbins=max(10, min(30, len(np.unique(counts))+4)))
    hist.update_traces(marker_color='rgba(0,180,120,0.65)', marker_line_color='white', marker_line_width=1)
    for label, xval, col in [('P90', p90, 'red'), ('P50', p50, 'black'), ('P10', p10, 'purple')]:
        hist.add_vline(x=xval, line_width=3, line_color=col)
        hist.add_annotation(x=xval, y=1, yref='paper', text=f'{label}: {xval:.1f}', showarrow=False,
                            bgcolor='white', bordercolor=col, borderwidth=2, yshift=-20)
    hist.update_layout(title='Well Count Distribution', template='plotly_white', height=360,
                       paper_bgcolor='#e9e9e9', plot_bgcolor='#e9e9e9',
                       xaxis_title='Number of wells selected', yaxis_title='Frequency',
                       margin=dict(l=20, r=20, t=55, b=20))
    hist.update_xaxes(showgrid=True, gridcolor='#b2b2b2')
    hist.update_yaxes(showgrid=True, gridcolor='#b2b2b2')
    return hist


def display_table_editor(prefs: UnitPrefs):
    st.subheader('Uncertainty parameter ranges')
    st.caption('Edit Low / Most likely / High for each parameter. Values are shown in the selected display-unit system, then converted internally to the field-unit solver.')
    df_internal = st.session_state['params']
    df_disp = display_table_for_editor(df_internal, prefs.system)
    edited_disp = st.data_editor(
        df_disp, hide_index=True, num_rows='fixed', use_container_width=True, height=570,
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
        }, key=f'parameter_grid_{prefs.system}')
    edited_internal = editor_to_internal(edited_disp, df_internal, prefs.system)
    st.session_state['params'] = edited_internal.copy()
    errs = validate_parameters(edited_internal)
    if errs:
        st.error('Invalid inputs: ' + ' | '.join(errs[:6]))
    c1, c2 = st.columns(2)
    with c1:
        st.download_button('⬇ Export current ranges (CSV)', edited_internal.to_csv(index=False).encode('utf-8'),
                           'frac_uncertainty_ranges.csv', 'text/csv', use_container_width=True)
    with c2:
        upload = st.file_uploader('Import ranges CSV', type=['csv'], key='upload_ranges')
        if upload is not None:
            try:
                imported = pd.read_csv(upload)
                if (set(imported['Key']) != set(df_internal['Key'])) or set(df_internal.columns) != set(imported.columns):
                    st.error('Import must have exact keys and columns as exported template.')
                elif validate_parameters(imported):
                    st.error('Imported CSV has invalid ranges: ' + '; '.join(validate_parameters(imported)[:5]))
                elif st.button('Apply imported ranges'):
                    st.session_state['params'] = imported[df_internal.columns].copy()
                    st.session_state['last_result'] = None
                    st.rerun()
            except Exception as exc:
                st.error(f'Could not read CSV: {exc}')
    return errs


def result_archive(res) -> bytes:
    curves = summarize_curves(res)
    monthly = {'Run': np.repeat(np.arange(1, len(res.well_count) + 1), len(res.months)),
               'Month': np.tile(res.months, len(res.well_count))}
    for name, ar in [('Baseline rate bopd', res.base_rate), ('Post frac rate bopd', res.frac_rate),
                     ('Incremental rate bopd', res.incremental_rate), ('Baseline cum stb', res.base_cum),
                     ('Post frac cum stb', res.frac_cum), ('Incremental cum stb', res.incremental_cum)]:
        monthly[name] = ar.ravel()
    extra = pd.DataFrame({'Run': np.arange(1, len(res.well_count)+1), 'Well count': res.well_count,
                          'Success count': res.success_count, 'IOR per well bopd': res.ior_per_well,
                          'EOR per well stb': res.eor_per_well, 'Incremental per well stb': res.inc_per_well})
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zz:
        zz.writestr('monthly_percentiles.csv', curves.to_csv(index=False))
        zz.writestr('run_level_results.csv', res.diagnostic.to_csv(index=False))
        zz.writestr('all_realizations_monthly.csv', pd.DataFrame(monthly).to_csv(index=False))
        zz.writestr('per_well_summary.csv', extra.to_csv(index=False))
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
        years = int(st.number_input('Forecast years', min_value=1, max_value=10, value=8, step=1))
    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        initial_rate = st.radio('Pre-frac baseline initialization', ['Measured rate anchor', 'Mechanistic radial Darcy PI'], horizontal=True)
    with c2:
        grid_size = st.select_slider('Grid lookup resolution', options=[15,19,21,25,29], value=21)
    with c3:
        seed = int(st.number_input('Random seed', min_value=0, max_value=10**8, value=42))
    st.info('Well count is treated as an uncertain input. For each realization the model samples a well count from the Low / Most Likely / High range, then samples reservoir and fracture properties independently for those wells.')
    errors = display_table_editor(PREFS)
    if not (nlo <= nmid <= nhi):
        st.error('Well count inputs must satisfy Low ≤ Most likely ≤ High.')
    run_now = st.button('▶ Run Monte Carlo', type='primary', use_container_width=True, disabled=bool(errors) or not (nlo <= nmid <= nhi))
    if run_now:
        with st.spinner('Running Monte Carlo forecast...'):
            try:
                res = run_mc(st.session_state['params'], nlo, nmid, nhi, n_runs=simulations, years=years,
                             seed=seed, q0_mode=initial_rate, grid_size=grid_size)
                st.session_state['last_result'] = res
            except Exception as exc:
                st.error(f'Run failed: {exc}')
    res = st.session_state['last_result']
    if res is not None:
        rate_factor, rate_label = rate_factor_and_label(PREFS.rate_scale)
        cum_factor, cum_label = cum_factor_and_label(PREFS.cum_scale)
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        final_inc = res.incremental_cum[:, -1] * cum_factor
        final_rate = res.frac_rate[:, 0] * rate_factor
        with m1:
            st.metric(f'P50 Incremental Cum ({cum_label})', f'{np.percentile(final_inc, 50):,.2f}')
        with m2:
            st.metric(f'P10 Incremental Cum ({cum_label})', f'{np.percentile(final_inc, 90):,.2f}')
        with m3:
            st.metric('P50 Well Count', f'{np.percentile(res.well_count, 50):.0f}')
        with m4:
            st.metric(f'P50 Initial Frac Rate ({rate_label})', f'{np.percentile(final_rate, 50):,.2f}')

        st.plotly_chart(percentile_cloud(res.frac_rate, res.months, 'Total Oil Production Profile', f'Oil Rate ({rate_label})', baseline=res.base_rate, y_factor=rate_factor, annotation_mode='peak'), use_container_width=True)
        st.plotly_chart(percentile_cloud(res.incremental_cum, res.months, 'Incremental Oil Production (P90, P50, P10)', f'Incremental Production ({cum_label})', baseline=None, y_factor=cum_factor, annotation_mode='end'), use_container_width=True)
        c_left, c_right = st.columns(2)
        with c_left:
            st.plotly_chart(percentile_cloud(res.incremental_rate, res.months, 'Incremental Oil Rate Profile', f'Incremental Rate ({rate_label})', baseline=None, y_factor=rate_factor, annotation_mode='peak'), use_container_width=True)
        with c_right:
            st.plotly_chart(well_count_chart(res.well_count), use_container_width=True)
        s1, s2 = st.columns(2)
        ior_vals = res.ior_per_well * rate_factor
        eor_vals = res.eor_per_well * cum_factor
        with s1:
            st.plotly_chart(distribution_scatter(ior_vals, 'IOR per Well Scatter', f'Initial Oil Rate per Well ({rate_label})', count=res.well_count), use_container_width=True)
        with s2:
            st.plotly_chart(distribution_scatter(eor_vals, 'EOR per Well Scatter', f'Estimated Oil Recovery per Well ({cum_label})', count=res.well_count), use_container_width=True)
        st.download_button('⬇ Download result package (.zip)', result_archive(res), 'frac_hyperion_v2_results.zip', 'application/zip', use_container_width=True)

elif page == '2D Fracture Grid':
    st.markdown('### 2D Fracture Grid & Productivity Gain')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = int(st.select_slider('Grid size', options=[15,19,21,25,29], value=21))
    with c2:
        xf_frac = st.slider('Fracture half-length / domain side', 0.0, 0.49, 0.18, 0.01)
    with c3:
        link_ratio = st.select_slider('Enhanced link ratio Cf/(k·dy)·hf/h', options=[0.1,0.3,1,3,10,30,100,300,1000], value=30)
    with c4:
        show_contours = st.checkbox('Show pressure contours', value=True)
    q_base = solve_grid(n, 0, 0.0)
    q_frac, pmap = solve_grid(n, xf_frac, link_ratio, return_map=True)
    multiplier = q_frac / q_base
    x = np.arange(n); y = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=pmap, x=x, y=y, colorscale='Blues', colorbar_title='Normalized Pressure'))
    c = n // 2
    span = xf_frac * n
    fig.add_trace(go.Scatter(x=[c-span, c+span], y=[c, c], mode='lines', line=dict(color='red', width=6), name='Fracture'))
    fig.add_trace(go.Scatter(x=[c], y=[c], mode='markers', marker=dict(size=12, color='black'), name='Well'))
    if show_contours:
        fig.add_trace(go.Contour(z=pmap, x=x, y=y, showscale=False, line=dict(color='white', width=1), contours=dict(coloring='none')))
    fig.update_layout(height=560, title=f'2D Grid Pressure Map | Productivity Multiplier = {multiplier:.2f}×', template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.info('This page uses the same normalized transmissibility concept that feeds the Monte Carlo productivity multiplier. It is a transparent educational approximation, not a commercial hydraulic fracture simulator.')

elif page == '3D Fracture View':
    st.markdown('### 3D Fracture Schematic')
    c1, c2, c3 = st.columns(3)
    with c1:
        length = st.slider('Fracture half-length', 20.0, 500.0, float(mode('xf_ft')), 5.0)
    with c2:
        height = st.slider('Fracture contacted pay fraction', 0.1, 1.0, float(mode('frac_height_frac')), 0.05)
    with c3:
        width = st.slider('Visual width factor', 0.1, 3.0, 1.0, 0.1)
    # 3D box and fracture plane
    x = np.array([-1,1,1,-1,-1,-1,1,1,-1])
    y = np.array([-1,-1,1,1,-1,-1,-1,1,1])
    z = np.array([0,0,0,0,0,2,2,2,2])
    fig = go.Figure()
    edges = [(0,1),(1,2),(2,3),(3,0),(5,6),(6,7),(7,8),(8,5),(0,5),(1,6),(2,7),(3,8)]
    for i,j in edges:
        fig.add_trace(go.Scatter3d(x=[x[i],x[j]], y=[y[i],y[j]], z=[z[i],z[j]], mode='lines', showlegend=False, line=dict(color='gray', width=4)))
    frac_x = np.array([-length/500, length/500, length/500, -length/500, -length/500])
    frac_y = np.array([0,0,0,0,0])
    frac_z = np.array([1-height, 1-height, 1+height, 1+height, 1-height])
    fig.add_trace(go.Scatter3d(x=frac_x, y=frac_y, z=frac_z, mode='lines', line=dict(color='red', width=10*width), name='Fracture'))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,2], mode='lines', line=dict(color='black', width=7), name='Well'))
    fig.update_layout(height=560, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown('### Guidelines & SPE References')
    st.markdown(r"""
**What the app does**
- Samples Low / Most likely / High uncertainty ranges for rock, pressure, well and fracture parameters.
- Samples a **well-count distribution** each run.
- Computes a **2D transmissibility-based fracture gain** and applies it to the post-frac productivity denominator.
- Forecasts baseline and frac cases, then reports P90 / P50 / P10 outcomes.

**Core equations (screening level)**
1. Radial base productivity:
\[
q_o = \frac{0.00708\,k h (\bar p - p_{wf})}{\mu_o B_o (\ln(r_e/r_w)-0.75 + s)}
\]
2. Dimensionless finite-conductivity indicator:
\[
F_{CD} = \frac{k_f w_f}{k x_f}
\]
3. Post-frac productivity gain is approximated numerically from the 2D grid solver and then mapped into an equivalent post-frac denominator.

**Important limitations**
- Single-phase screening model.
- No explicit fracture propagation physics.
- No transient pressure analysis matching.
- No well-to-well interference, scheduling, injectors or multiphase relative permeability.
- Therefore it is **not** a REVEAL clone and should not be used as a certified reserves tool.

**Suggested references**
- Prats, M. 1961. *Effect of Vertical Fractures on Reservoir Behavior.*
- Cinco-Ley, H. and Samaniego, F. 1981. *Transient Pressure Analysis for Fractured Wells.*
- SPE 88960 and similar papers on practical fracture conductivity and post-frac productivity screening.

**How to improve next**
- Well-by-well input sheet.
- History matching versus actual well performance.
- Waterflood / injector support.
- Multiphase relative permeability.
- Frac schedule and ramp-up logic.
- Economics, NPV and ranking.
""")
