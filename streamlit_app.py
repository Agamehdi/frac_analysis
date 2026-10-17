"""Frac Hyperion v2.1 – GitHub/Streamlit Cloud-ready demonstration application."""
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
st.markdown('<div class="hero"><h1>FRAC HYPERION <small style="font-size:0.42em">V2.1</small></h1>'
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
        rate_scale = st.sidebar.radio('Rate unit', ['m³/d', '10³ m³/d'], horizontal=True,
                                      help='10³ m³/d means thousand cubic metres per day.')
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
    if unit == '10³ m³/d':
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


# Consistent high-contrast presentation: white background and bold lettering.
P90_COLOR = '#F32626'
P50_COLOR = '#101820'
P10_COLOR = '#8A05A8'
CLOUD_COLOR = 'rgba(0, 220, 115, 0.15)'
FONT_FAMILY = 'Arial Black, Arial, sans-serif'


def chart_style(fig, title, x_title, y_title, height=410, right_margin=58):
    fig.update_layout(
        template='plotly_white', title=dict(text=f'<b>{title}</b>', x=0.02,
                                            font=dict(family=FONT_FAMILY, size=16, color='#101820')),
        height=height, paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        font=dict(family=FONT_FAMILY, size=11, color='#101820'),
        xaxis_title=f'<b>{x_title}</b>', yaxis_title=f'<b>{y_title}</b>',
        margin=dict(l=68, r=right_margin, t=67, b=60),
        hovermode='closest',
#        legend=dict(orientation='h', x=0, y=1.08, xanchor='left', yanchor='bottom',
#                    bgcolor='rgba(255,255,255,0.85)', font=dict(size=10)))
       legend=dict(
           orientation='v',
           x=0.99,
           y=0.99,
           xanchor='right',
           yanchor='top',
           bgcolor='rgba(255,255,255,0.90)',
           font=dict(size=11)
       )


       
    fig.update_xaxes(showgrid=True, gridcolor='#E0E6EC', linecolor='#AAB5C2',
                     linewidth=1, showline=True, zeroline=False,
                     ticks='outside', tickfont=dict(family=FONT_FAMILY, size=11))
    fig.update_yaxes(showgrid=True, gridcolor='#E0E6EC', linecolor='#AAB5C2',
                     linewidth=1, showline=True, zeroline=False,
                     ticks='outside', tickfont=dict(family=FONT_FAMILY, size=11))
    return fig


def percentile_cloud(values, months, title, y_title, baseline=None, y_factor=1.0,
                     annotation_mode='peak'):
    t = np.asarray(months) / 12.0
    val = np.asarray(values, dtype=float) * y_factor
    p90, p50, p10 = np.percentile(val, (10, 50, 90), axis=0)
    fig = go.Figure()
    draw_idx = np.linspace(0, len(val)-1, min(150, len(val)), dtype=int)
    for i in draw_idx:
        fig.add_trace(go.Scatter(
            x=t, y=val[i], mode='lines', hoverinfo='skip', showlegend=False,
            line=dict(color=CLOUD_COLOR, width=1)))
    for lab, arr, color, width in (
        ('P90', p90, P90_COLOR, 3.2), ('P50', p50, P50_COLOR, 3.7),
        ('P10', p10, P10_COLOR, 3.2)):
        fig.add_trace(go.Scatter(
            x=t, y=arr, mode='lines', name=lab,
            line=dict(color=color, width=width),
            hovertemplate=f'{lab}: %{{y:,.3f}}<extra></extra>'))
    if baseline is not None:
        base = np.median(np.asarray(baseline)*y_factor, axis=0)
        fig.add_trace(go.Scatter(x=t, y=base, mode='lines', name='Base P50',
                                 line=dict(color='#65758B', width=2.2, dash='dash'),
                                 hovertemplate='Base P50: %{y:,.3f}<extra></extra>'))
    for arr, color in ((p90, P90_COLOR),(p50,P50_COLOR),(p10,P10_COLOR)):
        k = int(np.argmax(arr)) if annotation_mode == 'peak' else len(t)-1
        # Rightmost cumulative annotations are kept within the plot; don't crop labels.
        xshift = -31 if k == len(t)-1 else 28
        fig.add_annotation(x=t[k], y=float(arr[k]), text=f'<b>{arr[k]:,.2f}</b>',
                           showarrow=False, xshift=xshift, yshift=14,
                           bgcolor='#FFFFFF', bordercolor=color, borderwidth=1.5,
                           borderpad=3, font=dict(color=color, size=11, family=FONT_FAMILY))
    chart_style(fig, title, 'Time (years)', y_title, height=440)
    fig.update_xaxes(range=[0, float(t[-1]) * 1.02])
    return fig


def distribution_scatter(values, title, y_title, count, well_index=None, run_index=None):
    """One dot per real well draw (V2.1), or one per MC run for old engines."""
    v = np.asarray(values, dtype=float)
    counts = np.asarray(count, dtype=int)
    full_n = len(v)
    if well_index is None:
        index = counts.astype(float)
        run = np.arange(1, full_n+1)
        x_title = 'Number of wells (sampled project size)'
    else:
        index = np.asarray(well_index, dtype=float)
        run = np.asarray(run_index, dtype=int)
        x_title = 'Well index within Monte Carlo realization'
    # Keep clouds responsive on Streamlit Cloud: markers are sampled for
    # display, but percentiles always use ALL physical well draws.
    selected = np.linspace(0, full_n-1, min(full_n, 9000), dtype=int)
    jitter = 0.18 * np.sin(selected * 2.399963229728653)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=index[selected]+jitter, y=v[selected], mode='markers', name='Monte Carlo draws',
        marker=dict(color='rgba(0,197,109,0.42)', size=5.5),
        customdata=np.column_stack((run[selected], counts[selected])),
        hovertemplate='MC run %{customdata[0]}<br>Wells in project %{customdata[1]}'
                      '<br>Value %{y:,.3f}<extra></extra>'))
    percentile_values = np.percentile(v, [10, 50, 90])
    for (name, col), yv in zip((('P90',P90_COLOR),('P50',P50_COLOR),('P10',P10_COLOR)),percentile_values):
        fig.add_hline(y=float(yv), line_color=col, line_width=2.7,
                      annotation_text=f'<b>{name}: {yv:,.2f}</b>',
                      annotation_position='top right',
                      annotation_font_color=col, annotation_font_size=10)
    chart_style(fig, title, x_title, y_title, height=440, right_margin=65)
    fig.update_xaxes(range=[max(0, float(np.min(index))-0.8), float(np.max(index))+0.8])
    return fig


def well_count_chart(counts):
    counts = np.asarray(counts, dtype=int)
    uniques, frequencies = np.unique(counts, return_counts=True)
    fig = go.Figure(go.Bar(x=uniques, y=frequencies, name='Realizations',
                           marker=dict(color='#28C89A', line=dict(color='#11765E', width=0.5))))
    for label, xval, col in zip(('P90', 'P50', 'P10'), np.percentile(counts,[10,50,90]),
                                 (P90_COLOR,P50_COLOR,P10_COLOR)):
        fig.add_vline(x=float(xval), line_width=2.8, line_color=col,
                      annotation_text=f'<b>{label}: {xval:.0f}</b>',
                      annotation_position='top', annotation_font_color=col)
    chart_style(fig, 'Well Count Distribution', 'Number of Wells',
                'Monte Carlo Frequency', height=440)
    if len(uniques)<30:
        fig.update_xaxes(dtick=max(1, int((uniques.max()-uniques.min())//12)))
    return fig


def per_well_metrics(res):
    """Version-safe calculations: work with both V1 and V2 Result objects.

    IOR = average post-frac oil rate during first forecast month per selected well
    (including frac/cleanup downtime); EOR = cumulative post-frac oil produced
    during chosen forecast horizon per selected well, NOT reserve booking or EUR.
    """
    counts = np.asarray(res.well_count, dtype=float)
    if np.any(counts <= 0):
        raise ValueError('Well counts must be positive')
    ior = np.asarray(res.frac_rate[:, 0], dtype=float) / counts
    eor = np.asarray(res.frac_cum[:, -1], dtype=float) / counts
    incremental = np.asarray(res.incremental_cum[:, -1], dtype=float) / counts
    return ior, eor, incremental


def individual_well_scatter_data(res):
    """Returns individual well MC draws if engine supports them.

    Old GitHub engines are supported with project-average-per-well draws.
    """
    mask = getattr(res, 'well_active_mask', None)
    ior_matrix = getattr(res, 'ior_well_draws', None)
    eor_matrix = getattr(res, 'eor_well_draws', None)
    if mask is None or ior_matrix is None or eor_matrix is None:
        ior, eor, _ = per_well_metrics(res)
        counts = np.asarray(res.well_count, dtype=int)
        return ior, eor, counts, None, None, False
    mask = np.asarray(mask, dtype=bool)
    ior_matrix = np.asarray(ior_matrix)
    eor_matrix = np.asarray(eor_matrix)
    if ior_matrix.shape != mask.shape or eor_matrix.shape != mask.shape:
        raise ValueError('Per-well arrays and well-count mask are inconsistent')
    n_runs, max_wells = mask.shape
    run_ids = np.broadcast_to(np.arange(1,n_runs+1)[:,None], mask.shape)[mask]
    well_ids = np.broadcast_to(np.arange(1,max_wells+1)[None,:],mask.shape)[mask]
    counts = np.broadcast_to(np.asarray(res.well_count)[:,None],mask.shape)[mask]
    return ior_matrix[mask], eor_matrix[mask], counts, well_ids, run_ids, True


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
    monthly = {'Run': np.repeat(np.arange(1, len(res.well_count)+1), len(res.months)),
               'Month': np.tile(res.months, len(res.well_count))}
    for name, ar in [('Baseline rate bopd', res.base_rate), ('Post frac rate bopd', res.frac_rate),
                     ('Incremental rate bopd', res.incremental_rate), ('Baseline cum stb', res.base_cum),
                     ('Post frac cum stb', res.frac_cum), ('Incremental cum stb', res.incremental_cum)]:
        monthly[name] = ar.ravel()
    ior, eor, inc = per_well_metrics(res)
    extra = pd.DataFrame({
        'Run': np.arange(1, len(res.well_count)+1), 'Well count': res.well_count,
        'Success count': res.success_count,
        'IOR first-month avg per well bopd': ior,
        'EOR forecast-horizon per well stb': eor,
        'Incremental forecast per well stb': inc})
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, 'w', zipfile.ZIP_DEFLATED) as zz:
        zz.writestr('monthly_percentiles.csv', curves.to_csv(index=False))
        zz.writestr('run_level_results.csv', res.diagnostic.to_csv(index=False))
        zz.writestr('all_realizations_monthly.csv', pd.DataFrame(monthly).to_csv(index=False))
        zz.writestr('per_well_summary.csv', extra.to_csv(index=False))
        if getattr(res, 'well_active_mask', None) is not None:
            mask = np.asarray(res.well_active_mask, dtype=bool)
            n_run, n_well = mask.shape
            idx = np.where(mask)
            raw_wells = pd.DataFrame({
                'Run': idx[0] + 1, 'Well index': idx[1] + 1,
                'IOR first-month avg bopd': np.asarray(res.ior_well_draws)[mask],
                'EOR forecast-horizon stb': np.asarray(res.eor_well_draws)[mask]})
            zz.writestr('individual_well_draws.csv', raw_wells.to_csv(index=False))
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

        st.subheader('Probabilistic Production Forecasts')
        r1left, r1right = st.columns(2, gap='medium')
        with r1left:
            st.plotly_chart(percentile_cloud(
                res.frac_rate, res.months, 'Total Oil Production Profile',
                f'Oil Rate ({rate_label})', baseline=res.base_rate,
                y_factor=rate_factor, annotation_mode='peak'),
                use_container_width=True)
        with r1right:
            st.plotly_chart(percentile_cloud(
                res.incremental_cum, res.months, 'Incremental Oil Production (P90 / P50 / P10)',
                f'Incremental Oil ({cum_label})', y_factor=cum_factor,
                annotation_mode='end'), use_container_width=True)
        r2left, r2right = st.columns(2, gap='medium')
        with r2left:
            st.plotly_chart(percentile_cloud(
                res.incremental_rate, res.months, 'Incremental Oil Rate Profile',
                f'Incremental Rate ({rate_label})', y_factor=rate_factor,
                annotation_mode='peak'), use_container_width=True)
        with r2right:
            st.plotly_chart(well_count_chart(res.well_count), use_container_width=True)

        st.subheader('Per-Well Results — Monte Carlo Scatter')
        ior, eor, counts, well_ids, run_ids, individual = individual_well_scatter_data(res)
        if individual:
            st.caption('Each dot represents one active well in one Monte Carlo run. '
                       'P90/P50/P10 are calculated across all simulated well outcomes. '
                       'IOR is first-month average post-frac oil rate including downtime; '
                       'EOR is the selected forecast-period post-frac cumulative oil, not ultimate reserves.')
        else:
            st.info('Older frac_physics.py detected: scatter plots display project-average '
                    'per-well values per realization. Replace frac_physics.py from this ZIP '
                    'to show every well individually.')
        r3left, r3right = st.columns(2, gap='medium')
        with r3left:
            st.plotly_chart(distribution_scatter(
                ior * rate_factor, 'Initial Oil Rate (IOR) per Well',
                f'IOR per Well ({rate_label})', counts, well_ids, run_ids),
                use_container_width=True)
        with r3right:
            st.plotly_chart(distribution_scatter(
                eor * cum_factor, 'EOR per Well — Forecast Horizon',
                f'EOR per Well ({cum_label})', counts, well_ids, run_ids),
                use_container_width=True)
        st.download_button('⬇ Download result package (.zip)', result_archive(res),
                           'frac_hyperion_v2_1_results.zip', 'application/zip', use_container_width=True)


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
