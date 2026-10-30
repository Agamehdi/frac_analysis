"""Frac Hyperion v2.3 – one-click all-realization KAPPA Monte Carlo matching."""
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
from kappa_benchmark import (DEFAULT_BENCHMARK, POST, INCREMENTAL, GLOBAL, LOCAL,
                             check_benchmark, reference_rates, reference_signature,
                             fit_benchmark, calibration_factor, performance)

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
st.markdown('<div class="hero"><h1>FRAC HYPERION <small style="font-size:0.42em">V2.3</small></h1>'
            '<p>Probabilistic fracture productivity • Monte Carlo forecast • 2D grid flow model</p></div>',
            unsafe_allow_html=True)

if 'params' not in st.session_state:
    st.session_state['params'] = default_parameter_table()
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None
if 'kappa_points' not in st.session_state:
    st.session_state['kappa_points'] = DEFAULT_BENCHMARK.copy()
if 'kappa_calibration' not in st.session_state:
    st.session_state['kappa_calibration'] = None
if 'kappa_definition_saved' not in st.session_state:
    st.session_state['kappa_definition_saved'] = INCREMENTAL
if 'kappa_method_saved' not in st.session_state:
    st.session_state['kappa_method_saved'] = LOCAL
if 'mc_apply_kappa_widget' not in st.session_state:
    st.session_state['mc_apply_kappa_widget'] = False
if 'raw_result' not in st.session_state:
    st.session_state['raw_result'] = None
if 'mc_match_comparison' not in st.session_state:
    st.session_state['mc_match_comparison'] = None
if 'mc_match_parameters' not in st.session_state:
    st.session_state['mc_match_parameters'] = None

PAGES = ['Monte Carlo Studio', 'KAPPA Benchmark & Match', '2D Fracture Grid', '3D Fracture View', 'Guidelines & SPE References']
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
        margin=dict(l=68, r=right_margin, t=69, b=94),
        hovermode='closest',
        legend=dict(orientation='h', x=0, y=-0.19, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.85)', font=dict(size=10)))
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


def _calibrate_from_mc_inputs(table, mode_name, grid_size):
    """Fit with the exact modal inputs used in the MC run; no prerequisite Save tab."""
    bench = check_benchmark(st.session_state['kappa_points'])
    raw = reference_rates(table, bench['Perm (mD)'].to_numpy(float), mode_name, grid_size)
    definition = st.session_state.get('kappa_definition_saved', INCREMENTAL)
    method = st.session_state.get('kappa_method_saved', LOCAL)
    sig = reference_signature(table, mode_name, grid_size)
    return fit_benchmark(bench, raw, definition, method, sig)


def _apply_kappa_sampling_span(params, enable):
    """Optional copy: never overwrite user-edited parameter table."""
    p = params.copy(deep=True)
    if enable:
        bench = check_benchmark(st.session_state['kappa_points'])
        lo = float(bench['Perm (mD)'].min())
        hi = float(bench['Perm (mD)'].max())
        ix = p.index[p['Key'] == 'k_md'][0]
        mode_value = float(p.loc[ix, 'Most likely'])
        p.loc[ix, 'Low'] = lo
        p.loc[ix, 'High'] = hi
        p.loc[ix, 'Most likely'] = float(np.clip(mode_value, lo, hi))
        p.loc[ix, 'Distribution'] = 'Log-triangular'
    return p


def _enable_match_checkbox():
    """Button callback runs before widgets are instantiated on the next rerun."""
    st.session_state['mc_apply_kappa_widget'] = True
    st.session_state['mc_result_view'] = 'KAPPA-matched cloud'


def compare_cloud_plot(raw, matched, attr, title, ylabel, unit_factor=1.0,
                       raw_cloud=True):
    """Same seeds and sampled well properties, before/after comparison."""
    t = np.asarray(matched.months, dtype=float) / 12
    old = np.asarray(getattr(raw, attr)) * unit_factor
    new = np.asarray(getattr(matched, attr)) * unit_factor
    fig = go.Figure()
    ids = np.linspace(0, len(old)-1, min(85,len(old)), dtype=int)
    if raw_cloud:
        for i in ids:
            fig.add_trace(go.Scatter(x=t, y=old[i], mode='lines', showlegend=False,
                                     line=dict(color='rgba(100,116,139,0.12)', width=1), hoverinfo='skip'))
    for i in ids:
        fig.add_trace(go.Scatter(x=t, y=new[i], mode='lines', showlegend=False,
                                 line=dict(color='rgba(0,180,105,0.13)', width=1), hoverinfo='skip'))
    for quant, label, color in [(10,'P90',P90_COLOR),(50,'P50',P50_COLOR),(90,'P10',P10_COLOR)]:
        fig.add_trace(go.Scatter(x=t,y=np.percentile(old,quant,axis=0), mode='lines',
                                 name=f'Original {label}', line=dict(color=color,width=1.8,dash='dot'),
                                 hovertemplate=f'Original {label}: %{{y:,.2f}}<extra></extra>'))
        fig.add_trace(go.Scatter(x=t,y=np.percentile(new,quant,axis=0), mode='lines',
                                 name=f'Matched {label}', line=dict(color=color,width=3.4),
                                 hovertemplate=f'Matched {label}: %{{y:,.2f}}<extra></extra>'))
    chart_style(fig,title,'Time (years)',ylabel,height=490)
    return fig


def before_after_table(original, matched):
    entries = [('Initial post-frac rate','frac_rate',0,'stb/d'),
               ('Initial incremental rate','incremental_rate',0,'stb/d'),
               ('End incremental rate','incremental_rate',-1,'stb/d'),
               ('Forecast incremental oil','incremental_cum',-1,'stb')]
    data=[]
    for label, attr, t, units in entries:
        orig = getattr(original,attr)[:,t]
        mtch = getattr(matched,attr)[:,t]
        for q,name in [(10,'P90'),(50,'P50'),(90,'P10')]:
            v0,v1=float(np.percentile(orig,q)),float(np.percentile(mtch,q))
            data.append({'Metric':label,'Percentile':name,'Unit':units,
                         'Original':v0,'KAPPA-matched':v1,'Difference':v1-v0,
                         'Change (%)': (100*(v1/v0-1)) if abs(v0)>1e-9 else np.nan})
    return pd.DataFrame(data)


def result_archive(res, original=None, comparison=None) -> bytes:
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
        zz.writestr('input_ranges.csv', st.session_state.get('mc_match_parameters', st.session_state['params']).to_csv(index=False))
        zz.writestr('calibration_used.txt',
                    ('Method: '+str(res.calibration_method)+'\nDefinition: '+str(res.calibration_definition))
                    if res.calibration_method else 'Calibration: OFF')
        if res.calibration_method:
            zz.writestr('kappa_benchmark_points.csv', st.session_state['kappa_points'].to_csv(index=False))
        if original is not None:
            zz.writestr('original_monthly_percentiles.csv', summarize_curves(original).to_csv(index=False))
            zz.writestr('original_run_level_results.csv', original.diagnostic.to_csv(index=False))
            zz.writestr('original_all_realizations_monthly.csv', pd.DataFrame({
                'Run':np.repeat(np.arange(1,len(original.well_count)+1),len(original.months)),
                'Month':np.tile(original.months,len(original.well_count)),
                'Original rate bopd':original.frac_rate.ravel(),
                'Original incremental rate bopd':original.incremental_rate.ravel(),
                'Original incremental cumulative stb':original.incremental_cum.ravel()}).to_csv(index=False))
            zz.writestr('before_after_percentiles.csv',before_after_table(original,res).to_csv(index=False))
        if comparison is not None:
            zz.writestr('kappa_model_anchor_match.csv',comparison.to_csv(index=False))
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
        initial_rate = st.radio('Pre-frac baseline initialization',
                                ['Measured rate anchor', 'Mechanistic radial Darcy PI'],
                                horizontal=True, key='mc_rate_mode_widget')
    with c2:
        grid_size = st.select_slider('Grid lookup resolution', options=[15,19,21,25,29], value=21,
                                     key='mc_grid_size_widget')
    with c3:
        seed = int(st.number_input('Random seed', min_value=0, max_value=10**8, value=42))
    st.info('Well count is treated as an uncertain input. For each realization the model samples a well count from the Low / Most Likely / High range, then samples reservoir and fracture properties independently for those wells.')
    errors = display_table_editor(PREFS)
    if not (nlo <= nmid <= nhi):
        st.error('Well count inputs must satisfy Low ≤ Most likely ≤ High.')
    # The checkbox remains enabled: unlike v2.2 there is no prerequisite saved
    # signature or manually synchronized baseline mode / grid on another page.
    st.markdown('#### KAPPA calibration of the entire Monte Carlo ensemble')
    st.checkbox('Apply KAPPA-matched calibration to Monte Carlo forecast',
                key='mc_apply_kappa_widget',
                help='Always available. Calibration is fitted automatically from the current MC inputs, '
                     'then used for EACH successful fractured well in EVERY realization and forecast month.')
    coverage = check_benchmark(st.session_state['kappa_points'])
    cspan0 = float(coverage['Perm (mD)'].min())
    cspan1 = float(coverage['Perm (mD)'].max())
    sample_full_k = st.checkbox(
        f'Include full KAPPA permeability span ({cspan0:g}–{cspan1:g} mD) in both MC scenarios',
        value=False, key='mc_expand_k',
        help='Optional: makes a temporary copy of the parameter ranges, changes only matrix '
             'permeability Low/High to benchmark extremes and uses Log-triangular draws. '
             'Original and Matched use exactly the same sampled reservoir cases. '
             'Does not edit your parameter table.')
    existing_k = st.session_state['params'].set_index('Key').loc['k_md']
    if existing_k['High'] < cspan1 or existing_k['Low'] > cspan0:
        st.caption(f"Current Monte Carlo k range: {existing_k['Low']:g}–{existing_k['High']:g} mD. "
                   f"KAPPA benchmarks: {cspan0:g}–{cspan1:g} mD. "
                   'The 50 and 100 mD anchors will not appear in the cloud unless you expand its k range.')
    if initial_rate == 'Measured rate anchor' and st.session_state['mc_apply_kappa_widget']:
        st.warning('Measured rate anchor makes the pre-frac oil rate independent of k. For a true k-dependent KAPPA PI comparison, consider Mechanistic radial Darcy PI. The existing baseline mode is preserved, not overridden.')
    st.caption('One click fits the KAPPA anchors to the current modal physics, runs paired forecasts '
               'with the same random seed and rebuilds all oil-rate clouds, P90/P50/P10 and cumulative oil. '
               'Only successful fractured wells are corrected. The base case is left unchanged.')
    actions = st.columns([1,1])
    with actions[0]:
        run_now = st.button('▶ Run Monte Carlo (selected calibration mode)',
                            use_container_width=True, disabled=bool(errors) or not(nlo<=nmid<=nhi))
    with actions[1]:
        direct_match = st.button('⚡ Match ENTIRE Monte Carlo Cloud to KAPPA',
                                 on_click=_enable_match_checkbox,
                                 type='primary', use_container_width=True,
                                 disabled=bool(errors) or not(nlo<=nmid<=nhi))
    if run_now or direct_match:
        use_match = direct_match or st.session_state['mc_apply_kappa_widget']
        with st.spinner('Computing full matched Monte Carlo ensemble...'
                        if use_match else 'Running original Monte Carlo...'):
            try:
                working = _apply_kappa_sampling_span(st.session_state['params'], sample_full_k)
                matching = None
                comp = None
                if use_match:
                    matching, comp = _calibrate_from_mc_inputs(working, initial_rate, grid_size)
                original = run_mc(working,nlo,nmid,nhi,n_runs=simulations,years=years,
                                  seed=seed,q0_mode=initial_rate,grid_size=grid_size)
                if use_match:
                    matched = run_mc(working,nlo,nmid,nhi,n_runs=simulations,years=years,
                                     seed=seed,q0_mode=initial_rate,grid_size=grid_size,
                                     calibration=matching)
                    st.session_state['last_result'] = matched
                    st.session_state['raw_result'] = original
                    st.session_state['kappa_calibration'] = matching
                    st.session_state['mc_match_comparison'] = comp
                    st.session_state['mc_benchmark_snapshot'] = check_benchmark(st.session_state['kappa_points']).copy()
                    st.session_state['mc_result_view'] = 'KAPPA-matched cloud'
                else:
                    st.session_state['last_result'] = original
                    st.session_state['raw_result'] = None
                    st.session_state['mc_match_comparison'] = None
                st.session_state['mc_match_parameters'] = working.copy()
            except Exception as exc:
                st.error(f'Monte Carlo failed: {exc}')
    result_matched = st.session_state['last_result']
    original_result = st.session_state.get('raw_result')
    paired = bool(result_matched is not None and original_result is not None
                  and result_matched.calibration_method)
    if result_matched is not None:
        if paired:
            snap = st.session_state.get('mc_benchmark_snapshot')
            if snap is not None and not snap.equals(coverage):
                st.warning('Benchmark inputs were edited since this cloud was calculated. Rerun matching to update the results.')
            st.success('Whole-ensemble KAPPA calibration complete: each successful fractured well in every '
                       'Monte Carlo realization was recalculated, including depletion and cumulative oil. '
                       'Original and matched scenarios used identical random draws.')
            view = st.radio('Show results in main charts',
                            ['KAPPA-matched cloud','Original cloud'], horizontal=True,
                            key='mc_result_view')
            res = result_matched if view == 'KAPPA-matched cloud' else original_result
            active_fit = st.session_state.get('kappa_calibration') or {}
            limits = active_fit.get('perms_md', [cspan0,cspan1])
            st.caption(f"Benchmark target: {result_matched.calibration_definition} | "
                       f"Matching: {result_matched.calibration_method}; "
                       f"applied across {limits[0]:g}–{limits[-1]:g} mD only. "
                       'No correction outside this span.')
        else:
            res = result_matched
            st.caption('Forecast calibration: OFF (original physics)')
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

        if paired:
            st.subheader('KAPPA | Entire Cloud — Before vs After')
            st.caption('Solid percentile curves = matched, dotted = original. Green = recalculated '
                       'individual realizations; grey = original realizations. P90/P50/P10 use all draws, '
                       'not only plotted samples.')
            draw_original = st.checkbox('Overlay original Monte Carlo realizations', value=True,
                                        key='mc_overlay_original')
            c_left, c_right = st.columns(2, gap='medium')
            with c_left:
                st.plotly_chart(compare_cloud_plot(original_result,result_matched,'frac_rate',
                                                   'Oil Rate | Cloud Before vs After',
                                                   f'Oil Rate ({rate_label})',rate_factor,
                                                   draw_original),use_container_width=True)
            with c_right:
                st.plotly_chart(compare_cloud_plot(original_result,result_matched,'incremental_cum',
                                                   'Incremental Cumulative | Before vs After',
                                                   f'Incremental ({cum_label})',cum_factor,
                                                   draw_original),use_container_width=True)
            st.plotly_chart(compare_cloud_plot(original_result,result_matched,'incremental_rate',
                                               'Incremental Oil Rate | Before vs After',
                                               f'Incremental Rate ({rate_label})',rate_factor,
                                               draw_original),use_container_width=True)
            st.dataframe(before_after_table(original_result,result_matched).style.format({
                'Original':'{:,.1f}','KAPPA-matched':'{:,.1f}',
                'Difference':'{:+,.1f}', 'Change (%)':'{:+,.1f}%'},na_rep='—'),
                use_container_width=True,hide_index=True)
            st.caption('KAPPA anchors specify only start-rate vs permeability. These charts retain the '
                       'existing model assumptions for decline, water-cut, pressure support, fracture '
                       'conductivity loss, downtime and failure. A time-dependent KAPPA history is needed '
                       'to calibrate the *shape* of decline.')
            with st.expander('KAPPA anchor matching details'):
                comp=st.session_state.get('mc_match_comparison')
                if comp is not None:
                    st.dataframe(comp,use_container_width=True,hide_index=True)
            st.divider()
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
        st.download_button('⬇ Download result package (.zip)',
                           result_archive(result_matched if paired else res,
                                          original_result if paired else None,
                                          st.session_state.get('mc_match_comparison') if paired else None),
                           'frac_hyperion_v2_3_cloud_match_results.zip',
                           'application/zip', use_container_width=True)


elif page == 'KAPPA Benchmark & Match':
    st.markdown('### KAPPA | Single-Zone Benchmark & Rate Matching')
    st.info('Compare stabilized **single-zone** initial deliverability at identical modal rock, fluid, '
            'pressure and fracture inputs. No downtime, uptime, depletion or Monte Carlo well count is '
            'included in this comparison. KAPPA points are **simulation benchmarks**, not field production history.')
    edit_left, edit_right = st.columns([3, 2], gap='large')
    with edit_left:
        st.markdown('**KAPPA benchmark data — editable**')
        edited = st.data_editor(st.session_state['kappa_points'], hide_index=True,
                                num_rows='dynamic', use_container_width=True, key='kappa_edit_table',
                                column_config={
                                    'Perm (mD)': st.column_config.NumberColumn('Permeability (mD)', min_value=0.000001,
                                                                               format='%.3f', required=True),
                                    'KAPPA Oil Rate (stb/d)': st.column_config.NumberColumn('KAPPA Oil Rate (stb/d)',
                                                                                             min_value=0.000001, format='%.2f', required=True),
                                })
        st.session_state['kappa_points'] = edited.copy()
        try:
            bench = check_benchmark(edited)
            st.download_button('⬇ Download KAPPA benchmark CSV', bench.to_csv(index=False).encode(),
                               'kappa_benchmark.csv', 'text/csv', use_container_width=True)
            valid = True
        except Exception as exc:
            st.error('Benchmark validation: ' + str(exc))
            valid = False
            bench = None
        uploaded = st.file_uploader('Import benchmark CSV (same two columns)', type=['csv'],
                                    key='kappa_csv_upload')
        if uploaded is not None and st.button('Apply uploaded KAPPA benchmark'):
            try:
                new_points = check_benchmark(pd.read_csv(uploaded))
                st.session_state['kappa_points'] = new_points
                st.session_state['kappa_calibration'] = None
                if 'kappa_edit_table' in st.session_state:
                    del st.session_state['kappa_edit_table']
                st.session_state['last_result'] = None
                st.rerun()
            except Exception as exc:
                st.error(f'Could not apply CSV: {exc}')
    with edit_right:
        st.markdown('**Comparison basis**')
        definition = st.radio('What do the KAPPA oil rates represent?',
                              [INCREMENTAL, POST],
                              index=0 if st.session_state['kappa_definition_saved'] == INCREMENTAL else 1,
                              help='Post-frac = full after-frac oil rate. Incremental = after-frac minus pre-frac.')
        st.session_state['kappa_definition_saved'] = definition
        baseline_mode = st.radio('Baseline PI model for permeability sweep',
                                  ['Mechanistic radial Darcy PI', 'Measured rate anchor'],
                                  help='Mechanistic mode changes baseline PI with permeability. Measured anchor uses '
                                       'the same fixed pre-frac oil rate at every permeability.')
        grid_res = st.select_slider('Benchmark grid resolution', options=[15,19,21,25,29], value=21)
        method = st.selectbox('Automatic match method', [LOCAL, GLOBAL],
                              index=0 if st.session_state['kappa_method_saved'] == LOCAL else 1,
                              help='PCHIP reproduces input targets in the 1-zone anchor scenario. Global multiplier '
                                   'can only reduce aggregate error, not necessarily fit all four points.')
        st.session_state['kappa_method_saved'] = method
        threshold = st.slider('Acceptable raw error (MAPE, %)', 1, 50, 15)
        st.caption('Inputs are taken from the **Most likely** column on Monte Carlo Studio. '
                   'The same fracture geometry, fluid properties and pressure differential are used at all k values.')
    if not valid:
        st.stop()
    if baseline_mode == 'Measured rate anchor':
        st.warning('A fixed measured q0 for all permeability points is rarely a valid k-sensitivity assumption. '
                   'The mechanistic PI mode is preferred when benchmarking permeability response.')
    if definition == INCREMENTAL:
        st.caption('Using your stated interpretation: KAPPA target = incremental rate Δq = post-frac minus pre-frac.')
    try:
        raw = reference_rates(st.session_state['params'], bench['Perm (mD)'].to_numpy(),
                              baseline_mode, grid_res)
        if definition == POST and (raw['Base rate (stb/d)'].to_numpy() >=
                                   bench['KAPPA Oil Rate (stb/d)'].to_numpy()).any():
            st.warning('At one or more points, benchmark post-frac oil is no higher than the model baseline. '
                       'That would imply zero/negative incremental oil. Check which rate definition is intended, '
                       'or revisit the assumed baseline/Pwf before applying a post-frac-rate calibration.')
        signature = reference_signature(st.session_state['params'], baseline_mode, grid_res)
        calibration, matched = fit_benchmark(bench, raw, definition, method, signature)
        raw_stats = performance(matched, 'Hyperion Raw (stb/d)')
        fit_stats = performance(matched, 'Matched (stb/d)')
        st.divider()
        x1,x2,x3,x4 = st.columns(4)
        x1.metric('Raw MAPE', f"{raw_stats['MAPE (%)']:.1f}%")
        x2.metric('Matched MAPE', f"{fit_stats['MAPE (%)']:.1f}%")
        x3.metric('Raw RMSE', f"{raw_stats['RMSE (stb/d)']:.1f} stb/d")
        x4.metric('Max raw error', f"{raw_stats['Max APE (%)']:.1f}%")
        if raw_stats['MAPE (%)'] <= threshold:
            st.success(f'Raw screening result is within the selected {threshold}% average-error tolerance '
                       'for these KAPPA points.')
        else:
            st.warning(f'Raw screening result exceeds {threshold}% MAPE tolerance. '
                       'Check zone net pay, ΔP, PVT, skin and frac geometry before relying on empirical matching.')
        st.markdown('**KAPPA vs Hyperion (before / after matching)**')
        st.dataframe(matched.style.format({
            'Perm (mD)':'{:,.2f}', 'KAPPA Oil Rate (stb/d)':'{:,.1f}',
            'Hyperion Raw (stb/d)':'{:,.1f}', 'Matched (stb/d)':'{:,.1f}',
            'Raw Error (%)':'{:+.1f}%', 'Matched Error (%)':'{:+.1f}%',
            'Applied Factor':'{:.3f}×'}), use_container_width=True, hide_index=True)
        k_grid = np.geomspace(float(bench['Perm (mD)'].min()), float(bench['Perm (mD)'].max()), 100)
        ref_grid = reference_rates(st.session_state['params'], k_grid, baseline_mode, grid_res)
        y_col = 'Post-frac rate (stb/d)' if definition == POST else 'Incremental rate (stb/d)'
        y_raw = ref_grid[y_col].to_numpy(float)
        y_matched = y_raw * calibration_factor(k_grid, calibration)
        left, right = st.columns([3,2], gap='medium')
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_grid, y=y_raw, name='Hyperion raw', mode='lines',
                                     line=dict(color='#64748B', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=k_grid, y=y_matched, name='Hyperion matched', mode='lines',
                                     line=dict(color='#00875A', width=3.5)))
            fig.add_trace(go.Scatter(x=bench['Perm (mD)'], y=bench['KAPPA Oil Rate (stb/d)'],
                                     name='KAPPA reference', mode='markers',
                                     marker=dict(color='#EA580C', size=12, symbol='diamond',
                                                 line=dict(color='#FFFFFF', width=1.5))))
            chart_style(fig, 'Single-Zone Oil Rate vs Permeability', 'Permeability (mD)',
                        'Oil rate (stb/d)', 430)
            fig.update_xaxes(type='log')
            st.plotly_chart(fig, use_container_width=True)
        with right:
            factors = calibration_factor(k_grid, calibration)
            fig2 = go.Figure(go.Scatter(x=k_grid, y=factors, mode='lines',
                                         line=dict(color='#2563EB', width=3), name='Applied match'))
            fig2.add_hline(y=1, line_color='#94A3B8', line_dash='dash')
            chart_style(fig2, 'Calibration Factor vs Permeability', 'Permeability (mD)',
                        'Rate correction factor (×)', 430)
            fig2.update_xaxes(type='log')
            st.plotly_chart(fig2, use_container_width=True)
        st.caption('The matched curve is an **empirical, in-sample** correction to the same rate quantity. '
                   'PCHIP exactly passes through the supplied KAPPA anchors by construction; this is not '
                   'independent validation and does not automatically resolve the high-permeability plateau. '
                   'Rates are not corrected outside the entered permeability range.')
        old = st.session_state.get('kappa_calibration')
        definition_matches = bool(old and old['definition'] == definition and
                                  old['signature'] == signature and
                                  old['method'] == method and
                                  np.array_equal(old['perms_md'], calibration['perms_md']) and
                                  np.allclose(old['factors'], calibration['factors'], rtol=1e-9))
        c_apply, c_remove = st.columns(2)
        with c_apply:
            if st.button('✓ Save match for Monte Carlo', type='primary', use_container_width=True):
                st.session_state['kappa_calibration'] = calibration
                st.session_state['last_result'] = None
                st.session_state['raw_result'] = None
                st.success('Match saved for this session. Go to Monte Carlo Studio, '
                           'select the same baseline mode/grid, enable calibration and run again.')
        with c_remove:
            if st.button('Remove active match', use_container_width=True):
                st.session_state['kappa_calibration'] = None
                st.session_state['last_result'] = None
                st.session_state['raw_result'] = None
                st.info('Saved match removed. Original physics is unchanged.')
        if definition_matches:
            st.caption('Currently saved match matches the benchmark settings shown above.')
        else:
            st.caption('The displayed match is a preview until you click **Save match for Monte Carlo**.')
        st.download_button('⬇ Export benchmark comparison (CSV)',
                           matched.to_csv(index=False).encode(), 'kappa_hyperion_comparison.csv',
                           'text/csv', use_container_width=True)
        with st.expander('Interpretation / required engineering QC'):
            st.markdown("""- Ensure KAPPA rates are all from **one fractured zone** at the same test/forecast time.
- Keep **Pwf, reservoir pressure, net pay, skin, Bo, viscosity and fracture design** aligned with KAPPA.
- The strong flattening at 50–100 mD may involve VLP, choke/facility constraints, or finite fracture conductivity: these are **hypotheses** to verify, not facts established by the four points.
- An empirical match may change incremental-oil and cumulative-oil forecasts. It does **not** calibrate decline, water cut, pressure support or fracture conductivity degradation.
- Validate the calibrated forecast against **independent** KAPPA runs or measured performance before using for decisions.
""")
    except Exception as exc:
        st.error(f'Benchmark comparison failed: {exc}')


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
