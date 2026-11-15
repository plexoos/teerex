# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25.2", "numpy", "pandas", "matplotlib", "plotly"]
# ///
import marimo

__generated_with = '0.25.2'
app = marimo.App(width='full', app_title='Measured multithreaded Geant4 timelines')


@app.cell(hide_code=True)
def _():
    import os
    from pathlib import Path
    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from teerex.analysis import read_one_csv
    root = Path(__file__).resolve().parents[1]
    directory = Path(os.environ.get('TEEREX_MT_DATA_DIR', str(root/'docs/mt-study'))).expanduser()
    labels = {
        'pfRICH · 10 primaries':'pfrich_n10', 'pfRICH · 100 primaries':'pfrich_n100',
        'dRICH · 10 primaries':'drich_n10', 'dRICH · 100 primaries':'drich_n100',
        'DUNE mock · balanced':'argon_balanced', 'DUNE mock · 10 × 10 MeV':'argon_gpu_n10',
        'DUNE mock · 100 × 1 MeV':'argon_gpu_n100',
    }
    return directory, go, labels, make_subplots, mo, pd, read_one_csv


@app.cell(hide_code=True)
def _(directory, mo, read_one_csv, pd):
    try:
        events = read_one_csv(directory/'events-raw.csv.gz')
        runs = pd.read_csv(directory/'run-summary.csv')
        summary = pd.read_csv(directory/'summary.csv')
        sensitivity = pd.read_csv(directory/'sensitivity.csv')
    except (OSError, ValueError) as _error:
        mo.stop(True, mo.callout(
            f'Cannot load the MT evidence bundle: {_error}. '
            'Set TEEREX_MT_DATA_DIR to a compatible bundle directory (see docs/mt-study/README.md).',
            kind='warn'))
    return events, runs, sensitivity, summary


@app.cell(hide_code=True)
def _(events, mo, runs):
    mo.md(f'''
    # Measured multithreaded Geant4 CPU/GPU timelines

    **Measured dataset:** {len(runs)} process runs, {len(events):,} raw events,
    {int(runs.analysis_events.sum()):,} retained events, and {runs.replication.nunique()} seed pairs.
    Timelines show every recorded event, starting at event 0, including the 16 initial warm-up events.
    The scaling summaries and duration distributions below use the 500 retained events per run.

    These measurements repeat seven electron workloads at **1, 2, 4, and 8 workers**.
    RICH primaries use **20 GeV/c**. The DUNE mock liquid-argon workloads use 65,536 electrons
    at 1 keV, 10 electrons at 10 MeV, or 100 electrons at 1 MeV kinetic energy.
    Workers track charged particles concurrently, then wait for a shared GPU gate that serves
    events in event-ID order. Every process uses one GPU.

    Blue shows CPU generation; teal shows CPU cleanup. Gray shows queue waiting, and pale gray
    shows the GPU call, which includes host work and device waiting. Orange shows host-observed GPU service;
    purple shows cleanup while the GPU gate remains held. Queue waiting is separate from CPU phases.
    Host service includes staging, transfers, and synchronization; its occupancy is not GPU SM utilization.
    ''')
    return


@app.cell(hide_code=True)
def _(labels, mo, runs):
    workload = mo.ui.dropdown(labels, value='dRICH · 100 primaries', label='Workload')
    replication = mo.ui.dropdown({f'Seed pair {int(x)+1}':int(x) for x in sorted(runs.replication.unique())},
                                 value='Seed pair 1', label='Replication')
    mo.hstack([workload,replication],justify='start',gap=2)
    return replication, workload


@app.cell(hide_code=True)
def _(events, go, make_subplots, mo, replication, workload):
    chosen = events.loc[events.workload.eq(workload.value)&events.replication.eq(replication.value)]
    _threads = sorted(chosen.num_threads.unique())
    timeline = make_subplots(rows=len(_threads),cols=1,vertical_spacing=.07,
                             subplot_titles=[f'{n} Geant4 worker'+('' if n==1 else 's') for n in _threads])
    for _index,_n in enumerate(_threads,start=1):
        _run=chosen.loc[chosen.num_threads.eq(_n)].sort_values('event_id')
        mo.stop(_run.run_id.nunique()!=1,mo.callout('A timeline must contain one process only.',kind='danger'))
        _lo=float(_run.cpu_pre_start_time_offset_s.min())
        _hi=float(_run.end_offset_s.max())
        _workers=[f'Worker {w}' if w>=0 else 'Serial CPU' for w in _run.worker_id]
        _phases=[
            ('cpu_pre_start_time_offset_s','cpu_pre_end_time_offset_s','CPU generation','#1f77b4',False),
            ('gpu_submit_time_offset_s','gpu_start_time_offset_s','GPU queue wait','#aeb8c4',False),
            ('gpu_start_time_offset_s','gpu_end_time_offset_s','GPU call: host work + device wait','#e0e5eb',False),
            ('cpu_post_start_time_offset_s','cpu_post_end_time_offset_s','CPU cleanup','#159a9c',False),
            ('gpu_start_time_offset_s','gpu_end_time_offset_s','GPU host service','#f28e2b',True),
            ('gpu_end_time_offset_s','gpu_release_time_offset_s','Gate held for cleanup','#9467bd',True),
        ]
        for _a,_b,_name,_color,_gpu in _phases:
            _starts=(_run[_a]-_lo)*1000
            _ends=(_run[_b]-_lo)*1000
            _durations=(_run[_b]-_run[_a])*1000
            _lanes=['GPU gate']*len(_run) if _gpu else _workers
            timeline.add_trace(go.Bar(
                orientation='h',x=_durations.tolist(),base=_starts.tolist(),
                y=_lanes,name=_name,marker_color=_color,width=.65,legendgroup=_name,showlegend=_index==1,
                customdata=list(zip(_run.event_id,_starts,_ends,_workers,_durations)),
                hovertemplate='Event %{customdata[0]} · %{customdata[3]}<br>'
                              'Start: %{customdata[1]:.6f} ms<br>End: %{customdata[2]:.6f} ms<br>'
                              'Duration: %{customdata[4]:.6f} ms<extra>%{fullData.name}</extra>'),
                row=_index,col=1)
        _lanes=['GPU gate']+(['Serial CPU'] if _n==1 else [f'Worker {w}' for w in range(_n)])
        timeline.update_yaxes(categoryorder='array',categoryarray=_lanes,fixedrange=True,row=_index,col=1)
        timeline.update_xaxes(title_text='Elapsed from first recorded CPU start in this process (ms)' if _index==len(_threads) else '',range=[0,(_hi-_lo)*1000],row=_index,col=1)
    timeline.update_layout(template='plotly_white',height=1100,barmode='overlay',dragmode='zoom',
                           legend=dict(orientation='h',y=1.08),margin=dict(l=80,r=30,t=100,b=50))
    mo.vstack([mo.md('''## Worker timelines

    Each panel shows all 516 recorded events (IDs 0–515), including warm-up and the final cleanup.
    Zero is the earliest recorded CPU-generation start in that process; initialization before that
    timestamp is outside the recorded event timeline. Each process has its own clock and time range.
    No events or phase durations are cut. Drag horizontally to zoom into startup or later events;
    double-click to restore the full run. Hover for event ID, worker, start, end, and full duration.
    '''),timeline])
    return


@app.cell(hide_code=True)
def _(go, mo, runs, workload):
    _data=runs.loc[runs.workload.eq(workload.value)]
    _flagged=int(_data.cleanup_stall_events.sum())
    _warning=mo.callout(f'{_flagged} retained events have cleanup above 50 ms with thread CPU below 5% of wall time. They remain in the statistics; a delayed serial control can inflate apparent speedup.',kind='warn') if _flagged else mo.md('')
    scaling=go.Figure()
    for _rep,_group in _data.groupby('replication'):
        _group=_group.sort_values('num_threads')
        scaling.add_scatter(x=_group.num_threads,y=_group.speedup,mode='lines+markers',name=f'Seed pair {int(_rep)+1}',
                            line=dict(color=['#0072B2','#D55E00','#009E73','#CC79A7'][int(_rep)]))
    scaling.add_scatter(x=[1,8],y=[1,8],mode='lines',line=dict(dash='dash',color='#aaaaaa'),name='Ideal linear scaling')
    scaling.update_layout(template='plotly_white',height=400,xaxis=dict(title='Geant4 workers',tickvals=[1,2,4,8]),
                          yaxis_title='Throughput / matched serial throughput')
    mo.vstack([mo.md('## Scaling across independent seed runs\n\nEvery point uses completion throughput and the same seed’s fresh serial control. Four seed pairs are four replication units; thousands of events do not create thousands of independent replications.'),_warning,scaling,
               mo.ui.table(_data.round(4),selection=None,show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    histogram_threads=mo.ui.dropdown({f'{n} workers':n for n in (1,2,4,8)},value='4 workers',label='Duration distributions')
    histogram_threads
    return (histogram_threads,)


@app.cell(hide_code=True)
def _(events, go, histogram_threads, make_subplots, mo, workload):
    import numpy as np
    _data=events.loc[events.workload.eq(workload.value)&events.num_threads.eq(histogram_threads.value)&events.event_id.ge(16)]
    distribution=make_subplots(rows=1,cols=2,subplot_titles=['CPU generation + cleanup','GPU host service'])
    for _column,_values in enumerate([_data.cpu_pre_runtime_s+_data.cpu_post_runtime_s,_data.gpu_runtime_s],start=1):
        _positive=_values[_values.gt(0)]*1000
        _bins=np.geomspace(_positive.min()*.99,_positive.max()*1.01,41)
        for _rep in sorted(_data.replication.unique()):
            _subset=_values[_data.replication.eq(_rep)]*1000
            _counts,_=np.histogram(_subset,bins=_bins)
            distribution.add_trace(go.Scatter(x=np.sqrt(_bins[:-1]*_bins[1:]),y=_counts/len(_subset)*100,
                mode='lines',name=f'Seed pair {int(_rep)+1}',legendgroup=str(_rep),showlegend=_column==1,
                line=dict(color=['#0072B2','#D55E00','#009E73','#CC79A7'][int(_rep)])),row=1,col=_column)
        distribution.update_xaxes(type='log',title_text='Phase duration (ms, logarithmic)',row=1,col=_column)
        distribution.update_yaxes(title_text='Retained events per bin (%)',row=1,col=_column)
    distribution.update_layout(template='plotly_white',height=380,margin=dict(t=60,b=60))
    mo.vstack([mo.md('## Duration distributions\n\nAll retained events and all seed pairs contribute. Bin edges span the full positive duration range; no slow events are removed. These phase durations are separate from throughput and occupancy.'),distribution])
    return


@app.cell(hide_code=True)
def _(mo, sensitivity, summary, workload):
    mo.vstack([
        mo.md('## All workloads\n\nOccupancy is an interval union clipped to the same completion window. Ranges describe run-to-run variation, not confidence intervals.'),
        mo.ui.table(summary.round(4),selection=None,show_column_summaries=False),
        mo.md('## Warm-up and drain sensitivity\n\nCompare cutoffs of 8, 16, and 32 initial events and exclusions of 0, 8, and 16 final events. Raw warm-up history is preserved.'),
        mo.ui.table(sensitivity.loc[sensitivity.workload.eq(workload.value),
                    ['num_threads','replication','warmup','tail','completion_events','events_per_second','gpu_gate_occupancy']].round(4),
                    selection=None,show_column_summaries=False),
        mo.md('''## Interpretation limits

        Serial and MT Geant4 use different RNG initialization paths. Matching seed labels do not promise
        identical serial/MT event histories. Retained MT histories are checked across worker counts.
        The machine is shared, and the earlier serial study used the other RTX 4090; use this campaign’s
        fresh serial controls for scaling comparisons. CPU wall phases can include filesystem waits or descheduling;
        event-thread CPU time for generation plus cleanup is reported separately; it excludes host
        work inside the GPU call. Do not sum overlapping process-CPU samples.

        These are measured synchronous workers sharing an ordered GPU gate. They do not measure
        asynchronous dispatch or concurrent GPU kernels, and no Bayesian policy is fitted here.
        ''')])
    return


if __name__ == '__main__':
    app.run()
