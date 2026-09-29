"""Snapshot loading and evidence export; no trading-rule changes."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = {
    '3M': 'DGS3MO', '6M': 'DGS6MO', '1Y': 'DGS1', '2Y': 'DGS2',
    '3Y': 'DGS3', '5Y': 'DGS5', '7Y': 'DGS7', '10Y': 'DGS10',
    '20Y': 'DGS20', '30Y': 'DGS30',
}

def use_snapshot(module, path):
    """Provide the same immutable raw frame to both legacy and calendar workflows.

    Replace only the module-local reader objects, not pandas_datareader globally.
    The original baseline ffill and the calendar audit's missingness checks still run.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError('Snapshot dates must be unique and sorted.')
    if set(raw.columns) != set(TICKERS.values()):
        raise ValueError('Snapshot must contain the ten configured FRED series IDs.')
    def reader(series, source, start, end):
        if source != 'fred':
            raise ValueError('Only the FRED snapshot is supported.')
        requested = [series] if isinstance(series, str) else list(series)
        return raw.loc[pd.Timestamp(start):pd.Timestamp(end), requested].copy()
    module.web = SimpleNamespace(DataReader=reader)
    module.fred_reader = SimpleNamespace(DataReader=reader)
    return raw


def export_results(root, output, oos, timing, calendar):
    """Persist exact frames underlying the reports; returns are decimal unless named (%)."""
    tables = Path(root) / 'reports' / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    frames = {
        'training_parameter_grid': oos['train_grid'],
        'baseline_period_performance': oos['summary'],
        'baseline_annual_returns': oos['test_annual_returns'],
        'benchmark_comparison': oos['benchmark_summary'],
        'baseline_daily_results': oos['frozen_results'],
        'baseline_target_positions': oos['frozen_positions'],
        'baseline_comparison_returns': oos['comparison_returns'],
        'extreme_days': timing['extreme_days'],
        'extreme_maturity_details': timing['extreme_details'],
        'legacy_delay_summary': timing['delay_summary'],
        'paired_extreme_days': timing['paired_extreme_days'],
        'calendar_summary': calendar['summary'],
        'no_quote_rebalancing': calendar['no_quote_audit'],
        'execution_timeline': calendar['timing'],
        'quote_date_yields': calendar['quote_yields'],
        'quote_date_residuals': calendar['residuals'],
        'quote_date_positions': calendar['quote_positions'],
    }
    for name, frame in frames.items():
        frame.to_csv(tables / f'{name}.csv', index=True)
    for name, frame in calendar['results'].items():
        variant = name.split()[0].lower()
        frame.to_csv(tables / f'calendar_variant_{variant}_daily_results.csv')
    (tables / 'selected_parameters.json').write_text(json.dumps(oos['selected_params'], indent=2)+'\n')


def plot_calendar_comparison(calendar, path):
    """Same-period equity comparison; every period starts at one before its first return."""
    periods = {
        'Training': (calendar['common_train_start'], pd.Timestamp('2017-12-31')),
        'Historical test': (pd.Timestamp('2018-01-01'), calendar['quote_yields'].index.max()),
    }
    colors = ['#666666', '#2463A6', '#BC6C25']
    styles = ['--', '-', '-.']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, (period, (start, end)) in zip(axes, periods.items()):
        for (name, result), color, style in zip(calendar['results'].items(), colors, styles):
            r = result.loc[start:end, 'Net Return']
            origin = result.index[result.index.get_loc(r.index[0])-1]
            curve = pd.concat([pd.Series([1.0], index=[origin]), (1+r).cumprod()])
            ax.plot(curve.index, curve, label=name, color=color, linestyle=style, linewidth=1.6)
        ax.axhline(1, color='#999999', linewidth=.7)
        ax.set_title(period)
        ax.set_ylabel('Growth of $1 (net price-return proxy)')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Quote-calendar and execution sensitivity | fixed W=252, entry=2.0')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.show()
    return fig
