"""Run the supplied research workflow and export tables/figures from the repo root."""
import argparse
from pathlib import Path
import json
import hashlib
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import treasury_rich_cheap_nss as research
from research_io import use_snapshot, export_results, plot_calendar_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='Raw FRED CSV; defaults to data/fred_yields_raw.csv')
    parser.add_argument('--live', action='store_true', help='Explicitly use live FRED requests instead of the snapshot')
    args = parser.parse_args()
    if args.live and args.snapshot:
        parser.error('Choose either --live or --snapshot.')
    root = Path(__file__).resolve().parents[1]
    snapshot = args.snapshot or root / 'data' / 'fred_yields_raw.csv'
    if not args.live:
        if not snapshot.exists():
            parser.error('Snapshot not found. Run src/download_data.py first, or use --live.')
        use_snapshot(research, snapshot)
    figures = root / 'reports' / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    names = iter(['pca_loadings.png', 'nss_curve_snapshot.png', 'selection_and_test.png',
                  'test_vs_benchmarks.png', 'legacy_execution_delay.png'])
    original_show = plt.show
    def save_show(*args, **kwargs):
        name = next(names)
        plt.gcf().savefig(figures / name, dpi=160, bbox_inches='tight')
        plt.close(plt.gcf())
    plt.show = save_show
    log = root / 'reports' / 'run_summary.txt'
    try:
        with log.open('w') as stream, contextlib.redirect_stdout(stream):
            output = research.main()
            oos = research.run_oos_test(output)
            timing = research.audit_extreme_days_and_delay(output, oos)
            calendar = research.audit_quote_dates_and_execution(output, oos)
            export_results(root, output, oos, timing, calendar)
            plt.show = lambda *a, **k: None
            plot_calendar_comparison(calendar, figures / 'quote_calendar_comparison.png')
        status = {
            'status': 'completed',
            'data_mode': 'live' if args.live else 'snapshot',
            'snapshot_sha256': None if args.live else hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            'selected_parameters': oos['selected_params'],
            'data_start': str(output['yields'].index.min().date()),
            'data_end': str(output['yields'].index.max().date()),
        }
        (root / 'reports' / 'reproduction_status.json').write_text(json.dumps(status, indent=2)+'\n')
    finally:
        plt.show = original_show
        plt.close('all')
    print('Completed. See reports/run_summary.txt, reports/tables/ and reports/figures/.')


if __name__ == '__main__':
    main()
