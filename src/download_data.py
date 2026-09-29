"""Explicitly fetch the raw, unfilled FRED data; do not overwrite a snapshot silently."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
from pandas_datareader import data as reader
from research_io import TICKERS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'data/fred_yields_raw.csv')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists. Choose another --output path to retain the original snapshot.')
    raw = reader.DataReader(list(TICKERS.values()), 'fred', '2010-01-01', '2026-08-31')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output, index_label='DATE')
    metadata = {
        'source': 'Federal Reserve Bank of St. Louis FRED; underlying Treasury constant-maturity series',
        'series': TICKERS, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'requested_start': '2010-01-01', 'requested_end': '2026-08-31',
        'actual_start': str(raw.index.min().date()), 'actual_end': str(raw.index.max().date()),
        'rows': len(raw), 'all_missing_rows': int(raw.isna().all(axis=1).sum()),
        'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'processing': 'None: missing observations remain empty. No forward fill.',
    }
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
