# Reproduction and evidence

## Environment

The reference run used **Python 3.12.7**. Install `requirements.txt` for supported dependency ranges, or `requirements-tested.txt` for the exact direct-package versions recorded in the verification environment. The latter is a reference record, not a complete transitive dependency lock. Numerical optimization can vary with library and platform versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/run_research.py
```

On Windows use `.venv\Scripts\activate` instead of the shell activation command.

## Data snapshot

`data/fred_yields_raw.csv` contains raw observations for the ten FRED series, including missing rows. `data/fred_yields_raw.metadata.json` records the retrieval time, requested dates, series IDs and SHA-256 hash. The runner defaults to this snapshot and performs no network request. Both baseline and calendar audit see the same raw frame through a module-local reader adapter.

The raw range begins on 2010-01-01; the usable legacy range begins on 2010-01-04. The source notebook and this run end on 2026-08-31. The snapshot is a retrieved historical dataset, not an as-of vintage database.

To deliberately obtain a separate updated snapshot:

```bash
python src/download_data.py --output data/fred_yields_refreshed.csv
python src/run_research.py --snapshot data/fred_yields_refreshed.csv
```

The downloader refuses to overwrite an existing file. `--live` explicitly bypasses the stored snapshot; the baseline/calendar equality check may stop if their independent requests receive different data. New data or libraries may change results; record them as a new experiment rather than overwriting the research conclusion silently. The runner writes generated reports to the existing `reports/` paths, so commit or copy a prior run before reproducing with altered inputs.

## Notebook

```bash
python -m pip install jupyterlab
jupyter lab notebooks/treasury_nss_research.ipynb
```

Run all cells from top to bottom. The import cell locates the project root and uses the versioned snapshot. `output`, `oos_output`, `timing_audit` and `calendar_audit` contain intermediate evidence. Core functions live in one module rather than being duplicated in the notebook.

For a noninteractive notebook execution:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/treasury_nss_research.ipynb
```

This replaces saved notebook outputs. Headless script execution is the simpler route to exported CSV/PNG reports.

## Outputs and units

| Artifact | Meaning |
|---|---|
| `reports/figures/` | Baseline comparison, test-only chart, two timing/calendar comparisons, PCA and NSS snapshot |
| `training_parameter_grid.csv` | All nine training candidates; decimal return/volatility fields |
| `baseline_period_performance.csv` | Baseline training, full test and test subperiods; decimal return/volatility fields |
| `baseline_daily_results.csv` | Decimal daily gross/net returns and costs, turnover, wealth |
| `baseline_target_positions.csv` | End-of-observation target weights; returns use a lag |
| `calendar_summary.csv` | A/B/C metrics; fields labeled `(%)` are already multiplied by 100 |
| `calendar_variant_*_daily_results.csv` | Daily results underlying the calendar comparison figure |
| `extreme_maturity_details.csv` | Yield changes in bp; PnL contributions in percentage points |
| `execution_timeline.csv` | Return interval and signal-date alignment |
| `reports/source_tables/` | Display-precision tables extracted from the supplied notebook for provenance |
| `reports/run_summary.txt` | Console output from the verified script run; CSVs retain full tables |
| `reports/reproduction_status.json` | Last successful headless run and snapshot identity |

All net returns subtract only the modeled transaction cost. The final year's return is partial, not an annualized full-year observation.

## Validation boundaries

The English module was checked against the supplied notebook's retained functions: after neutralizing text constants, the abstract syntax trees match. Chinese comments, messages and table labels were translated; numerical expressions and control flow were preserved. Unused historical helper functions were omitted from the maintained module.

The full headless workflow was run from the snapshot. Calendar-comparison and legacy-delay metrics were checked against the supplied saved tables at their four-decimal display precision. The notebook execution and offline test outcomes are recorded in `reports/validation.json`.

Offline tests cover lagged-target accounting, return/cost reconciliation, the strict threshold and one-sided exposure rules, causal rolling scores, and preservation of raw missing observations. These are implementation checks, not proof of statistical significance or economic validity. No live orders, execution connectivity or prospective trading validation is included.
