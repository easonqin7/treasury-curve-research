"""Offline checks for accounting, target timing and preservation of raw missingness."""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import treasury_rich_cheap_nss as model
from research_io import use_snapshot


class ResearchChecks(unittest.TestCase):
    def test_current_target_cannot_earn_current_price_move(self):
        idx = pd.date_range('2020-01-01', periods=4)
        yields = pd.DataFrame(4.0, index=idx, columns=model.USE_MATURITIES)
        yields.loc[idx[1]:, '10Y'] = 3.90
        positions = yields * 0.0
        positions.loc[idx[1], '10Y'] = 1.0
        with patch.object(model, 'estimate_trade_cost_bps', return_value=0.0):
            results, _ = model.backtest_strategy(yields, positions)
        # The only price move happens before the new target earns any returns.
        np.testing.assert_allclose(results['Gross Return'], 0.0)

    def test_held_long_benefits_from_lower_yields_and_pnl_reconciles(self):
        idx = pd.date_range('2020-01-01', periods=3)
        yields = pd.DataFrame(4.0, index=idx, columns=model.USE_MATURITIES)
        yields.loc[idx[1]:, '10Y'] = 3.90
        positions = yields * 0.0
        positions.loc[idx[0], '10Y'] = 1.0
        with patch.object(model, 'estimate_trade_cost_bps', return_value=1.0):
            results, _ = model.backtest_strategy(yields, positions)
        expected = 8 * .001 + .5 * .75 * .001**2
        self.assertAlmostEqual(results.loc[idx[1], 'Gross Return'], expected)
        self.assertGreater(results.loc[idx[1], 'Gross Return'], 0)
        np.testing.assert_allclose(results['Net Return'], results['Gross Return']-results['Transaction Cost'])
        np.testing.assert_allclose(results['Equity Curve'], (1+results['Net Return']).cumprod())

    def test_threshold_and_one_sided_exposure_are_explicit(self):
        z = pd.DataFrame(0., index=[0,1,2], columns=model.USE_MATURITIES)
        z.loc[0, '30Y'] = 2.0  # strict threshold, no entry
        z.loc[1, '30Y'] = 2.01
        z.loc[2, ['10Y','30Y']] = [2.1,-2.1]
        p = model.build_rich_cheap_positions(z, entry_z=2.0)
        self.assertEqual(p.loc[0].abs().sum(), 0)
        self.assertEqual(p.loc[1,'30Y'], 1.0)
        self.assertEqual(p.loc[2,'10Y'], .5)
        self.assertEqual(p.loc[2,'30Y'], -.5)
        self.assertNotEqual(p.loc[2].dot(pd.Series(model.DURATION_PROXY)), 0)

    def test_rolling_scores_do_not_change_when_future_data_change(self):
        residual = pd.DataFrame({'10Y': np.sin(np.arange(30)/3.)})
        changed = residual.copy()
        changed.iloc[20:] += 100
        a = model.compute_residual_zscores(residual, window=10)
        b = model.compute_residual_zscores(changed, window=10)
        pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])

    def test_snapshot_preserves_missing_rows_and_valid_unchanged_quotes(self):
        from types import SimpleNamespace
        namespace = SimpleNamespace()
        raw = use_snapshot(namespace, ROOT/'data/fred_yields_raw.csv')
        self.assertTrue(raw.loc['2016-07-04'].isna().all())
        requested = namespace.web.DataReader(['DGS10'], 'fred', '2016-07-01', '2016-07-05')
        self.assertTrue(pd.isna(requested.loc['2016-07-04','DGS10']))
        clean = raw.dropna(how='all')
        # Equal consecutive quotes are legitimate observations, not deleted holidays.
        self.assertTrue(clean.diff().eq(0).any().any())
        self.assertFalse(clean.isna().any().any())


if __name__ == '__main__':
    unittest.main()
