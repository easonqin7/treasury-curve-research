# Limitations and research status

This project demonstrates a research process. It does not establish a production-ready trading edge.

## Market representation

- **Yield buckets are not tradable instruments.** Constant-maturity yields are curve statistics. No CUSIPs, bid/ask bond quotes, Treasury futures contracts or actual execution prices are used. The `future` label only changes an illustrative cost schedule.
- **Returns are incomplete price proxies.** Carry, coupon cash flows, roll-down, financing, short borrow costs, futures rolls, delivery options, margin and cash collateral returns are not modeled. Duration and convexity are fixed illustrative coefficients; convexity scaling is not validated against actual bonds.
- **The source curve is already constructed.** Fitting NSS to published constant-maturity yields can detect differences between curve representations rather than exploitable security mispricing. This is not an arbitrage-free discount-curve model.

## Signal and risk

- **Residual convergence is not the same as profitable price convergence.** A residual can shrink because the fitted reference curve changes. The current analysis has not isolated that mechanism.
- **Directional rate risk remains.** Both sides need not be present. A sole long or short bucket can receive 100% target weight, and equal dollar long/short allocations can still have substantial net duration. No DV01 neutrality is claimed.
- **NSS numerical stability is unresolved.** Six nonlinear parameters are estimated from ten points with one warm start. A successful optimizer result is not a uniqueness or stability guarantee. Removing stale rows changes the optimization path as well as the rolling signal history.
- **Threshold decisions are discontinuous.** Small score changes near the entry threshold or ranking boundary can create large target changes. Initialization/threshold-crossing stability is a planned investigation, not a completed result.

## Execution and costs

- **The execution timeline is assumed.** Lagging weights avoids using the current return directly in its own target, but does not prove that all information was available before the assumed fill. FRED daily quotes do not establish publication-to-trade timing.
- **An extra observation is not always one calendar day.** The legacy calendar includes forward-filled no-quote dates. The quote-date audit addresses this distinction but is not an exchange-calendar or intraday fill model.
- **The delay experiment moves exits too.** It cannot identify entry-signal lifetime in isolation. Better delayed performance in one period does not justify selecting that delay after seeing the test.
- **Costs are hypothetical.** ADV, base costs and the impact coefficient are assumptions. Yield-change volatility is fed into a price-return cost model without an instrument-calibrated conversion. Gross-versus-net differences are model scenarios, not measured execution costs.

## Statistical interpretation

- **The test has been viewed repeatedly.** The initial chronological split separates parameter selection from evaluation, but later diagnostics were designed with knowledge of test results. It is a retrospective test, not an untouched holdout for every revision.
- **No uncertainty estimate establishes significance.** Sharpe confidence intervals, autocorrelation-adjusted inference, multiple-testing corrections and independent prospective validation are absent. Annualization uses conventional formulas; serial dependence may affect their interpretation.
- **Tail days matter.** Large gains and losses frequently arise from single-sided long-duration exposures. Removing the best days is a sensitivity description, not an implementable filter. Positive average returns do not imply a high-probability or persistent signal.
- **Benchmarks are not risk matched.** The passive price proxies have different duration and participation. Sharpe is measured against zero rather than a cash return. Neither benchmark outperformance nor a positive Sharpe proves alpha.
- **Data can be revised.** The committed raw snapshot makes this version inspectable but is not an as-of historical vintage database.

## What is complete

Daily curve fitting; training-only selection; frozen-rule historical evaluation; approximate turnover costs; baseline benchmark comparisons; extreme-day PnL reconstruction; missing-date audit; and timing sensitivity on two calendars.

## What is deliberately not claimed

A deployable futures strategy, market neutrality, confirmed arbitrage, optimal holding duration, statistically established excess returns, or proof that one timing variant should be traded.

## Next research step

Hold valid quote dates fixed and compare legacy versus refitted NSS residuals, fitted-yield differences in basis points, and threshold/ranking changes. Then examine alternative initializations using training data and explicit numerical diagnostics. This should precede another search for better backtest parameters.
