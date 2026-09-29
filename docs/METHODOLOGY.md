# Methodology

## 1. Inputs and units

The universe is `3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`, mapped respectively to FRED `DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30`.

The requested range is 2010-01-01 through 2026-08-31. The first usable baseline row is 2010-01-04. Raw observations are in percent: 4.00 to 4.01 is a one-basis-point increase. Calibration and return formulas convert yields to decimals; one basis point is then 0.0001.

The legacy baseline applies `ffill().dropna()` to reproduce the supplied study. The quote-date audit instead preserves the original missingness mask, removes only rows missing all ten maturities, and stops on partial missingness. A valid quote identical to yesterday's quote is retained.

## 2. Daily NSS curve

For maturity $m$, define

$$A(m,\tau)=\frac{1-e^{-m/\tau}}{m/\tau}.$$

The fitted yield is

$$\hat y_t(m)=\beta_{0,t}+\beta_{1,t}A(m,\tau_{1,t})+\beta_{2,t}[A(m,\tau_{1,t})-e^{-m/\tau_{1,t}}]+\beta_{3,t}[A(m,\tau_{2,t})-e^{-m/\tau_{2,t}}].$$

Each day's six parameters minimize unweighted squared yield errors across ten maturities using bounded L-BFGS-B. Bounds are 0.0001–0.20 for beta0, −0.30–0.30 for the other betas, and 0.05–30 for both decay constants. The preceding successful fit supplies the next initial guess (`maxiter=1000`, `ftol=1e-12`). No future day is used to warm-start an earlier fit.

The residual is $e_{t,m}=y_{t,m}-\hat y_t(m)$. This implementation fits an NSS shape directly to constant-maturity yields; it is not a bond cash-flow calibration of a zero-coupon discount curve. Optimizer success does not establish that the residual is economically meaningful or numerically stable.

## 3. Descriptive diagnostics

KMO evaluates the correlation structure of yield changes. PCA uses standardized full-sample daily yield changes to describe level, slope and curvature-like modes. The saved variance shares are approximately 68.68%, 17.17% and 7.57%. These diagnostics do not participate in trading, parameter selection or risk neutralization. Their full-sample estimation would be inappropriate if introduced as historical trading features without changing the estimation protocol.

## 4. Residual signal

For each maturity independently:

$$z_{t,m}=\frac{e_{t,m}-\operatorname{mean}_{W}(e_{t,m})}{\operatorname{std}_{W}(e_{t,m})}.$$

Rolling statistics include the current signal observation; minimum history is `W // 2`. The trading return uses a lagged target. Non-finite z-scores are omitted. A positive score means the residual is unusually high relative to its own recent history; it does not necessarily mean the absolute residual is positive.

Candidate windows are 126, 252 and 378 observations, approximately half, one and one-and-a-half trading years on a quote-date calendar. They determine residual normalization, not holding duration.

## 5. Selection and positions

Only the training period ending 2017-12-31 ranks the nine combinations of windows and thresholds 1.5, 2.0 and 2.5. All candidates start scoring after the largest 378-observation warm-up, on 2011-06-16 in the legacy calendar. The highest finite net-return training Sharpe wins; exact ties use smaller window, then smaller threshold. The supplied selection is W=252 and entry=2.0.

On every date:

1. Select up to two highest z-scores strictly above the threshold as longs.
2. Select up to two lowest z-scores strictly below its negative as shorts.
3. Weight selected buckets inversely to their fixed duration proxy within each side.
4. Allocate half the gross budget to each side if both exist; allocate the entire gross budget to the sole active side otherwise. Remain flat if neither exists.

Targets are recomputed daily. A bucket exits when it no longer meets the threshold/ranking rules; there is no independent exit threshold, stop-loss or fixed holding period. A one-bucket side normalizes to its entire side allocation. This explains why inverse-duration weighting does not cap aggregate duration or create DV01 neutrality.

The frozen strategy is evaluated from 2018 onward. The 2018–2021 and 2022–2026 tables are descriptive subperiods, not separate selections. Positions carry across period boundaries.

## 6. Return and cost accounting

For each maturity:

$$r^{proxy}_{t,m}=-D_m\Delta y_{t,m}+\tfrac12 C_m(\Delta y_{t,m})^2.$$

Portfolio gross return is $\sum_m w_{t-1,m}r^{proxy}_{t,m}$. The baseline therefore assumes the target formed using date $t-1$ information earns the entire next quote-to-quote move. Data publication and executable prices are not verified.

| Bucket | 3M | 6M | 1Y | 2Y | 3Y | 5Y | 7Y | 10Y | 20Y | 30Y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Duration proxy | .25 | .50 | .95 | 1.8 | 2.7 | 4.5 | 6.2 | 8 | 15 | 20 |
| Convexity coefficient | .001 | .003 | .01 | .05 | .10 | .25 | .45 | .75 | 2.5 | 4.5 |

These are illustrative fixed coefficients, not calibrated security analytics. In particular, the convexity coefficients are not established as conventionally scaled bond convexities and contribute very little in this implementation.

Turnover is $|w_t-w_{t-1}|$ per bucket. Cost in return units is turnover multiplied by estimated one-way cost in basis points divided by 10,000. The cost schedule combines a base (0.5 bp for `future`, 3 bp for `cash`) with a square-root impact term using assumed ADV, $10 million capital and 60-observation yield-change volatility. The volatility estimate includes the current quote observation.

Costs are booked on the target-change date while that date's gross return comes from the preceding target. This is an end-of-period rebalance convention, not a validated execution simulation. The helper reads the module-level `INSTRUMENT`; the backtest's `instrument` argument alone does not switch the cost schedule. Keep module configuration consistent when reproducing this study.

## 7. Performance and benchmarks

Net returns compound into wealth starting at one. CAGR uses elapsed calendar time from the observation before the first scored return. Drawdown includes an initial wealth of one. Baseline volatility uses daily sample standard deviation times sqrt(252); Sharpe uses annualized mean divided by that volatility, without subtracting a risk-free return.

The quote-calendar audit uses each period's actual observations per elapsed year for volatility and Sharpe. Its common training scoring start is 2011-07-06. Do not compare differently annualized Sharpe values without that qualification.

Benchmarks are long 10Y, long 20Y and equal-weight bucket price-return proxies from the same yield-change formula. They omit costs, carry and financing, and have different exposures from the strategy. Relative outperformance is not proof of alpha.

## 8. Robustness workflow

The extreme-day audit reconciles gross PnL to linear duration and convexity contributions. It displays the same original extreme dates under a delayed target sequence; the delayed variant is not allowed to choose a new set of favorable dates for that comparison.

The calendar audit compares A (legacy), B (valid quote dates) and C (valid quote dates plus one extra observation of target delay). All entry, rebalancing and exit targets are delayed in C. It refits NSS and recalculates rolling statistics; it does not reselect W or the threshold. These are retrospective diagnostics on already-inspected data, not independent fresh validation.
