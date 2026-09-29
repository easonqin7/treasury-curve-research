"""Treasury NSS research functions, translated from the supplied research notebook.

No workflow runs on import. See docs/METHODOLOGY.md for model assumptions.
The instrument selector controls an illustrative cost schedule, not futures data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web

from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Configuration
# ============================================================

START_DATE = dt.datetime(2010, 1, 1)
END_DATE = dt.datetime(2026, 8, 31)

ZSCORE_WINDOW = 252
ENTRY_Z = 2.0
GROSS_LEVERAGE = 1.0
TOP_N = 2

# ------------------------------------------------------------
# Transaction cost / capital / market impact
# ------------------------------------------------------------


CAPITAL = 10_000_000


INSTRUMENT = "future"


BASE_COST_BPS = {
    "future": 0.5,
    "cash":   3.0,
}


# impact_bps = K * sigma_daily_bps * sqrt(notional / ADV)
IMPACT_K = 1.0
USE_MARKET_IMPACT = True





ADV_USD = {
    "3M":  20e9,
    "6M":  15e9,
    "1Y":  25e9,
    "2Y":  60e9,
    "3Y":  40e9,
    "5Y":  80e9,
    "7Y":  35e9,
    "10Y": 100e9,
    "20Y": 25e9,
    "30Y": 30e9,
}

USE_MATURITIES = [
    "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"
]

MATURITY_YEARS = {
    "3M": 0.25, "6M": 0.50, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}

DURATION_PROXY = {
    "3M": 0.25, "6M": 0.50, "1Y": 0.95, "2Y": 1.8, "3Y": 2.7,
    "5Y": 4.5, "7Y": 6.2, "10Y": 8.0, "20Y": 15.0, "30Y": 20.0,
}

CONVEXITY_PROXY = {
    "3M": 0.001, "6M": 0.003, "1Y": 0.01, "2Y": 0.05, "3Y": 0.10,
    "5Y": 0.25, "7Y": 0.45, "10Y": 0.75, "20Y": 2.50, "30Y": 4.50,
}


# ============================================================
# 2. Data Fetching from FRED
# ============================================================

from IPython.display import display
from pandas_datareader import data as fred_reader
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter


def fetch_fred_yields(start_date, end_date):
    fred_tickers = {
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "7Y": "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30",
    }

    tickers = [fred_tickers[m] for m in USE_MATURITIES]

    data = web.DataReader(tickers, "fred", start_date, end_date)
    data.columns = USE_MATURITIES

    # Keep business-day observations where all maturities exist.
    data = data.ffill().dropna()

    return data


def nss_yield(maturity, beta0, beta1, beta2, tau1, beta3, tau2):
    """
    Nelson-Siegel-Svensson yield curve.

    y(m) = beta0
         + beta1 * A(m, tau1)
         + beta2 * [A(m, tau1) - exp(-m/tau1)]
         + beta3 * [A(m, tau2) - exp(-m/tau2)]

    where A(m,tau) = (1 - exp(-m/tau)) / (m/tau)
    """
    m = np.asarray(maturity, dtype=float)

    tau1 = max(tau1, 1e-6)
    tau2 = max(tau2, 1e-6)

    x1 = m / tau1
    x2 = m / tau2

    A1 = (1 - np.exp(-x1)) / x1
    A2 = (1 - np.exp(-x2)) / x2

    term1 = beta0
    term2 = beta1 * A1
    term3 = beta2 * (A1 - np.exp(-x1))
    term4 = beta3 * (A2 - np.exp(-x2))

    return term1 + term2 + term3 + term4


def nss_sse(params, maturities, market_yields):
    fitted = nss_yield(maturities, *params)
    residuals = fitted - market_yields
    return np.sum(residuals ** 2)


def calibrate_nss_single_day(maturities, market_yields, initial_guess):
    """
    Calibrate NSS parameters for one date.
    market_yields should be in decimal form, e.g. 0.045.
    """

    bounds = [
        (0.0001, 0.20),    # beta0
        (-0.30, 0.30),     # beta1
        (-0.30, 0.30),     # beta2
        (0.05, 30.0),      # tau1
        (-0.30, 0.30),     # beta3
        (0.05, 30.0),      # tau2
    ]

    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    initial_guess = np.clip(np.asarray(initial_guess), lower, upper)

    result = minimize(
        nss_sse,
        initial_guess,
        args=(maturities, market_yields),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        return result.x, True, result.fun

    return np.full(6, np.nan), False, np.nan


def calibrate_nss_history(yields_percent):
    """
    Fit NSS curve every day.

    Returns:
        params_df
        fitted_df
        residual_df
    """

    yields_decimal = yields_percent / 100.0
    maturities = np.array([MATURITY_YEARS[m] for m in USE_MATURITIES], dtype=float)

    params_history = []
    fitted_history = []
    residual_history = []

    first_curve = yields_decimal.iloc[0].values

    initial_guess = np.array([
        first_curve[-1],               # beta0, long-end level
        first_curve[0] - first_curve[-1],  # beta1, short-long spread
        0.0,                           # beta2
        1.5,                           # tau1
        0.0,                           # beta3
        5.0,                           # tau2
    ])

    last_good_params = initial_guess.copy()

    print("Calibrating NSS curve day by day...")

    for date, row in yields_decimal.iterrows():
        market_yields = row.values.astype(float)

        params, success, sse = calibrate_nss_single_day(
            maturities=maturities,
            market_yields=market_yields,
            initial_guess=last_good_params,
        )

        if success:
            last_good_params = params
            fitted = nss_yield(maturities, *params)
            residual = market_yields - fitted
        else:
            fitted = np.full(len(USE_MATURITIES), np.nan)
            residual = np.full(len(USE_MATURITIES), np.nan)

        params_history.append(params)
        fitted_history.append(fitted)
        residual_history.append(residual)

    params_df = pd.DataFrame(
        params_history,
        index=yields_percent.index,
        columns=["beta0", "beta1", "beta2", "tau1", "beta3", "tau2"],
    )

    fitted_df = pd.DataFrame(
        fitted_history,
        index=yields_percent.index,
        columns=USE_MATURITIES,
    )

    residual_df = pd.DataFrame(
        residual_history,
        index=yields_percent.index,
        columns=USE_MATURITIES,
    )

    return params_df, fitted_df, residual_df


def calculate_kmo(data: pd.DataFrame):
    """
    Calculate Kaiser-Meyer-Olkin (KMO) measure for sampling adequacy.

    KMO is used to evaluate whether variables share enough common variance
    to justify PCA or factor analysis.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of variables.
        In this project, use daily Treasury yield changes.

    Returns
    -------
    overall_kmo : float
        Overall KMO statistic.

    kmo_per_variable : pd.Series
        KMO statistic for each maturity.
    """

    data = data.dropna()

    # Correlation matrix
    corr = data.corr().values

    # Use pseudo-inverse for numerical stability
    inv_corr = np.linalg.pinv(corr)

    # Partial correlation matrix
    partial_corr = np.zeros_like(corr)

    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i == j:
                partial_corr[i, j] = 0.0
            else:
                partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(
                    inv_corr[i, i] * inv_corr[j, j]
                )

    # Squared correlations and partial correlations
    corr_squared = corr ** 2
    partial_corr_squared = partial_corr ** 2

    # Remove diagonal terms
    np.fill_diagonal(corr_squared, 0.0)
    np.fill_diagonal(partial_corr_squared, 0.0)

    # Overall KMO
    numerator = np.sum(corr_squared)
    denominator = numerator + np.sum(partial_corr_squared)

    overall_kmo = numerator / denominator

    # KMO per variable
    kmo_per_variable_values = np.sum(corr_squared, axis=0) / (
        np.sum(corr_squared, axis=0) + np.sum(partial_corr_squared, axis=0)
    )

    kmo_per_variable = pd.Series(
        kmo_per_variable_values,
        index=data.columns,
        name="KMO"
    )

    return overall_kmo, kmo_per_variable


def interpret_kmo(kmo_value: float) -> str:
    """
    Interpret KMO statistic.
    """

    if kmo_value >= 0.90:
        return "excellent"
    elif kmo_value >= 0.80:
        return "great"
    elif kmo_value >= 0.70:
        return "acceptable"
    elif kmo_value >= 0.60:
        return "mediocre"
    elif kmo_value >= 0.50:
        return "poor"
    else:
        return "not suitable"


def run_kmo_test(yields_percent: pd.DataFrame):
    """
    Run KMO test on Treasury yield changes.

    We use yield changes rather than yield levels because PCA on yield curve
    risk is usually applied to changes in yields.
    """

    yield_changes = yields_percent.diff().dropna()

    overall_kmo, kmo_by_maturity = calculate_kmo(yield_changes)

    print("\nKMO Test Results:")
    print(f"Overall KMO: {overall_kmo:.4f}")
    print(f"Interpretation: {interpret_kmo(overall_kmo)}")

    print("\nKMO by Maturity:")
    print(kmo_by_maturity.round(4))

    return overall_kmo, kmo_by_maturity


def run_pca_analysis(yields_percent):
    """
    PCA on daily yield changes.

    Usually:
        PC1 ≈ level
        PC2 ≈ slope
        PC3 ≈ curvature
    """

    yield_changes = yields_percent.diff().dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(yield_changes)

    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=USE_MATURITIES,
        columns=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    explained = pd.Series(
        pca.explained_variance_ratio_,
        index=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    pc_scores = pd.DataFrame(
        pcs,
        index=yield_changes.index,
        columns=["PC1_Level", "PC2_Slope", "PC3_Curvature"],
    )

    return loadings, explained, pc_scores


def plot_pca_loadings(loadings, explained):
    plt.figure(figsize=(10, 6))

    for col in loadings.columns:
        plt.plot(loadings.index, loadings[col], marker="o", label=col)

    plt.title("PCA Loadings of Treasury Yield Changes")
    plt.xlabel("Maturity")
    plt.ylabel("Loading")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nPCA Explained Variance:")
    print(explained)


def compute_residual_zscores(residual_df, window=252):
    """
    rolling z-score of NSS residuals.

    residual = market yield - fitted yield.

    positive z-score:
        market yield unusually above fair yield
        bond price unusually cheap

    negative z-score:
        market yield unusually below fair yield
        bond price unusually rich
    """

    rolling_mean = residual_df.rolling(window, min_periods=window // 2).mean()
    rolling_std = residual_df.rolling(window, min_periods=window // 2).std()

    zscores = (residual_df - rolling_mean) / rolling_std
    zscores = zscores.replace([np.inf, -np.inf], np.nan)

    return zscores


def build_rich_cheap_positions(zscores, top_n=2, entry_z=1.0, gross_leverage=1.0):
    """
    Build daily long-cheap / short-rich portfolio.

    Long:
        maturities with highest positive residual z-score.

    Short:
        maturities with lowest negative residual z-score.

    We use inverse-duration scaling so that long and short legs
    have more balanced rate-risk exposure.
    """

    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)

    durations = pd.Series(DURATION_PROXY)

    for date, row in zscores.iterrows():
        row = row.dropna()

        if row.empty:
            continue

        cheap = row[row > entry_z].sort_values(ascending=False).head(top_n)
        rich = row[row < -entry_z].sort_values(ascending=True).head(top_n)

        daily_pos = pd.Series(0.0, index=zscores.columns)

        if len(cheap) > 0:
            cheap_weights = 1.0 / durations.loc[cheap.index]
            cheap_weights = cheap_weights / cheap_weights.abs().sum()
            daily_pos.loc[cheap.index] = cheap_weights

        if len(rich) > 0:
            rich_weights = 1.0 / durations.loc[rich.index]
            rich_weights = rich_weights / rich_weights.abs().sum()
            daily_pos.loc[rich.index] = -rich_weights

        # If both long and short legs exist, scale each side to 50%.
        # If only one side exists, use gross leverage on that side.
        long_gross = daily_pos[daily_pos > 0].sum()
        short_gross = daily_pos[daily_pos < 0].abs().sum()

        final_pos = pd.Series(0.0, index=zscores.columns)

        if long_gross > 0 and short_gross > 0:
            final_pos[daily_pos > 0] = daily_pos[daily_pos > 0] / long_gross * (gross_leverage / 2)
            final_pos[daily_pos < 0] = daily_pos[daily_pos < 0] / short_gross * (gross_leverage / 2)
        elif long_gross > 0:
            final_pos[daily_pos > 0] = daily_pos[daily_pos > 0] / long_gross * gross_leverage
        elif short_gross > 0:
            final_pos[daily_pos < 0] = daily_pos[daily_pos < 0] / short_gross * gross_leverage

        positions.loc[date] = final_pos

    return positions


def compute_bond_proxy_returns(yields_percent):
    """
    Approximate Treasury bucket returns using duration + convexity:

        dP/P ≈ -D * dy + 0.5 * C * dy^2

    yields are in percent, so divide by 100 before differencing.
    """
    yields_decimal = yields_percent / 100.0
    yield_changes = yields_decimal.diff()

    durations = pd.Series(DURATION_PROXY)
    convexities = pd.Series(CONVEXITY_PROXY)

    duration_term = -yield_changes.mul(durations, axis=1)
    convexity_term = 0.5 * (yield_changes ** 2).mul(convexities, axis=1)

    bond_returns = duration_term + convexity_term
    return bond_returns


def backtest_strategy(yields_percent, positions,
                      capital=CAPITAL,
                      instrument=INSTRUMENT):
    """
    Strategy return:
        return_t = sum(position_{t-1} * bond_return_t) - transaction_cost_t

    Transaction cost now depends on:
        - capital
        - instrument (future / cash)
        - market impact via ADV and daily vol
    """
    bond_returns = compute_bond_proxy_returns(yields_percent)
    durations = pd.Series(DURATION_PROXY)

    lagged_positions = positions.shift(1).fillna(0.0)

    # 1. Gross return
    gross_returns = (lagged_positions * bond_returns).sum(axis=1)

    # 2. Daily vol per maturity, in bps
    yields_decimal = yields_percent / 100.0
    yield_changes = yields_decimal.diff()
    daily_vol_bps = (yield_changes.rolling(60, min_periods=20).std() * 1e4).ffill()

    # 3. Turnover per maturity
    position_changes = positions.diff().fillna(0.0).abs()   # DataFrame

    # 4. Notional per maturity
    notional_per_maturity = position_changes * capital        # DataFrame

    # 5. Cost per maturity per day
    cost_bps_df = pd.DataFrame(
        0.0, index=positions.index, columns=positions.columns
    )

    for maturity in positions.columns:
        notional = notional_per_maturity[maturity]
        vol = daily_vol_bps[maturity].fillna(0.0)

        cost_bps_df[maturity] = [
            estimate_trade_cost_bps(n, maturity, v)
            for n, v in zip(notional.values, vol.values)
        ]

    # 6. Total cost in return space
    #    cost_return = sum_over_maturities( position_change * cost_bps / 1e4 )
    transaction_cost = (position_changes * cost_bps_df / 1e4).sum(axis=1)

    net_returns = gross_returns - transaction_cost

    results = pd.DataFrame(index=yields_percent.index)
    results["Gross Return"] = gross_returns
    results["Turnover"] = position_changes.sum(axis=1)
    results["Transaction Cost"] = transaction_cost
    results["Net Return"] = net_returns
    results["Equity Curve"] = (1 + net_returns.fillna(0)).cumprod()


    results["Avg Cost Bps"] = (
        (cost_bps_df * position_changes).sum(axis=1)
        / position_changes.sum(axis=1).replace(0, np.nan)
    )

    return results, bond_returns


def estimate_trade_cost_bps(notional_usd, maturity, daily_vol_bps):
    """
    Estimate one-way transaction cost in bps for a single trade.

    Components:
        1. base_cost_bps : instrument-specific base cost
        2. market impact : square-root model

    Parameters
    ----------
    notional_usd : float
        Trade notional in USD
    maturity : str
        Maturity label used to look up assumed ADV
    daily_vol_bps : float
        Daily yield-change volatility for this maturity in basis points

    Returns
    -------
    cost_bps : float
    """
    if notional_usd <= 0:
        return 0.0

    base = BASE_COST_BPS.get(INSTRUMENT, 1.0)

    if not USE_MARKET_IMPACT:
        return base

    adv = ADV_USD.get(maturity, 1e9)
    participation = notional_usd / adv


    impact_bps = IMPACT_K * daily_vol_bps * np.sqrt(max(participation, 0.0))

    return base + impact_bps


def build_benchmarks(yields_percent):
    """
    Build simple duration-proxy benchmarks:
        1. Long 10Y buy-and-hold
        2. Long 20Y buy-and-hold
        3. Equal-weight duration buckets
    """

    bond_returns = compute_bond_proxy_returns(yields_percent)

    benchmarks = pd.DataFrame(index=yields_percent.index)

    benchmarks["Long 10Y"] = bond_returns["10Y"]
    benchmarks["Long 20Y"] = bond_returns["20Y"]
    benchmarks["Equal Weight Curve"] = bond_returns.mean(axis=1)

    benchmark_equity = (1 + benchmarks.fillna(0)).cumprod()

    return benchmarks, benchmark_equity


def plot_yield_curve_snapshot(yields_percent, fitted_df):
    date = yields_percent.index[-1]

    x = np.array([MATURITY_YEARS[m] for m in USE_MATURITIES])

    plt.figure(figsize=(10, 6))
    plt.scatter(x, yields_percent.loc[date, USE_MATURITIES] / 100, label="Market Yield")
    plt.plot(x, fitted_df.loc[date, USE_MATURITIES], marker="o", label="NSS Fitted Yield")

    plt.title(f"NSS Yield Curve Fit on {date.date()}")
    plt.xlabel("Maturity, years")
    plt.ylabel("Yield, decimal")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """Prepare data and curve diagnostics; strategy selection runs below."""
    print("Fetching FRED Treasury yield data...")
    yields = fetch_fred_yields(START_DATE, END_DATE)
    print("Actual data range:", yields.index.min().date(), "to", yields.index.max().date())
    overall_kmo, kmo_by_maturity = run_kmo_test(yields)
    loadings, explained, pc_scores = run_pca_analysis(yields)
    plot_pca_loadings(loadings, explained)
    params_df, fitted_df, residual_df = calibrate_nss_history(yields)
    plot_yield_curve_snapshot(yields, fitted_df)
    return {
        "yields": yields, "params": params_df, "fitted": fitted_df,
        "residuals": residual_df, "overall_kmo": overall_kmo,
        "kmo_by_maturity": kmo_by_maturity, "pca_loadings": loadings,
        "pca_explained": explained, "pc_scores": pc_scores,
    }


def plot_strategy_report(returns, score_start, train_end, test_start, window, entry_z):
    """Selected-parameter IS history and frozen-rule OOS performance."""
    strategy = "NSS strategy (net)"
    split = pd.Timestamp(test_start)
    orange, blue = "#BC6C25", "#2463A6"
    benchmark_styles = [
        ("#404040", "--"), ("#808080", ":"), ("#88754B", "-."),
    ]

    def equity_from(start):
        r = returns.loc[start:].copy()
        if r.empty or r.isna().any().any():
            raise ValueError("Comparison returns must be complete on shared dates.")
        first = returns.index.get_loc(r.index[0])
        origin = returns.index[first - 1] if first else r.index[0] - pd.Timedelta(days=1)
        initial = pd.DataFrame(1.0, index=[origin], columns=r.columns)
        return pd.concat([initial, (1.0 + r).cumprod()])

    full = equity_from(score_start)
    test = equity_from(test_start)

    def benchmarks(ax, equity):
        for col, (color, style) in zip(equity.columns[1:], benchmark_styles):
            ax.plot(equity.index, equity[col], color=color, linestyle=style,
                    linewidth=1.3, label=col)

    def format_axes(axes):
        for ax in axes:
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axes[-1].set_xlabel("Date")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    insample = full.loc[full.index < split, strategy]
    # Include the last IS point to connect segments without resetting NAV.
    boundary = full.index[full.index < split][-1]
    outsample = full.loc[boundary:, strategy]
    axes[0].plot(insample, color=orange, linewidth=2.2,
                 label="NSS: selection / in-sample")
    axes[0].plot(outsample, color=blue, linewidth=2.2,
                 label="NSS: frozen / out-of-sample")
    benchmarks(axes[0], full)
    axes[0].axhline(1, color="#999999", linewidth=0.7)
    axes[0].set_ylabel("Growth of $1")
    axes[0].set_title(f"NSS strategy vs benchmarks | Selected W={window}, Z={entry_z}",
                      loc="left", fontsize=14, pad=30)
    axes[0].legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
    for ax in axes:
        ax.axvspan(full.index[0], split, color=orange, alpha=0.07)
        ax.axvspan(split, full.index[-1], color=blue, alpha=0.05)
        ax.axvline(split, color="#555555", linestyle="--", linewidth=1)
    axes[0].text(0.01, 1.025, "2010–2017 selection (2010 onward includes warm-up)",
                 transform=axes[0].transAxes, color=orange, fontsize=9)
    axes[0].text(0.60, 1.025, "2018 onward: fixed parameters", transform=axes[0].transAxes,
                 color=blue, fontsize=9)
    dd = full[strategy] / full[strategy].cummax() - 1
    for mask, color in [(dd.index < split, orange), (dd.index >= split, blue)]:
        axes[1].fill_between(dd.index, dd.values, 0, where=mask, color=color, alpha=0.35)
    axes[1].plot(dd, color="#555555", linewidth=0.6)
    axes[1].set_ylabel("Strategy drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    format_axes(axes)
    fig.text(0.08, 0.02,
             f"Shared comparison: {full.index[0].date()}–{full.index[-1].date()}. "
             "IS is retrospective; OOS begins 2018.\n"
             "Duration-based price proxies; benchmarks exclude carry, funding and trading costs. "
             "No volatility matching.", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.09, 1, 0.97])
    plt.show()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
    axes[0].plot(test[strategy], color=blue, linewidth=2.2, label=strategy)
    benchmarks(axes[0], test)
    axes[0].axhline(1, color="#999999", linewidth=0.7)
    axes[0].set_title("Out-of-sample comparison | All series rebased to $1 at test entry",
                      loc="left", fontsize=13)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="best", fontsize=9, frameon=False)
    test_dd = test[strategy] / test[strategy].cummax() - 1
    axes[1].fill_between(test_dd.index, test_dd.values, 0, color=blue, alpha=0.3)
    axes[1].set_ylabel("Strategy drawdown")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    format_axes(axes)
    fig.text(0.08, 0.02, "Frozen rules; daily NSS refitting continues. Same proxy assumptions as overview.",
             fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
    return full, test


def run_oos_test(output):
    # ========================================================

    # ========================================================
    yields = output["yields"].copy().sort_index()
    residuals = output["residuals"].copy().sort_index()

    yields = yields.loc["2010-01-01":"2026-12-31"]
    residuals = residuals.reindex(
        index=yields.index,
        columns=yields.columns,
    )

    if yields.index.has_duplicates:
        raise ValueError("Yield data contain duplicate dates.")

    train_start, train_end = "2010-01-01", "2017-12-31"
    test_start, test_end = "2018-01-01", "2026-12-31"

    if yields.loc[train_start:train_end].empty:
        raise ValueError("No observations in the training period.")
    if yields.loc[test_start:test_end].empty:
        raise ValueError("No observations in the test period.")


    windows = [126, 252, 378]
    entry_zs = [1.5, 2.0, 2.5]


    fixed_top_n = 2
    fixed_leverage = 1.0
    fixed_capital = CAPITAL
    fixed_instrument = INSTRUMENT



    train_index = yields.loc[train_start:train_end].index
    warmup = max(windows)

    if len(train_index) <= warmup:
        raise ValueError("Insufficient training history for warm-up.")

    train_score_start = train_index[warmup]

    # ========================================================

    # ========================================================
    def evaluate(window, entry_z, end_date):

        y = yields.loc[:end_date]
        e = residuals.loc[:end_date]

        zscores = compute_residual_zscores(
            e,
            window=window,
        )

        positions = build_rich_cheap_positions(
            zscores=zscores,
            top_n=fixed_top_n,
            entry_z=entry_z,
            gross_leverage=fixed_leverage,
        )

        results, _ = backtest_strategy(
            yields_percent=y,
            positions=positions,
            capital=fixed_capital,
            instrument=fixed_instrument,
        )

        return results, positions

    # ========================================================

    # ========================================================
    def summarize(results, start, end, name):
        returns = results.loc[start:end, "Net Return"].dropna()

        if returns.empty:
            raise ValueError(f"{name} has no valid returns.")
        if not np.isfinite(returns.to_numpy()).all():
            raise ValueError(f"{name} contains non-finite returns.")
        if (returns <= -1).any():
            raise ValueError(f"{name} contains a daily loss of at least 100%.")



        first_loc = results.index.get_loc(returns.index[0])
        if first_loc > 0:
            period_origin = results.index[first_loc - 1]
        else:
            period_origin = returns.index[0]

        years = (
            returns.index[-1] - period_origin
        ).total_seconds() / (365.25 * 24 * 60 * 60)

        equity = (1.0 + returns).cumprod()
        total_return = equity.iloc[-1] - 1.0

        cagr = (
            equity.iloc[-1] ** (1.0 / years) - 1.0
            if years > 0 else np.nan
        )


        ann_vol = returns.std(ddof=1) * np.sqrt(252)
        sharpe = (
            returns.mean() * 252 / ann_vol
            if ann_vol > 0 else np.nan
        )


        peak = equity.cummax().clip(lower=1.0)
        max_drawdown = (equity / peak - 1.0).min()

        return {
            "Period": name,
            "Start": returns.index[0].date(),
            "End": returns.index[-1].date(),
            "Obs": len(returns),
            "Total Return": total_return,
            "CAGR": cagr,
            "Ann Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown,
        }

    # ========================================================

    # ========================================================
    train_rows = []

    for window in windows:
        for entry_z in entry_zs:
            results, _ = evaluate(
                window=window,
                entry_z=entry_z,
                end_date=train_end,
            )

            metrics = summarize(
                results,
                start=train_score_start,
                end=train_end,
                name=f"W{window}_Z{entry_z}",
            )

            train_rows.append({
                "window": window,
                "entry_z": entry_z,
                **metrics,
            })

    train_grid = pd.DataFrame(train_rows)

    eligible = train_grid.loc[
        np.isfinite(train_grid["Sharpe"])
    ].copy()

    if eligible.empty:
        raise ValueError("No candidate has a finite training Sharpe.")



    ranked = eligible.sort_values(
        ["Sharpe", "window", "entry_z"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    best_window = int(ranked.loc[0, "window"])
    best_entry_z = float(ranked.loc[0, "entry_z"])

    print(
        "Common training scoring start:",
        train_score_start.date(),
        "(earlier observations are warm-up only)",
    )

    print("\nTraining parameter comparison:")
    print(
        ranked[
            [
                "window", "entry_z", "Sharpe",
                "CAGR", "Ann Vol", "Max Drawdown",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nFrozen parameters:")
    print(f"window = {best_window}, entry_z = {best_entry_z}")

    if ranked.loc[0, "Sharpe"] <= 0:
        print("Note: the best training Sharpe is not positive.")

    # ========================================================

    # ========================================================
    frozen_results, frozen_positions = evaluate(
        window=best_window,
        entry_z=best_entry_z,
        end_date=test_end,
    )

    summary_rows = [
        summarize(
            frozen_results,
            train_score_start,
            train_end,
            "Train (selection)",
        ),
        summarize(
            frozen_results,
            test_start,
            test_end,
            "Test 2018–2026",
        ),
    ]


    for name, start, end in [
        ("Test 2018–2021", "2018-01-01", "2021-12-31"),
        ("Test 2022–2026", "2022-01-01", "2026-12-31"),
    ]:
        if not frozen_results.loc[start:end].empty:
            summary_rows.append(
                summarize(frozen_results, start, end, name)
            )

    summary = pd.DataFrame(summary_rows)

    print("\nPeriod performance:")
    print(summary.round(4).to_string(index=False))
    print("Test subperiods decompose the same frozen strategy; parameters are not reselected.")

    # ========================================================

    # ========================================================
    test_returns = frozen_results.loc[
        test_start:test_end, "Net Return"
    ].dropna()

    annual_rows = []
    for year, returns in test_returns.groupby(test_returns.index.year):
        annual_rows.append({
            "Year": year,
            "Start": returns.index[0].date(),
            "End": returns.index[-1].date(),
            "Obs": len(returns),
            "Period Return": (1.0 + returns).prod() - 1.0,
        })

    annual_table = pd.DataFrame(annual_rows)

    print("\nTest returns by calendar year:")
    print(annual_table.round(4).to_string(index=False))
    print("The final calendar year is partial through the available data cutoff.")

    # 7. Compare the frozen strategy and benchmarks on identical dates.
    benchmark_returns, _ = build_benchmarks(yields)
    benchmark_returns = benchmark_returns.rename(columns={
        "Long 10Y": "10Y price proxy",
        "Long 20Y": "20Y price proxy",
        "Equal Weight Curve": "Equal-weight price proxy",
    })
    comparison_returns = benchmark_returns.copy()
    comparison_returns.insert(0, "NSS strategy (net)", frozen_results["Net Return"])
    comparison_rows = []
    for label, start_date, end_date in [
        ("Selection (in-sample)", train_score_start, train_end),
        ("Test 2018–2026", test_start, test_end),
    ]:
        for asset in comparison_returns.columns:
            r = comparison_returns[asset].to_frame("Net Return")
            metrics = summarize(r, start_date, end_date, label)
            metrics["Asset"] = asset
            comparison_rows.append(metrics)
    benchmark_summary = pd.DataFrame(comparison_rows)
    print("\nStrategy / benchmark comparison (same dates, no risk scaling):")
    print(benchmark_summary[[
        "Period", "Asset", "Start", "End", "Total Return", "CAGR",
        "Ann Vol", "Sharpe", "Max Drawdown",
    ]].round(4).to_string(index=False))

    full_comparison_equity, test_comparison_equity = plot_strategy_report(
        comparison_returns, train_score_start, train_end, test_start,
        best_window, best_entry_z,
    )
    test_equity = test_comparison_equity["NSS strategy (net)"]

    return {
        "benchmark_summary": benchmark_summary,
        "comparison_returns": comparison_returns,
        "full_comparison_equity": full_comparison_equity,
        "test_comparison_equity": test_comparison_equity,
        "selected_params": {
            "window": best_window,
            "entry_z": best_entry_z,
            "top_n": fixed_top_n,
            "gross_leverage": fixed_leverage,
        },
        "train_grid": ranked,
        "train_score_start": train_score_start,
        "summary": summary,
        "test_annual_returns": annual_table,
        "test_returns": test_returns,
        "test_equity": test_equity,
        "frozen_results": frozen_results,
        "frozen_positions": frozen_positions,
    }


def audit_extreme_days_and_delay(output, oos_output, top_n=10):
    # ========================================================

    # ========================================================
    positions = (
        oos_output["frozen_positions"]
        .copy()
        .sort_index()
    )
    original = (
        oos_output["frozen_results"]
        .copy()
        .sort_index()
        .reindex(positions.index)
    )
    yields = output["yields"].reindex(
        index=positions.index,
        columns=positions.columns,
    )

    if positions.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    params = oos_output["selected_params"]
    residuals = output["residuals"].reindex(
        index=positions.index,
        columns=positions.columns,
    )
    zscores = compute_residual_zscores(
        residuals,
        window=params["window"],
    )

    durations = pd.Series(DURATION_PROXY).reindex(positions.columns)
    convexities = pd.Series(CONVEXITY_PROXY).reindex(positions.columns)

    if durations.isna().any() or convexities.isna().any():
        raise ValueError("Missing duration or convexity for a maturity.")


    yield_change_bp = yields.diff() * 100
    yield_change_decimal = yields.diff() / 100


    held = positions.shift(1).fillna(0.0)
    signal_z = zscores.shift(1)

    linear_contribution = (
        -held * yield_change_decimal.mul(durations, axis=1)
    )
    convexity_contribution = (
        held * 0.5
        * yield_change_decimal.pow(2).mul(convexities, axis=1)
    )
    gross_contribution = linear_contribution + convexity_contribution

    np.testing.assert_allclose(
        gross_contribution.sum(axis=1),
        original["Gross Return"],
        atol=1e-12,
        rtol=1e-9,
        err_msg="PnL reconciliation failed; check the backtest, duration and convexity settings.",
    )

    periods = {
        "Train": (
            pd.Timestamp(oos_output["train_score_start"]),
            pd.Timestamp("2017-12-31"),
        ),
        "Test": (
            pd.Timestamp("2018-01-01"),
            positions.index[-1],
        ),
    }

    # ========================================================

    # ========================================================
    daily_tables = []
    detail_tables = []

    for period, (start, end) in periods.items():
        sample = original.loc[start:end]
        if sample.empty:
            continue

        selected_days = {
            "Best": sample["Net Return"].nlargest(top_n).index,
            "Worst": sample["Net Return"].nsmallest(top_n).index,
        }

        for group, dates in selected_days.items():
            for date in dates:
                i = positions.index.get_loc(date)
                if i == 0:
                    continue

                signal_date = positions.index[i - 1]
                weights = held.loc[date]
                active = weights.abs() > 1e-12
                longs = weights.gt(0).any()
                shorts = weights.lt(0).any()

                if longs and shorts:
                    regime = "Both sides"
                elif longs:
                    regime = "Long only"
                elif shorts:
                    regime = "Short only"
                else:
                    regime = "Flat"

                daily_tables.append({
                    "Period": period,
                    "Group": group,
                    "PnL Date": date,
                    "Position Signal Date": signal_date,
                    "Calendar Gap": (date - signal_date).days,
                    "Regime": regime,
                    "Holdings": ", ".join(
                        f"{m}: {weights[m]:+.1%}"
                        for m in weights.index[active]
                    ) or "None",
                    "Gross Return (%)":
                        original.loc[date, "Gross Return"] * 100,
                    "Cost (%)":
                        original.loc[date, "Transaction Cost"] * 100,
                    "Net Return (%)":
                        original.loc[date, "Net Return"] * 100,
                    "Signed Duration":
                        weights.dot(durations),
                })

                for maturity in weights.index[active]:
                    detail_tables.append({
                        "Period": period,
                        "Group": group,
                        "PnL Date": date,
                        "Position Signal Date": signal_date,
                        "Maturity": maturity,
                        "Held Weight": weights[maturity],
                        "Z at Signal Date": signal_z.loc[date, maturity],
                        "Previous Yield (%)":
                            yields.loc[signal_date, maturity],
                        "Current Yield (%)":
                            yields.loc[date, maturity],
                        "Yield Change (bp)":
                            yield_change_bp.loc[date, maturity],
                        "Duration": durations[maturity],
                        "Linear Contribution (pp)":
                            linear_contribution.loc[date, maturity] * 100,
                        "Convexity Contribution (pp)":
                            convexity_contribution.loc[date, maturity] * 100,
                        "Gross Contribution (pp)":
                            gross_contribution.loc[date, maturity] * 100,
                    })

    extreme_days = pd.DataFrame(daily_tables)
    extreme_details = pd.DataFrame(detail_tables)

    for period in periods:
        for group in ["Best", "Worst"]:
            print(f"\n===== {period}: {group} {top_n} days =====")
            display(
                extreme_days.loc[
                    (extreme_days["Period"] == period)
                    & (extreme_days["Group"] == group)
                ].round(5)
            )

            print("Maturity-level holdings and PnL details:")
            display(
                extreme_details.loc[
                    (extreme_details["Period"] == period)
                    & (extreme_details["Group"] == group)
                ].round(6)
            )

    # ========================================================

    # ========================================================
    reproduced, _ = backtest_strategy(
        yields_percent=yields,
        positions=positions,
        capital=CAPITAL,
        instrument=INSTRUMENT,
    )

    for column in ["Gross Return", "Transaction Cost", "Net Return"]:
        np.testing.assert_allclose(
            reproduced[column],
            original[column],
            atol=1e-12,
            rtol=1e-9,
            err_msg=(
                f"{column} differs from the saved result."
                "Restore the original capital, cost settings and backtest function first."
            ),
        )

    # ========================================================

    # ========================================================



    #

    delayed_positions = positions.shift(1).fillna(0.0)

    delayed, _ = backtest_strategy(
        yields_percent=yields,
        positions=delayed_positions,
        capital=CAPITAL,
        instrument=INSTRUMENT,
    )

    variants = {
        "Original": original,
        "Extra 1-observation delay": delayed,
    }

    # ========================================================

    # ========================================================
    def summarize(results, start, end, period, variant):
        sample = results.loc[start:end]
        r = sample["Net Return"].dropna()

        if r.empty:
            return None
        if not np.isfinite(r.to_numpy()).all() or (r <= -1).any():
            raise ValueError("Returns are not suitable for compounding.")

        first = results.index.get_loc(r.index[0])
        origin = results.index[first - 1] if first > 0 else r.index[0]

        years = (r.index[-1] - origin).total_seconds() / (
            365.25 * 24 * 60 * 60
        )

        equity = (1 + r).cumprod()
        drawdown = equity / equity.cummax().clip(lower=1.0) - 1
        ann_vol = r.std(ddof=1) * np.sqrt(252)

        return {
            "Period": period,
            "Variant": variant,
            "Start": r.index[0].date(),
            "End": r.index[-1].date(),
            "Net Total Return (%)": (equity.iloc[-1] - 1) * 100,
            "Net CAGR (%)": (
                (equity.iloc[-1] ** (1 / years) - 1) * 100
                if years > 0 else np.nan
            ),
            "Gross Ann Mean (%)":
                sample["Gross Return"].mean() * 252 * 100,
            "Cost Ann (%)":
                sample["Transaction Cost"].mean() * 252 * 100,
            "Net Ann Mean (%)": r.mean() * 252 * 100,
            "Ann Vol (%)": ann_vol * 100,
            "Sharpe (vs zero)": (
                r.mean() * 252 / ann_vol
                if ann_vol > 0 else np.nan
            ),
            "Max Drawdown (%)": drawdown.min() * 100,
        }

    rows = []
    for period, (start, end) in periods.items():
        for variant, results in variants.items():
            row = summarize(results, start, end, period, variant)
            if row is not None:
                rows.append(row)

    delay_summary = pd.DataFrame(rows)

    print("\n===== Execution-delay sensitivity =====")
    display(delay_summary.round(4))

    # ========================================================


    # ========================================================
    paired_extreme_days = extreme_days[
        ["Period", "Group", "PnL Date", "Regime"]
    ].copy()

    dates = pd.DatetimeIndex(paired_extreme_days["PnL Date"])
    paired_extreme_days["Original Net (%)"] = (
        original["Net Return"].reindex(dates).to_numpy() * 100
    )
    paired_extreme_days["Delayed Net (%)"] = (
        delayed["Net Return"].reindex(dates).to_numpy() * 100
    )
    paired_extreme_days["Difference (pp)"] = (
        paired_extreme_days["Delayed Net (%)"]
        - paired_extreme_days["Original Net (%)"]
    )

    print("\n===== Original extreme dates: same-date comparison =====")
    display(paired_extreme_days.round(4))

    # ========================================================

    # ========================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "Original": "#555555",
        "Extra 1-observation delay": "#2463A6",
    }

    for ax, (period, (start, end)) in zip(axes, periods.items()):
        for variant, results in variants.items():
            r = results.loc[start:end, "Net Return"].dropna()
            if r.empty:
                continue

            first = results.index.get_loc(r.index[0])
            origin = (
                results.index[first - 1] if first > 0
                else r.index[0] - pd.Timedelta(days=1)
            )

            equity = pd.concat([
                pd.Series([1.0], index=[origin]),
                (1 + r).cumprod(),
            ])

            ax.plot(
                equity.index,
                equity.values,
                color=colors[variant],
                label=variant,
            )

        ax.axhline(1, color="gray", linewidth=0.7)
        ax.set_title(f"{period}: execution-delay sensitivity")
        ax.set_ylabel("Growth of $1")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    return {
        "extreme_days": extreme_days,
        "extreme_details": extreme_details,
        "delay_summary": delay_summary,
        "paired_extreme_days": paired_extreme_days,
        "delayed_positions": delayed_positions,
        "delayed_results": delayed,
    }


def audit_quote_dates_and_execution(output, oos_output):

    selected = oos_output["selected_params"]
    window = int(selected["window"])
    entry_z = float(selected["entry_z"])
    top_n = int(selected.get("top_n", 2))
    leverage = float(selected.get("gross_leverage", 1.0))

    old_yields = output["yields"].copy().sort_index()
    cols = list(old_yields.columns)

    tickers = {
        "3M": "DGS3MO", "6M": "DGS6MO",
        "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3",
        "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10",
        "20Y": "DGS20", "30Y": "DGS30",
    }


    raw = fred_reader.DataReader(
        [tickers[c] for c in cols],
        "fred",
        old_yields.index.min(),
        old_yields.index.max(),
    )
    raw = raw.rename(
        columns={tickers[c]: c for c in cols}
    )[cols].sort_index().astype(float)

    if raw.index.has_duplicates or old_yields.index.has_duplicates:
        raise ValueError("Duplicate dates detected; inspect the raw data.")


    extra_dates = raw.index.difference(old_yields.index)
    if len(extra_dates):
        raise ValueError(
            "Downloaded dates differ from the baseline notebook; "
            "align the data snapshots before comparing."
        )

    raw = raw.reindex(old_yields.index)

    all_missing = raw.isna().all(axis=1)
    partial_missing = raw.isna().any(axis=1) & ~all_missing

    print("===== Raw-data missingness audit =====")
    display(pd.DataFrame({
        "Count": [
            len(raw),
            int(all_missing.sum()),
            int(partial_missing.sum()),
            int(raw.notna().all(axis=1).sum()),
        ]
    }, index=[
        "Rows on the original date index",
        "Rows missing all maturities",
        "Rows missing some maturities",
        "Rows complete across all maturities",
    ]))

    if partial_missing.any():
        display(raw.loc[partial_missing].head(30))
        raise ValueError(
            "Partial maturity missingness detected; stopped. "
            "Inspect the table; no automatic filling or deletion is applied."
        )



    reconstructed = raw.ffill()

    if reconstructed.isna().any().any():
        raise ValueError("Unfillable missing observations at the start of the data.")

    np.testing.assert_allclose(
        reconstructed.to_numpy(),
        old_yields.to_numpy(),
        rtol=0,
        atol=1e-10,
        err_msg=(
            "Downloaded values differ from the baseline data. "
            "Align the snapshots before comparing variants."
        ),
    )

    old_pos = oos_output["frozen_positions"].reindex(
        index=old_yields.index, columns=cols
    )
    old_results = oos_output["frozen_results"].reindex(
        old_yields.index
    )

    if old_pos.isna().any().any():
        raise ValueError("Baseline positions do not fully align with yield dates.")


    target_change = old_pos.diff()
    target_change.iloc[0] = old_pos.iloc[0]
    target_turnover = target_change.abs().sum(axis=1)

    no_quote_audit = pd.DataFrame({
        "Target turnover": target_turnover,
        "Target gross exposure": old_pos.abs().sum(axis=1),
        "Recorded cost (%)":
            old_results["Transaction Cost"] * 100,
    }).loc[all_missing]

    print("\n===== Baseline rebalancing on dates without quotes =====")
    print(
        "No-quote rows with target position changes:",
        int((no_quote_audit["Target turnover"] > 1e-12).sum()),
    )
    display(
        no_quote_audit.sort_values(
            "Target turnover", ascending=False
        ).head(15).round(6)
    )


    original, _ = backtest_strategy(
        old_yields,
        old_pos,
        capital=CAPITAL,
        instrument=INSTRUMENT,
    )

    for column in ["Gross Return", "Transaction Cost", "Net Return"]:
        np.testing.assert_allclose(
            original[column].to_numpy(),
            old_results[column].to_numpy(),
            rtol=1e-8,
            atol=1e-12,
            err_msg=f"Baseline {column} cannot be reproduced; check global configuration.",
        )



    quote_yields = raw.loc[~all_missing].copy()

    print("\n===== Refitting NSS on valid quote dates =====")
    params, fitted, residuals = calibrate_nss_history(quote_yields)

    failed = residuals.isna().any(axis=1)
    print("Rows with missing NSS fits:", int(failed.sum()))

    if failed.any():
        display(quote_yields.loc[failed].head(15))
        raise ValueError(
            "NSS calibration failure detected; stopped "
            "to avoid mixing failed-fit flat positions into the calendar comparison."
        )

    zscores = compute_residual_zscores(
        residuals, window=window
    )
    quote_positions = build_rich_cheap_positions(
        zscores,
        top_n=top_n,
        entry_z=entry_z,
        gross_leverage=leverage,
    )

    clean_result, _ = backtest_strategy(
        quote_yields,
        quote_positions,
        capital=CAPITAL,
        instrument=INSTRUMENT,
    )



    delayed_positions = quote_positions.shift(1).fillna(0.0)
    delayed_result, _ = backtest_strategy(
        quote_yields,
        delayed_positions,
        capital=CAPITAL,
        instrument=INSTRUMENT,
    )



    warmup_obs = max(378, window)
    if len(quote_yields) <= warmup_obs:
        raise ValueError("Insufficient valid quotes for warm-up.")

    common_start = max(
        pd.Timestamp(oos_output["train_score_start"]),
        quote_yields.index[warmup_obs],
    )
    common_end = min(
        old_yields.index.max(),
        quote_yields.index.max(),
    )

    periods = {
        "Train": (
            common_start, pd.Timestamp("2017-12-31")
        ),
        "Test": (
            pd.Timestamp("2018-01-01"), common_end
        ),
    }

    variants = {
        "A Original": original,
        "B Quote dates": clean_result,
        "C Quote dates + delay": delayed_result,
    }

    def metrics(result, start, end):
        part = result.loc[start:end]
        if len(part) < 2:
            raise ValueError("Insufficient observations in the evaluation period.")

        r = part["Net Return"]
        if not np.isfinite(r).all() or (r <= -1).any():
            raise ValueError("Invalid daily returns detected.")

        equity = (1 + r).cumprod()


        first_loc = result.index.get_loc(r.index[0])
        if first_loc == 0:
            raise ValueError("No warm-up observations before the evaluation start.")
        origin = result.index[first_loc - 1]
        years = (r.index[-1] - origin).days / 365.25



        obs_per_year = len(r) / years
        ann_mean = r.mean() * obs_per_year
        ann_vol = r.std(ddof=1) * np.sqrt(obs_per_year)
        drawdown = equity / equity.cummax().clip(lower=1) - 1

        return {
            "Start": r.index[0].date(),
            "End": r.index[-1].date(),
            "Obs": len(r),
            "Obs/year": obs_per_year,
            "Net total (%)": (equity.iloc[-1] - 1) * 100,
            "Net CAGR (%)":
                (equity.iloc[-1] ** (1 / years) - 1) * 100,
            "Gross ann mean (%)":
                part["Gross Return"].mean() * obs_per_year * 100,
            "Cost ann (%)":
                part["Transaction Cost"].mean() * obs_per_year * 100,
            "Ann vol (%)": ann_vol * 100,
            "Sharpe vs zero":
                ann_mean / ann_vol if ann_vol > 0 else np.nan,
            "Max drawdown (%)": drawdown.min() * 100,
        }

    rows = []
    for period, (start, end) in periods.items():
        for name, result in variants.items():
            rows.append({
                "Period": period,
                "Variant": name,
                **metrics(result, start, end),
            })

    summary = pd.DataFrame(rows)

    print(
        f"\nFixed parameters:window={window}, entry_z={entry_z}, "
        f"top_n={top_n}, gross_leverage={leverage}"
    )
    print("Common training scoring start:", common_start.date())
    print("Volatility and Sharpe below use observed annual frequency and may differ from legacy tables.")
    print("\n===== Calendar and execution assumption comparison =====")
    display(summary.round(4))


    idx = quote_yields.index
    dates = pd.Series(idx, index=idx)
    timing = pd.DataFrame({
        "Return interval start": dates.shift(1),
        "Return interval end": dates,
        "B target signal date": dates.shift(1),
        "C target signal date": dates.shift(2),
        "Calendar gap": dates.diff().dt.days,
    })

    print("\n===== Timing example: around 2016-07-04 =====")
    display(timing.loc["2016-06-30":"2016-07-08"])

    return {
        "raw_yields": raw,
        "no_quote_audit": no_quote_audit,
        "quote_yields": quote_yields,
        "params": params,
        "fitted": fitted,
        "residuals": residuals,
        "zscores": zscores,
        "quote_positions": quote_positions,
        "delayed_positions": delayed_positions,
        "results": variants,
        "summary": summary,
        "timing": timing,
        "common_train_start": common_start,
    }
