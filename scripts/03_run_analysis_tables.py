from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


EVENT_DATE = pd.Timestamp("2023-08-08")


def safe_ttest_1samp(series: pd.Series, popmean: float = 0.0) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 2:
        return np.nan, np.nan
    t_stat, p_val = stats.ttest_1samp(s, popmean=popmean)
    return float(t_stat), float(p_val)


def safe_ttest_ind(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


def safe_ols(y: pd.Series, x: pd.Series) -> tuple[float, float, float, float, float]:
    tmp = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(tmp) < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(tmp["x"])
    model = sm.OLS(tmp["y"], X).fit()
    alpha = float(model.params.get("const", np.nan))
    beta = float(model.params.get("x", np.nan))
    alpha_p = float(model.pvalues.get("const", np.nan))
    beta_p = float(model.pvalues.get("x", np.nan))
    r2 = float(model.rsquared)
    return alpha, beta, alpha_p, beta_p, r2


def prepare_monthly_returns(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Monthly_Return"] = df.groupby("Ticker")["Adj Close"].pct_change(
        fill_method=None
    )

    spy = (
        df[df["Ticker"] == "SPY"][["Date", "Monthly_Return"]]
        .rename(columns={"Monthly_Return": "SPY_Return"})
        .dropna()
    )
    out = df.merge(spy, on="Date", how="left")
    out["Excess_vs_SPY"] = out["Monthly_Return"] - out["SPY_Return"]
    out["Period"] = period_label
    return out


def run_daily_event_analysis(daily_df: pd.DataFrame, output_dir: Path) -> None:
    daily_df = daily_df.copy()
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    daily_df["SuperGroup"] = np.where(
        daily_df["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )

    # Table 1: AAR by group and date
    t1 = (
        daily_df.groupby(["Date", "Group"], as_index=False)
        .agg(
            n_stocks=("Ticker", "nunique"),
            mean_daily_return=("Daily_Return", "mean"),
            mean_market_adjusted_return=("Market_Adjusted_Return", "mean"),
            std_market_adjusted_return=("Market_Adjusted_Return", "std"),
        )
        .sort_values(["Date", "Group"])
    )
    t1.to_csv(output_dir / "table_01_daily_aar_by_group_date.csv", index=False)

    # Table 2: AAR t-test vs zero
    rows = []
    for (date, group), g in daily_df.groupby(["Date", "Group"]):
        t_stat, p_val = safe_ttest_1samp(g["Market_Adjusted_Return"], popmean=0.0)
        rows.append(
            {
                "Date": date,
                "Group": group,
                "n_stocks": g["Ticker"].nunique(),
                "AAR": g["Market_Adjusted_Return"].mean(),
                "t_stat_vs_0": t_stat,
                "p_value_vs_0": p_val,
            }
        )
    t2 = pd.DataFrame(rows).sort_values(["Date", "Group"])
    t2.to_csv(output_dir / "table_02_daily_aar_ttest_vs_zero.csv", index=False)

    # Table 3: CAR by stock + CAR significance by group
    stock_car = (
        daily_df.groupby(["Ticker", "Name", "Group", "SuperGroup"], as_index=False)
        .agg(
            CAR_sum=("Market_Adjusted_Return", "sum"),
            CAR_compound=("Market_Adjusted_Return", lambda s: (1 + s).prod() - 1),
            mean_AR=("Market_Adjusted_Return", "mean"),
            worst_AR=("Market_Adjusted_Return", "min"),
        )
        .sort_values(["Group", "CAR_sum"])
    )
    stock_car.to_csv(output_dir / "table_03_daily_stock_car.csv", index=False)

    group_rows = []
    for group, g in stock_car.groupby("Group"):
        t_stat, p_val = safe_ttest_1samp(g["CAR_sum"], popmean=0.0)
        group_rows.append(
            {
                "Group": group,
                "n_stocks": g["Ticker"].nunique(),
                "CAAR_sum_mean": g["CAR_sum"].mean(),
                "CAAR_compound_mean": g["CAR_compound"].mean(),
                "t_stat_CAR_vs_0": t_stat,
                "p_value_CAR_vs_0": p_val,
            }
        )
    t3 = pd.DataFrame(group_rows).sort_values("Group")
    t3.to_csv(output_dir / "table_04_daily_caar_ttest_by_group.csv", index=False)

    # Table 4: Treatment vs Control Welch t-tests (by date and whole window)
    rows = []
    for date, g in daily_df.groupby("Date"):
        t_vals = g[g["SuperGroup"] == "Treatment"]["Market_Adjusted_Return"]
        c_vals = g[g["SuperGroup"] == "Control"]["Market_Adjusted_Return"]
        t_stat, p_val = safe_ttest_ind(t_vals, c_vals)
        rows.append(
            {
                "Scope": "ByDate",
                "Date": date,
                "Treatment_mean_AR": t_vals.mean(),
                "Control_mean_AR": c_vals.mean(),
                "Diff_Treat_minus_Control": t_vals.mean() - c_vals.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    all_t = daily_df[daily_df["SuperGroup"] == "Treatment"]["Market_Adjusted_Return"]
    all_c = daily_df[daily_df["SuperGroup"] == "Control"]["Market_Adjusted_Return"]
    t_stat, p_val = safe_ttest_ind(all_t, all_c)
    rows.append(
        {
            "Scope": "FullWindow",
            "Date": pd.NaT,
            "Treatment_mean_AR": all_t.mean(),
            "Control_mean_AR": all_c.mean(),
            "Diff_Treat_minus_Control": all_t.mean() - all_c.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
        }
    )
    t4 = pd.DataFrame(rows)
    t4.to_csv(output_dir / "table_05_daily_treatment_vs_control_tests.csv", index=False)


def run_monthly_analysis(
    monthly_before: pd.DataFrame,
    monthly_after: pd.DataFrame,
    universe: pd.DataFrame,
    output_dir: Path,
) -> None:
    before = prepare_monthly_returns(monthly_before, "Before")
    after = prepare_monthly_returns(monthly_after, "After")
    monthly = pd.concat([before, after], ignore_index=True)

    # keep non-benchmark stocks for stock-level metrics
    non_bench = monthly[monthly["Ticker"] != "SPY"].copy()

    # Table 6: stock-level metrics before/after
    rows = []
    for (period, ticker), g in non_bench.groupby(["Period", "Ticker"]):
        alpha, beta, alpha_p, beta_p, r2 = safe_ols(
            g["Monthly_Return"], g["SPY_Return"]
        )
        rows.append(
            {
                "Period": period,
                "Ticker": ticker,
                "Name": g["Name"].iloc[0],
                "Group": g["Group"].iloc[0],
                "n_months": int(g["Monthly_Return"].dropna().shape[0]),
                "mean_monthly_return": g["Monthly_Return"].mean(),
                "std_monthly_return": g["Monthly_Return"].std(),
                "alpha": alpha,
                "beta": beta,
                "alpha_p_value": alpha_p,
                "beta_p_value": beta_p,
                "reg_r2": r2,
            }
        )
    t6 = pd.DataFrame(rows).sort_values(["Period", "Group", "Ticker"])
    t6.to_csv(
        output_dir / "table_06_monthly_stock_metrics_before_after.csv", index=False
    )

    # Table 7: group metrics for 2 groups (Treatment vs Control)
    grp2 = non_bench.copy()
    grp2["TwoGroup"] = np.where(
        grp2["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )

    rows = []
    for (period, grp), g in grp2.groupby(["Period", "TwoGroup"]):
        alpha, beta, alpha_p, beta_p, r2 = safe_ols(
            g["Monthly_Return"], g["SPY_Return"]
        )
        rows.append(
            {
                "Period": period,
                "TwoGroup": grp,
                "n_obs": int(g["Monthly_Return"].dropna().shape[0]),
                "mean_monthly_return": g["Monthly_Return"].mean(),
                "std_monthly_return": g["Monthly_Return"].std(),
                "alpha": alpha,
                "beta": beta,
                "alpha_p_value": alpha_p,
                "beta_p_value": beta_p,
                "reg_r2": r2,
            }
        )
    t7 = pd.DataFrame(rows).sort_values(["Period", "TwoGroup"])
    t7.to_csv(output_dir / "table_07_monthly_group_metrics_2groups.csv", index=False)

    # Table 8: group metrics for 4 groups (already size-split by predefined group)
    rows = []
    for (period, grp), g in non_bench.groupby(["Period", "Group"]):
        alpha, beta, alpha_p, beta_p, r2 = safe_ols(
            g["Monthly_Return"], g["SPY_Return"]
        )
        rows.append(
            {
                "Period": period,
                "Group": grp,
                "n_obs": int(g["Monthly_Return"].dropna().shape[0]),
                "mean_monthly_return": g["Monthly_Return"].mean(),
                "std_monthly_return": g["Monthly_Return"].std(),
                "alpha": alpha,
                "beta": beta,
                "alpha_p_value": alpha_p,
                "beta_p_value": beta_p,
                "reg_r2": r2,
            }
        )
    t8 = pd.DataFrame(rows).sort_values(["Period", "Group"])
    t8.to_csv(output_dir / "table_08_monthly_group_metrics_4groups.csv", index=False)

    # Table 9: does beta explain average returns? cross-sectional regression by period
    rows = []
    for period in ["Before", "After"]:
        sub = t6[t6["Period"] == period].dropna(subset=["mean_monthly_return", "beta"])
        if len(sub) < 6:
            rows.append(
                {
                    "Period": period,
                    "n_stocks": len(sub),
                    "intercept": np.nan,
                    "beta_coef": np.nan,
                    "intercept_p_value": np.nan,
                    "beta_p_value": np.nan,
                    "r_squared": np.nan,
                }
            )
            continue
        X = sm.add_constant(sub["beta"])
        model = sm.OLS(sub["mean_monthly_return"], X).fit()
        rows.append(
            {
                "Period": period,
                "n_stocks": len(sub),
                "intercept": float(model.params.get("const", np.nan)),
                "beta_coef": float(model.params.get("beta", np.nan)),
                "intercept_p_value": float(model.pvalues.get("const", np.nan)),
                "beta_p_value": float(model.pvalues.get("beta", np.nan)),
                "r_squared": float(model.rsquared),
            }
        )
    t9 = pd.DataFrame(rows)
    t9.to_csv(output_dir / "table_09_beta_explains_return_regression.csv", index=False)

    # Table 10: before vs after changes by stock
    before_metrics = t6[t6["Period"] == "Before"].set_index("Ticker")
    after_metrics = t6[t6["Period"] == "After"].set_index("Ticker")
    common = before_metrics.index.intersection(after_metrics.index)
    change = pd.DataFrame(
        {
            "Ticker": common,
            "Name": before_metrics.loc[common, "Name"].values,
            "Group": before_metrics.loc[common, "Group"].values,
            "mean_return_before": before_metrics.loc[
                common, "mean_monthly_return"
            ].values,
            "mean_return_after": after_metrics.loc[
                common, "mean_monthly_return"
            ].values,
            "delta_mean_return": (
                after_metrics.loc[common, "mean_monthly_return"].values
                - before_metrics.loc[common, "mean_monthly_return"].values
            ),
            "beta_before": before_metrics.loc[common, "beta"].values,
            "beta_after": after_metrics.loc[common, "beta"].values,
            "delta_beta": (
                after_metrics.loc[common, "beta"].values
                - before_metrics.loc[common, "beta"].values
            ),
            "alpha_before": before_metrics.loc[common, "alpha"].values,
            "alpha_after": after_metrics.loc[common, "alpha"].values,
            "delta_alpha": (
                after_metrics.loc[common, "alpha"].values
                - before_metrics.loc[common, "alpha"].values
            ),
        }
    )
    change = change.sort_values(["Group", "Ticker"])
    change.to_csv(
        output_dir / "table_10_monthly_before_after_change_by_stock.csv", index=False
    )


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "event_study_data"
    out_dir = data_dir / "analysis_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(data_dir / "event_window_daily.csv")
    monthly_before = pd.read_csv(data_dir / "prices_before_event_monthly.csv")
    monthly_after = pd.read_csv(data_dir / "prices_after_event_monthly.csv")
    universe = pd.read_csv(data_dir / "universe_tickers.csv")

    run_daily_event_analysis(daily, out_dir)
    run_monthly_analysis(monthly_before, monthly_after, universe, out_dir)

    print("Analysis completed. Output tables:")
    for p in sorted(out_dir.glob("*.csv")):
        print(f"- {p.name}")


if __name__ == "__main__":
    main()
