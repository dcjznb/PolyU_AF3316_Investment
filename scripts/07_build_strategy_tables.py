from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 10

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "event_study_data"
ANALYSIS_DIR = DATA_DIR / "analysis_results"
OUT_DIR = DATA_DIR / "visualizations" / "report" / "strategy"

EVENT_DATE = pd.Timestamp("2023-08-08")
WINDOW_MONTHS = 24
RISK_FREE_ANNUAL = 0.02


def save_table(df: pd.DataFrame, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    df.to_csv(out_path, index=False)
    print(f"Saved table: {out_path}")
    return out_path


def save_fig(name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")
    return out_path


def build_strategy_table_04_sharpe() -> pd.DataFrame:
    before = pd.read_csv(DATA_DIR / "prices_before_event_monthly.csv")
    after = pd.read_csv(DATA_DIR / "prices_after_event_monthly.csv")
    monthly = pd.concat([before, after], ignore_index=True)

    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly = monthly.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    monthly["Monthly_Return"] = monthly.groupby("Ticker")["Adj Close"].pct_change(
        fill_method=None
    )
    monthly = monthly[monthly["Ticker"] != "SPY"].copy()

    before_start = EVENT_DATE - pd.DateOffset(months=WINDOW_MONTHS)
    after_end = EVENT_DATE + pd.DateOffset(months=WINDOW_MONTHS)

    monthly["Period"] = np.select(
        [
            (monthly["Date"] >= before_start) & (monthly["Date"] < EVENT_DATE),
            (monthly["Date"] >= EVENT_DATE) & (monthly["Date"] < after_end),
        ],
        ["Before", "After"],
        default="Outside",
    )

    sample = monthly[monthly["Period"].isin(["Before", "After"])].copy()

    rf_monthly = (1 + float(RISK_FREE_ANNUAL)) ** (1 / 12) - 1
    sample["Excess_vs_RF"] = sample["Monthly_Return"] - rf_monthly

    grouped = (
        sample.groupby(["Group", "Period"], as_index=False)
        .agg(
            n_obs=("Excess_vs_RF", lambda s: int(s.dropna().shape[0])),
            mean_excess=("Excess_vs_RF", "mean"),
            std_excess=("Excess_vs_RF", "std"),
        )
        .sort_values(["Group", "Period"])
    )

    grouped["sharpe_annualized"] = np.where(
        grouped["std_excess"] > 0,
        (grouped["mean_excess"] / grouped["std_excess"]) * np.sqrt(12),
        np.nan,
    )

    pivot = grouped.pivot(index="Group", columns="Period", values="sharpe_annualized")
    out = (
        pd.DataFrame(
            {
                "Group": pivot.index,
                "Sharpe_Before": pivot["Before"].values,
                "Sharpe_After": pivot["After"].values,
            }
        )
        .assign(
            Delta_After_Minus_Before=lambda d: d["Sharpe_After"] - d["Sharpe_Before"],
            Abs_Magnitude_Reduction=lambda d: np.abs(d["Sharpe_Before"])
            - np.abs(d["Sharpe_After"]),
        )
        .sort_values("Delta_After_Minus_Before", ascending=False)
        .reset_index(drop=True)
    )

    return out


def build_strategy_table_06_treatment_small() -> pd.DataFrame:
    t6 = pd.read_csv(ANALYSIS_DIR / "table_06_monthly_stock_metrics_before_after.csv")
    after_small = t6[
        (t6["Period"] == "After") & (t6["Group"] == "Treatment Small")
    ].copy()

    out = after_small[
        [
            "Ticker",
            "Name",
            "mean_monthly_return",
            "std_monthly_return",
            "alpha",
            "beta",
            "reg_r2",
        ]
    ].copy()
    out["mean_monthly_return_pct"] = out["mean_monthly_return"] * 100
    out["alpha_pct"] = out["alpha"] * 100

    out["Strategy_Role"] = "Other"
    out.loc[out["Ticker"] == "LMAT", "Strategy_Role"] = "Risk-Neutral Core"
    out.loc[out["Ticker"] == "ANGO", "Strategy_Role"] = "Alpha Focus"
    out.loc[out["Ticker"] == "TNDM", "Strategy_Role"] = "High-Beta Satellite"

    out = out.sort_values(
        ["alpha", "mean_monthly_return"], ascending=False
    ).reset_index(drop=True)
    return out


def build_strategy_table_08_group_beta() -> pd.DataFrame:
    t8 = pd.read_csv(ANALYSIS_DIR / "table_08_monthly_group_metrics_4groups.csv")
    after = t8[t8["Period"] == "After"].copy()
    out = after[
        [
            "Group",
            "mean_monthly_return",
            "std_monthly_return",
            "alpha",
            "beta",
            "reg_r2",
        ]
    ].copy()
    out["mean_monthly_return_pct"] = out["mean_monthly_return"] * 100
    out = out.sort_values("beta", ascending=False).reset_index(drop=True)
    out["beta_rank_desc"] = np.arange(1, len(out) + 1)
    return out


def build_strategy_table_figure7_screen() -> pd.DataFrame:
    daily = pd.read_csv(DATA_DIR / "event_window_daily.csv")
    t6 = pd.read_csv(ANALYSIS_DIR / "table_06_monthly_stock_metrics_before_after.csv")
    t1 = pd.read_csv(ANALYSIS_DIR / "table_01_daily_aar_by_group_date.csv")

    daily["Date"] = pd.to_datetime(daily["Date"])
    t1["Date"] = pd.to_datetime(t1["Date"])

    ts_daily = daily[daily["Group"] == "Treatment Small"].copy()
    ts_daily = ts_daily.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    event_day_group_avg = t1[
        (t1["Date"] == EVENT_DATE) & (t1["Group"] == "Treatment Small")
    ]["mean_market_adjusted_return"].iloc[0]

    event_stock = ts_daily[ts_daily["Date"] == EVENT_DATE][
        ["Ticker", "Name", "Market_Adjusted_Return"]
    ].rename(columns={"Market_Adjusted_Return": "event_day_ar"})

    end_window_car = (
        ts_daily.groupby(["Ticker", "Name"], as_index=False)["Market_Adjusted_Return"]
        .sum()
        .rename(columns={"Market_Adjusted_Return": "event_window_car"})
    )

    post_alpha = t6[(t6["Period"] == "After") & (t6["Group"] == "Treatment Small")][
        ["Ticker", "alpha"]
    ].copy()

    screen = event_stock.merge(end_window_car, on=["Ticker", "Name"], how="left")
    screen = screen.merge(post_alpha, on="Ticker", how="left")

    screen["group_avg_event_day_aar"] = event_day_group_avg
    screen["criterion1_less_decline_than_group_avg"] = (
        screen["event_day_ar"] > event_day_group_avg
    )
    screen["criterion2_end_window_car_above_neg1pct"] = (
        screen["event_window_car"] > -0.01
    )
    screen["criterion3_post_alpha_positive"] = screen["alpha"] > 0
    screen["criteria_passed"] = (
        screen[
            [
                "criterion1_less_decline_than_group_avg",
                "criterion2_end_window_car_above_neg1pct",
                "criterion3_post_alpha_positive",
            ]
        ]
        .sum(axis=1)
        .astype(int)
    )
    screen["recommended"] = screen["criteria_passed"] == 3

    screen = screen.sort_values(["criteria_passed", "alpha"], ascending=[False, False])
    screen = screen.reset_index(drop=True)
    return screen


def plot_strategy_figure_04_sharpe(table_04: pd.DataFrame) -> None:
    order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    t = table_04.set_index("Group").reindex(order)

    x = np.arange(len(order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_before = ax.bar(
        x - width / 2,
        t["Sharpe_Before"],
        width,
        label="Before",
        color="#7EA7D8",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_after = ax.bar(
        x + width / 2,
        t["Sharpe_After"],
        width,
        label="After",
        color="#EB7B59",
        edgecolor="black",
        linewidth=0.6,
    )

    ts_idx = order.index("Treatment Small")
    ax.add_patch(
        plt.Rectangle(
            (
                ts_idx - 0.58,
                min(t["Sharpe_Before"].min(), t["Sharpe_After"].min()) - 0.08,
            ),
            1.16,
            (
                max(t["Sharpe_Before"].max(), t["Sharpe_After"].max())
                - min(t["Sharpe_Before"].min(), t["Sharpe_After"].min())
                + 0.16
            ),
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="#A11212",
        )
    )

    ax.text(
        ts_idx,
        t.loc["Treatment Small", ["Sharpe_Before", "Sharpe_After"]].max() + 0.07,
        "Largest Improvement",
        ha="center",
        va="bottom",
        color="#A11212",
        fontsize=9,
        fontweight="bold",
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylabel("Annualized Sharpe ((R - Rf) / Std)")
    ax.set_title(
        "Strategy Figure 4: Annualized Sharpe Before vs After by Group",
        fontweight="bold",
    )
    ax.legend(frameon=True)

    for bars in [bars_before, bars_after]:
        for bar in bars:
            y = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (0.02 if y >= 0 else -0.02),
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
            )

    save_fig("figure_strategy_04_annualized_sharpe_4groups.png")


def plot_strategy_figure_07_quality_screen(screen: pd.DataFrame) -> None:
    daily = pd.read_csv(DATA_DIR / "event_window_daily.csv")
    daily["Date"] = pd.to_datetime(daily["Date"])
    ts_daily = daily[daily["Group"] == "Treatment Small"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    plot_df = screen.copy()
    plot_df["event_day_ar_pct"] = plot_df["event_day_ar"] * 100
    plot_df["alpha_pct"] = plot_df["alpha"] * 100
    plot_df["event_window_car_pct"] = plot_df["event_window_car"] * 100

    scatter = ax.scatter(
        plot_df["event_day_ar_pct"],
        plot_df["alpha_pct"],
        s=np.clip(np.abs(plot_df["event_window_car_pct"]) * 12 + 70, 70, 260),
        c=plot_df["criteria_passed"],
        cmap="RdYlGn",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.95,
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["event_day_ar_pct"] + 0.05,
            row["alpha_pct"] + 0.03,
            row["Ticker"],
            fontsize=8,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    group_threshold = float(plot_df["group_avg_event_day_aar"].iloc[0] * 100)
    ax.axvline(group_threshold, color="#A11212", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Event-Day AR (%)  [higher = less decline]")
    ax.set_ylabel("Post-Event Alpha (%/month)")
    ax.set_title(
        "Criterion 1 + 3: Less Event Drop and Positive Alpha", fontweight="bold"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Criteria Passed")

    ax = axes[1]
    ts_daily = ts_daily.sort_values(["Ticker", "Date"])
    ts_daily["CAR_path"] = ts_daily.groupby("Ticker")["Market_Adjusted_Return"].cumsum()

    for ticker, g in ts_daily.groupby("Ticker"):
        if ticker == "LMAT":
            ax.plot(
                g["Date"],
                g["CAR_path"] * 100,
                marker="o",
                linewidth=2.4,
                color="#A11212",
                label="LMAT",
                zorder=4,
            )
        else:
            ax.plot(
                g["Date"],
                g["CAR_path"] * 100,
                marker="o",
                linewidth=1.0,
                color="#8A8A8A",
                alpha=0.75,
                zorder=2,
            )

    ax.axhline(-1.0, color="#1F77B4", linewidth=1.2, linestyle="--")
    ax.text(
        ts_daily["Date"].max(),
        -1.0,
        "  Recovery threshold = -1%",
        va="bottom",
        ha="left",
        fontsize=8,
        color="#1F77B4",
    )
    ax.axvline(EVENT_DATE, color="black", linewidth=1.1, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("CAR Path (%)")
    ax.set_title("Criterion 2: Fast CAR Recovery in Event Window", fontweight="bold")
    ax.legend(loc="best", frameon=True)

    fig.suptitle(
        "Strategy Figure 7: Treatment-Small Quality Resilience Screen",
        fontweight="bold",
    )
    save_fig("figure_strategy_07_quality_resilience_screen.png")


def main() -> None:
    print("Building strategy evidence tables...")

    table_06 = build_strategy_table_06_treatment_small()
    table_08 = build_strategy_table_08_group_beta()

    save_table(table_06, "tbl_str_01_treatment_small_post_fundamentals.csv")
    save_table(table_08, "tbl_str_02_group_beta_after_event.csv")

    print(f"Done. All outputs are in: {OUT_DIR}")


if __name__ == "__main__":
    main()
