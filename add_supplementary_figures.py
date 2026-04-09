from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "event_study_data"
ANALYSIS_DIR = DATA_DIR / "analysis_results"
REPORT_DIR = DATA_DIR / "visualizations" / "report"
EVENT_DATE = pd.Timestamp("2023-08-08")
WINDOW_MONTHS = 24


def save_fig(file_name: str) -> None:
    output_path = REPORT_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def build_excess_sharpe_4groups_symmetric_window() -> pd.DataFrame:
    before = pd.read_csv(DATA_DIR / "prices_before_2019-03-11.csv")
    after = pd.read_csv(DATA_DIR / "prices_after_2019-03-11.csv")
    monthly = pd.concat([before, after], ignore_index=True)

    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly = monthly.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    monthly["Monthly_Return"] = monthly.groupby("Ticker")["Adj Close"].pct_change(
        fill_method=None
    )

    spy = (
        monthly[monthly["Ticker"] == "SPY"][["Date", "Monthly_Return"]]
        .rename(columns={"Monthly_Return": "SPY_Return"})
        .dropna()
    )

    merged = monthly.merge(spy, on="Date", how="left")
    merged["Excess_vs_SPY"] = merged["Monthly_Return"] - merged["SPY_Return"]
    merged = merged[merged["Ticker"] != "SPY"].copy()

    before_start = EVENT_DATE - pd.DateOffset(months=WINDOW_MONTHS)
    after_end = EVENT_DATE + pd.DateOffset(months=WINDOW_MONTHS)

    merged["Window"] = np.select(
        [
            (merged["Date"] >= before_start) & (merged["Date"] < EVENT_DATE),
            (merged["Date"] >= EVENT_DATE) & (merged["Date"] < after_end),
        ],
        ["Before", "After"],
        default="Outside",
    )

    win = merged[merged["Window"].isin(["Before", "After"])].copy()

    agg = (
        win.groupby(["Window", "Group"], as_index=False)
        .agg(
            n_obs=("Excess_vs_SPY", lambda s: int(s.dropna().shape[0])),
            mean_excess=("Excess_vs_SPY", "mean"),
            std_excess=("Excess_vs_SPY", "std"),
        )
        .sort_values(["Window", "Group"])
    )
    agg["Sharpe_Excess"] = np.where(
        agg["std_excess"] > 0,
        agg["mean_excess"] / agg["std_excess"],
        np.nan,
    )
    return agg


def plot_sharpe_4groups_before_after() -> pd.DataFrame:
    sharpe_df = build_excess_sharpe_4groups_symmetric_window()

    order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]

    pivot = sharpe_df.pivot_table(
        index="Group", columns="Window", values="Sharpe_Excess", aggfunc="mean"
    ).reindex(order)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(order))
    width = 0.36

    before_vals = pivot["Before"].to_numpy()
    after_vals = pivot["After"].to_numpy()

    bars_before = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label="Before",
        color="#8FA3BF",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_after = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label="After",
        color="#D46A6A",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=12)
    ax.set_ylabel("Excess Sharpe Ratio (Excess mean / Excess std)")
    ax.set_title(
        "Supplementary Fig 18: Excess Sharpe by 4 Groups (Symmetric 24M Before vs After)",
        fontweight="bold",
    )
    ax.legend(frameon=True)

    for bars in [bars_before, bars_after]:
        for bar in bars:
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (0.01 if y >= 0 else -0.01),
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
            )

    save_fig("R08_sharpe_ratio_4groups_before_after.png")
    return pivot


def plot_small_group_delta_sharpe(pivot: pd.DataFrame) -> None:
    small_groups = ["Treatment Small", "Control Small"]
    delta = (
        pivot.loc[small_groups, "After"] - pivot.loc[small_groups, "Before"]
    ).copy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#D46A6A", "#1F77B4"]

    bars = ax.bar(
        delta.index,
        delta.values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Delta Excess Sharpe (After - Before)")
    ax.set_title(
        "Supplementary Fig 21: Small Groups Delta Excess Sharpe (Symmetric 24M)",
        fontweight="bold",
    )

    for bar in bars:
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.005 if y >= 0 else -0.005),
            f"{y:.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=10,
        )

    save_fig("R10_small_group_delta_excess_sharpe.png")


def plot_top_bottom_5_stocks_car() -> None:
    t3 = pd.read_csv(ANALYSIS_DIR / "table_03_daily_stock_car.csv")

    stock_car = (
        t3[["Ticker", "Name", "Group", "CAR_sum"]].dropna(subset=["CAR_sum"]).copy()
    )
    stock_car["CAR_pct"] = stock_car["CAR_sum"] * 100

    bottom_5 = stock_car.nsmallest(5, "CAR_sum").copy()
    top_5 = stock_car.nlargest(5, "CAR_sum").copy()

    bottom_5 = bottom_5.sort_values("CAR_sum", ascending=True)
    top_5 = top_5.sort_values("CAR_sum", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=False)

    labels_bottom = [
        f"{r.Ticker} ({r.Group})" for r in bottom_5.itertuples(index=False)
    ]
    labels_top = [f"{r.Ticker} ({r.Group})" for r in top_5.itertuples(index=False)]

    axes[0].barh(
        labels_bottom,
        bottom_5["CAR_pct"],
        color="#B22222",
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Bottom 5 by Event-Window CAR", fontweight="bold")
    axes[0].set_xlabel("CAR (%)")

    axes[1].barh(
        labels_top, top_5["CAR_pct"], color="#1F77B4", edgecolor="black", linewidth=0.6
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Top 5 by Event-Window CAR", fontweight="bold")
    axes[1].set_xlabel("CAR (%)")

    for ax, data in zip(axes, [bottom_5, top_5]):
        for y, v in enumerate(data["CAR_pct"].to_list()):
            ax.text(
                v + (0.2 if v >= 0 else -0.2),
                y,
                f"{v:.2f}%",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=9,
            )

    fig.suptitle(
        "Supplementary Fig 19/20: Stock Winners & Losers (Event-Window CAR)",
        fontweight="bold",
        y=1.03,
    )
    save_fig("R09_top5_bottom5_stock_car_rankings.png")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = plot_sharpe_4groups_before_after()
    plot_small_group_delta_sharpe(pivot)
    plot_top_bottom_5_stocks_car()
    print(f"All supplementary figures generated in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
