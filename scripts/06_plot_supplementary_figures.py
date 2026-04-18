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
REPORT_DIR = DATA_DIR / "visualizations" / "report"
EVENT_DATE = pd.Timestamp("2023-08-08")
WINDOW_MONTHS = 24
RISK_FREE_ANNUAL = 0.02


def save_fig(file_name: str) -> None:
    output_path = REPORT_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def build_excess_sharpe_4groups_symmetric_window() -> pd.DataFrame:
    before = pd.read_csv(DATA_DIR / "prices_before_event_monthly.csv")
    after = pd.read_csv(DATA_DIR / "prices_after_event_monthly.csv")
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

    save_fig("figure_event_08_sharpe_ratio_4groups_before_after.png")
    return pivot


def build_regular_sharpe_4groups_symmetric_window(
    risk_free_annual: float = RISK_FREE_ANNUAL,
    annualize: bool = True,
) -> pd.DataFrame:
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

    monthly["Window"] = np.select(
        [
            (monthly["Date"] >= before_start) & (monthly["Date"] < EVENT_DATE),
            (monthly["Date"] >= EVENT_DATE) & (monthly["Date"] < after_end),
        ],
        ["Before", "After"],
        default="Outside",
    )

    win = monthly[monthly["Window"].isin(["Before", "After"])].copy()

    rf_monthly = (1 + float(risk_free_annual)) ** (1 / 12) - 1
    win["Excess_vs_RF"] = win["Monthly_Return"] - rf_monthly

    agg = (
        win.groupby(["Window", "Group"], as_index=False)
        .agg(
            n_obs=("Excess_vs_RF", lambda s: int(s.dropna().shape[0])),
            mean_excess_rf=("Excess_vs_RF", "mean"),
            std_excess_rf=("Excess_vs_RF", "std"),
        )
        .sort_values(["Window", "Group"])
    )

    scale = np.sqrt(12) if annualize else 1.0
    agg["Sharpe_Regular"] = np.where(
        agg["std_excess_rf"] > 0,
        (agg["mean_excess_rf"] / agg["std_excess_rf"]) * scale,
        np.nan,
    )
    agg["risk_free_annual"] = float(risk_free_annual)
    agg["annualized"] = bool(annualize)
    return agg


def plot_regular_sharpe_4groups_before_after(
    risk_free_annual: float = RISK_FREE_ANNUAL,
    annualize: bool = True,
) -> pd.DataFrame:
    sharpe_df = build_regular_sharpe_4groups_symmetric_window(
        risk_free_annual=risk_free_annual,
        annualize=annualize,
    )

    order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    pivot = sharpe_df.pivot_table(
        index="Group", columns="Window", values="Sharpe_Regular", aggfunc="mean"
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
        color="#5E9C76",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_after = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label="After",
        color="#E08E45",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=12)

    y_label = (
        "Annualized Sharpe Ratio ((R - Rf) mean / std)"
        if annualize
        else "Monthly Sharpe Ratio ((R - Rf) mean / std)"
    )
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Supplementary Fig 08B: Regular Sharpe by 4 Groups (Rf={risk_free_annual:.1%}, Symmetric 24M Before vs After)",
        fontweight="bold",
    )
    ax.legend(frameon=True)

    for bars in [bars_before, bars_after]:
        for bar in bars:
            y = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + (0.02 if y >= 0 else -0.02),
                f"{y:.3f}",
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
            )

    save_fig("figure_event_08b_regular_sharpe_ratio_4groups_before_after.png")
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

    save_fig("figure_event_10_small_group_delta_excess_sharpe.png")


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

    group_color = {
        "Treatment": "#0B789B",
        "Control": "#FF6A00",
    }
    colors_bottom = [
        group_color["Treatment"] if "Treatment" in g else group_color["Control"]
        for g in bottom_5["Group"].to_list()
    ]
    colors_top = [
        group_color["Treatment"] if "Treatment" in g else group_color["Control"]
        for g in top_5["Group"].to_list()
    ]

    axes[0].barh(
        labels_bottom,
        bottom_5["CAR_pct"],
        color=colors_bottom,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Bottom 5 by Event-Window CAR", fontweight="bold")
    axes[0].set_xlabel("CAR (%)")

    axes[1].barh(
        labels_top,
        top_5["CAR_pct"],
        color=colors_top,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Top 5 by Event-Window CAR", fontweight="bold")
    axes[1].set_xlabel("CAR (%)")

    for ax, data in zip(axes, [bottom_5, top_5]):
        for y, row in enumerate(data.itertuples(index=False)):
            v = float(row.CAR_pct)
            x_text = v + (0.2 if v >= 0 else -0.2)
            ha = "left" if v >= 0 else "right"

            # Avoid overlap for the leftmost label in R09 (RMD treatment-big row).
            if row.Ticker == "RMD" and row.Group == "Treatment Big":
                ax.text(
                    -0.08,
                    y - 0.22,
                    f"{v:.2f}%",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=9,
                    clip_on=False,
                )
                continue

            ax.text(x_text, y, f"{v:.2f}%", va="center", ha=ha, fontsize=9)

    fig.suptitle(
        "Best and Worst Performers (Top 5 vs. Bottom 5 CAR)",
        fontweight="bold",
        y=1.03,
    )
    save_fig("figure_event_09_top5_bottom5_stock_car_rankings.png")


def plot_control_decline_attribution() -> None:
    daily = pd.read_csv(DATA_DIR / "event_window_daily.csv")
    daily["Date"] = pd.to_datetime(daily["Date"])

    control = daily[daily["Group"].str.contains("Control", na=False)].copy()
    if control.empty:
        return

    group_daily = (
        control.groupby(["Date", "Group"], as_index=False)
        .agg(
            n_stocks=("Ticker", "nunique"),
            group_aar=("Market_Adjusted_Return", "mean"),
        )
        .sort_values(["Date", "Group"])
    )
    total_n = (
        group_daily.groupby("Date", as_index=False)["n_stocks"]
        .sum()
        .rename(columns={"n_stocks": "n_control_total"})
    )
    group_daily = group_daily.merge(total_n, on="Date", how="left")
    group_daily["weight_in_control"] = (
        group_daily["n_stocks"] / group_daily["n_control_total"]
    )
    group_daily["contribution_to_control_aar"] = (
        group_daily["group_aar"] * group_daily["weight_in_control"]
    )

    control_aar = (
        control.groupby("Date", as_index=False)
        .agg(control_aar=("Market_Adjusted_Return", "mean"))
        .sort_values("Date")
    )

    stock_daily = (
        control.groupby(["Date", "Ticker", "Name", "Group"], as_index=False)
        .agg(stock_ar=("Market_Adjusted_Return", "mean"))
        .merge(total_n, on="Date", how="left")
    )
    stock_daily["contribution_to_control_aar"] = (
        stock_daily["stock_ar"] / stock_daily["n_control_total"]
    )

    post = stock_daily[stock_daily["Date"] > EVENT_DATE].copy()
    post_stock = (
        post.groupby(["Ticker", "Name", "Group"], as_index=False)
        .agg(
            mean_post_ar=("stock_ar", "mean"),
            cum_contribution_post=("contribution_to_control_aar", "sum"),
            mean_contribution_post=("contribution_to_control_aar", "mean"),
            n_days=("Date", "nunique"),
        )
        .sort_values("cum_contribution_post")
    )
    post_stock["label"] = post_stock["Ticker"] + " (" + post_stock["Group"] + ")"

    group_export = group_daily.copy()
    group_export["group_aar_pct"] = group_export["group_aar"] * 100
    group_export["contribution_pct"] = group_export["contribution_to_control_aar"] * 100
    group_export.to_csv(
        REPORT_DIR / "tbl_lt_01_control_decline_group_contribution.csv", index=False
    )

    post_export = post_stock.copy()
    post_export["mean_post_ar_pct"] = post_export["mean_post_ar"] * 100
    post_export["cum_contribution_post_pct"] = (
        post_export["cum_contribution_post"] * 100
    )
    post_export["mean_contribution_post_pct"] = (
        post_export["mean_contribution_post"] * 100
    )
    post_export.to_csv(
        REPORT_DIR / "tbl_lt_02_control_decline_stock_contribution_post_event.csv",
        index=False,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    ax1, ax2, ax3 = axes

    timeline = control_aar.copy()
    timeline["DateStr"] = timeline["Date"].dt.strftime("%Y-%m-%d")
    x = np.arange(len(timeline))
    event_idx = timeline.index[timeline["Date"] == EVENT_DATE]
    event_loc = int(event_idx[0]) if len(event_idx) > 0 else None

    ax1.plot(x, timeline["control_aar"] * 100, color="#1F77B4", marker="o", linewidth=2)
    ax1.axhline(0, color="black", linewidth=1)
    if event_loc is not None:
        ax1.axvline(event_loc, linestyle="--", color="black", linewidth=1.2)
    ax1.set_ylabel("Control AAR (%)")
    ax1.set_title("Control Group AAR Around Event Date", fontweight="bold")

    pivot = group_daily.pivot_table(
        index="Date",
        columns="Group",
        values="contribution_to_control_aar",
        aggfunc="sum",
    ).reindex(timeline["Date"])
    c_big = (
        pivot.get("Control Big", pd.Series(0, index=pivot.index)).fillna(0) * 100
    ).to_numpy()
    c_small = (
        pivot.get("Control Small", pd.Series(0, index=pivot.index)).fillna(0) * 100
    ).to_numpy()

    width = 0.36
    big_bars = ax2.bar(
        x - width / 2,
        c_big,
        width=width,
        label="Control Big contribution",
        color="#4C78A8",
    )
    small_bars = ax2.bar(
        x + width / 2,
        c_small,
        width=width,
        label="Control Small contribution",
        color="#F58518",
    )
    net = c_big + c_small
    ax2.plot(
        x,
        net,
        color="#2F2F2F",
        linestyle="--",
        marker="o",
        linewidth=1.8,
        label="Net control contribution",
    )
    ax2.axhline(0, color="black", linewidth=1)
    if event_loc is not None:
        ax2.axvline(event_loc, linestyle="--", color="black", linewidth=1.2)
        ax2.axvspan(event_loc - 0.5, len(x) - 0.5, color="#ECECEC", alpha=0.35)
    ax2.set_ylabel("Contribution to Control AAR (%)")
    ax2.set_title(
        "Daily Contribution Decomposition (Big vs Small + Net)",
        fontweight="bold",
    )
    ax2.legend(loc="upper right", frameon=True, ncol=1)

    for bars in [big_bars, small_bars]:
        for bar in bars:
            v = float(bar.get_height())
            y = v + (0.06 if v >= 0 else -0.06)
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )

    worst = post_stock.head(8).copy()
    worst = worst.sort_values("cum_contribution_post", ascending=True)
    ax3.barh(
        worst["label"],
        worst["cum_contribution_post"] * 100,
        color="#D62728",
        edgecolor="black",
        linewidth=0.6,
    )
    ax3.axvline(0, color="black", linewidth=1)
    ax3.set_xlabel("Cum contribution to control AAR after event (pp)")
    ax3.set_title("Top Negative Stock Contributors After Event", fontweight="bold")

    for i, v in enumerate((worst["cum_contribution_post"] * 100).to_numpy()):
        ax3.text(v - 0.01, i, f"{v:.2f}", ha="right", va="center", fontsize=9)

    ax3.set_xticks(ax3.get_xticks())
    ax3.set_xticklabels([f"{v:.2f}" for v in ax3.get_xticks()])

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(timeline["DateStr"].to_list(), rotation=30, ha="right")

    save_fig("figure_event_11_control_decline_attribution.png")

    post_group = (
        group_daily[group_daily["Date"] > EVENT_DATE]
        .groupby("Group", as_index=False)["contribution_to_control_aar"]
        .sum()
        .sort_values("contribution_to_control_aar")
    )
    post_group["contribution_pct"] = post_group["contribution_to_control_aar"] * 100

    print("\nControl decline attribution (post-event cumulative contribution, pp):")
    for row in post_group.itertuples(index=False):
        print(f"- {row.Group}: {row.contribution_pct:.3f}")

    print("\nMost negative post-event stock contributors (pp):")
    for row in post_stock.head(5).itertuples(index=False):
        print(f"- {row.Ticker} ({row.Group}): {row.cum_contribution_post * 100:.3f}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pivot = plot_sharpe_4groups_before_after()
    plot_regular_sharpe_4groups_before_after()
    plot_small_group_delta_sharpe(pivot)
    plot_top_bottom_5_stocks_car()
    plot_control_decline_attribution()
    print(f"All supplementary figures generated in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
