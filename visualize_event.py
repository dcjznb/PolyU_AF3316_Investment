from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "event_study_data"
ANALYSIS_DIR = DATA_DIR / "analysis_results"
OUT_DIR = DATA_DIR / "visualizations"
REPORT_DIR = OUT_DIR / "report"
EVENT_DATE = pd.Timestamp("2023-08-08")
EVENT_X = float(mdates.date2num(EVENT_DATE.to_pydatetime()))


def load_tables() -> dict[str, pd.DataFrame]:
    daily_candidates = sorted(DATA_DIR.glob("event_window_daily_*.csv"))
    if not daily_candidates:
        raise FileNotFoundError(
            "No event-window daily file found under event_study_data."
        )
    daily_path = daily_candidates[-1]

    tables = {
        "daily_raw": pd.read_csv(daily_path),
        "t1": pd.read_csv(ANALYSIS_DIR / "table_01_daily_aar_by_group_date.csv"),
        "t3": pd.read_csv(ANALYSIS_DIR / "table_03_daily_stock_car.csv"),
        "t5": pd.read_csv(
            ANALYSIS_DIR / "table_05_daily_treatment_vs_control_tests.csv"
        ),
        "t6": pd.read_csv(
            ANALYSIS_DIR / "table_06_monthly_stock_metrics_before_after.csv"
        ),
        "t7": pd.read_csv(ANALYSIS_DIR / "table_07_monthly_group_metrics_2groups.csv"),
        "t8": pd.read_csv(ANALYSIS_DIR / "table_08_monthly_group_metrics_4groups.csv"),
        "t10": pd.read_csv(
            ANALYSIS_DIR / "table_10_monthly_before_after_change_by_stock.csv"
        ),
    }
    tables["daily_raw"]["Date"] = pd.to_datetime(tables["daily_raw"]["Date"])
    tables["t1"]["Date"] = pd.to_datetime(tables["t1"]["Date"])
    tables["t5"]["Date"] = pd.to_datetime(tables["t5"]["Date"], errors="coerce")
    return tables


def save_fig(output_name: str) -> None:
    output_path = REPORT_DIR / output_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_01_aar_with_ci(daily_raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    tmp = daily_raw.copy()
    tmp["SuperGroup"] = np.where(
        tmp["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )
    tmp = (
        tmp.groupby(["Date", "SuperGroup"], as_index=False)
        .agg(
            n_stocks=("Ticker", "nunique"),
            mean_market_adjusted_return=("Market_Adjusted_Return", "mean"),
            std_market_adjusted_return=("Market_Adjusted_Return", "std"),
        )
        .sort_values(["SuperGroup", "Date"])
    )
    tmp["ci"] = 1.96 * tmp["std_market_adjusted_return"] / np.sqrt(tmp["n_stocks"])

    group_order = ["Treatment", "Control"]
    colors = {
        "Treatment": "#b22222",
        "Control": "#1f77b4",
    }

    for group in group_order:
        s = tmp[tmp["SuperGroup"] == group]
        y = s["mean_market_adjusted_return"] * 100
        ci = s["ci"] * 100
        ax.plot(s["Date"], y, marker="o", linewidth=2, label=group, color=colors[group])
        ax.fill_between(s["Date"], y - ci, y + ci, color=colors[group], alpha=0.15)

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(EVENT_X, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(
        "Part 1: Daily Market-Adjusted Return (Treatment vs Control, 95% CI)",
        fontweight="bold",
    )
    ax.set_ylabel("AAR (%)")
    ax.set_xlabel("Date")
    ax.legend()
    save_fig("R01_aar_with_ci.png")


def plot_02_caar_paths(daily_raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    tmp = daily_raw.copy()
    tmp["SuperGroup"] = np.where(
        tmp["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )
    tmp = (
        tmp.groupby(["Date", "SuperGroup"], as_index=False)["Market_Adjusted_Return"]
        .mean()
        .sort_values(["SuperGroup", "Date"])
    )
    tmp["CAAR"] = tmp.groupby("SuperGroup")["Market_Adjusted_Return"].transform(
        lambda s: (1 + s).cumprod() - 1
    )

    for group, s in tmp.groupby("SuperGroup"):
        ax.plot(s["Date"], s["CAAR"] * 100, marker="o", linewidth=2, label=group)

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(EVENT_X, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Part 1: CAAR Paths (Treatment vs Control)", fontweight="bold")
    ax.set_ylabel("CAAR (%)")
    ax.set_xlabel("Date")
    ax.legend()
    save_fig("R02_caar_paths.png")


def plot_03_treatment_control_diff(t5: pd.DataFrame) -> None:
    df = t5[t5["Scope"] == "ByDate"].copy().sort_values("Date")
    fig, ax = plt.subplots(figsize=(12, 6))

    diff_pct = df["Diff_Treat_minus_Control"] * 100
    bars = ax.bar(
        df["Date"],
        diff_pct,
        color=np.where(diff_pct < 0, "#b22222", "#2e8b57"),
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(EVENT_X, color="black", linestyle="--", linewidth=1.5)

    for bar, p in zip(bars, df["p_value"]):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (0.1 if y >= 0 else -0.1),
            f"p={p:.3f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=9,
        )

    ax.set_title("Part 1: Treatment - Control Daily AR Difference", fontweight="bold")
    ax.set_ylabel("Difference in AR (%)")
    ax.set_xlabel("Date")
    save_fig("R03_treatment_control_diff.png")


def plot_04_stock_car_distribution(t3: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    tmp = t3.copy()
    tmp["SuperGroup"] = np.where(
        tmp["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )
    tmp["CAR_pct"] = tmp["CAR_sum"] * 100

    sns.boxplot(
        data=tmp, x="SuperGroup", y="CAR_pct", order=["Treatment", "Control"], ax=ax
    )
    sns.stripplot(
        data=tmp,
        x="SuperGroup",
        y="CAR_pct",
        order=["Treatment", "Control"],
        color="black",
        alpha=0.45,
        size=4,
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(
        "Part 1: Stock-Level CAR Distribution (Treatment vs Control)", fontweight="bold"
    )
    ax.set_ylabel("CAR over Event Window (%)")
    ax.set_xlabel("")
    save_fig("R04_stock_car_distribution.png")


def plot_05_group_risk_return_map(t8: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    tmp = t8.copy()
    tmp["ret_pct"] = tmp["mean_monthly_return"] * 100
    tmp["vol_pct"] = tmp["std_monthly_return"] * 100

    markers = {"Before": "o", "After": "s"}
    colors = {
        "Treatment Big": "#b22222",
        "Treatment Small": "#ff8c00",
        "Control Big": "#1f77b4",
        "Control Small": "#2ca02c",
    }

    for _, row in tmp.iterrows():
        ax.scatter(
            row["vol_pct"],
            row["ret_pct"],
            s=130,
            marker=markers.get(row["Period"], "o"),
            color=colors.get(row["Group"], "gray"),
            edgecolor="black",
            alpha=0.9,
        )
        ax.text(
            row["vol_pct"] + 0.12,
            row["ret_pct"] + 0.05,
            f"{row['Group']} ({row['Period']})",
            fontsize=8,
        )

    ax.set_title("Part 2: Risk-Return Map (4 Groups, Monthly)", fontweight="bold")
    ax.set_xlabel("Std Dev of Monthly Return (%)")
    ax.set_ylabel("Mean Monthly Return (%)")
    save_fig("R05_group_risk_return_map.png")


def plot_06_beta_vs_return_stock_level(t6: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, period in zip(axes, ["Before", "After"]):
        s = t6[t6["Period"] == period].dropna(subset=["beta", "mean_monthly_return"])

        sns.scatterplot(
            data=s,
            x="beta",
            y="mean_monthly_return",
            hue="Group",
            style="Group",
            s=70,
            alpha=0.9,
            ax=ax,
            legend=(period == "After"),
        )

        if len(s) >= 3:
            x = s["beta"].to_numpy()
            y = s["mean_monthly_return"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.5)

        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(
            f"Part 2: {period} Beta vs Mean Return (Stock Level)", fontweight="bold"
        )
        ax.set_xlabel("Beta")
        ax.set_ylabel("Mean Monthly Return")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(
            handles=handles,
            labels=labels,
            title="Group",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    save_fig("R06_beta_vs_return_stock_level.png")


def plot_07_group_beta_return_relation(t7: pd.DataFrame, t8: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    tmp2 = t7.copy()
    tmp2["ret_pct"] = tmp2["mean_monthly_return"] * 100
    for _, row in tmp2.iterrows():
        axes[0].scatter(
            row["beta"],
            row["ret_pct"],
            s=120,
            c="#b22222" if row["TwoGroup"] == "Treatment" else "#1f77b4",
            marker="o" if row["Period"] == "Before" else "s",
            edgecolor="black",
            alpha=0.9,
        )
        axes[0].text(
            row["beta"] + 0.02,
            row["ret_pct"] + 0.03,
            f"{row['TwoGroup']} ({row['Period']})",
            fontsize=8,
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Part 2: Beta vs Return (2 Groups)", fontweight="bold")
    axes[0].set_xlabel("Beta")
    axes[0].set_ylabel("Mean Monthly Return (%)")

    tmp4 = t8.copy()
    tmp4["ret_pct"] = tmp4["mean_monthly_return"] * 100
    colors = {
        "Treatment Big": "#b22222",
        "Treatment Small": "#ff8c00",
        "Control Big": "#1f77b4",
        "Control Small": "#2ca02c",
    }
    for _, row in tmp4.iterrows():
        axes[1].scatter(
            row["beta"],
            row["ret_pct"],
            s=120,
            c=colors.get(row["Group"], "gray"),
            marker="o" if row["Period"] == "Before" else "s",
            edgecolor="black",
            alpha=0.9,
        )
        axes[1].text(
            row["beta"] + 0.02,
            row["ret_pct"] + 0.03,
            f"{row['Group']} ({row['Period']})",
            fontsize=7,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Part 2: Beta vs Return (4 Groups)", fontweight="bold")
    axes[1].set_xlabel("Beta")

    save_fig("R07_delta_mean_return_by_group.png")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading analysis tables...")
    tables = load_tables()

    print("Generating assignment-oriented evidence charts...")
    plot_01_aar_with_ci(tables["daily_raw"])
    plot_02_caar_paths(tables["daily_raw"])
    plot_03_treatment_control_diff(tables["t5"])
    plot_04_stock_car_distribution(tables["t3"])
    plot_05_group_risk_return_map(tables["t8"])
    plot_06_beta_vs_return_stock_level(tables["t6"])
    plot_07_group_beta_return_relation(tables["t7"], tables["t8"])

    print(f"All charts generated in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
