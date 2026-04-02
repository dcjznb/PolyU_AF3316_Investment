from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_data() -> pd.DataFrame:
    """Load the event window daily data."""
    data_path = (
        Path(__file__).resolve().parent
        / "event_study_data"
        / "event_window_daily_2023-03-08_to_2023-03-12.csv"
    )
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def plot_daily_returns_by_group(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot daily returns by group."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Daily Stock Returns Around SVB Collapse Event (2023-03-10)",
        fontsize=16,
        fontweight="bold",
    )

    groups = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for idx, (ax, group, color) in enumerate(zip(axes.flat, groups, colors)):
        group_data = df[df["Group"] == group].sort_values("Date")

        ax.plot(
            group_data["Date"],
            group_data["Daily_Return"] * 100,
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
            label="Daily Return",
        )
        ax.axvline(
            x=pd.Timestamp("2023-03-10"),
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Event Date",
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        ax.set_title(
            f"{group}\n(Avg: {group_data['Daily_Return'].mean()*100:.2f}%)",
            fontweight="bold",
        )
        ax.set_ylabel("Daily Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Format x-axis
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = output_dir / "01_daily_returns_by_group.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_market_adjusted_returns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot market-adjusted returns by group."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Market-Adjusted Returns Around SVB Collapse Event",
        fontsize=16,
        fontweight="bold",
    )

    groups = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for idx, (ax, group, color) in enumerate(zip(axes.flat, groups, colors)):
        group_data = df[df["Group"] == group].sort_values("Date")

        ax.plot(
            group_data["Date"],
            group_data["Market_Adjusted_Return"] * 100,
            marker="s",
            linewidth=2,
            markersize=8,
            color=color,
            label="Market-Adj Return",
        )
        ax.axvline(
            x=pd.Timestamp("2023-03-10"),
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Event Date",
        )
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        ax.set_title(
            f"{group}\n(Avg: {group_data['Market_Adjusted_Return'].mean()*100:.2f}%)",
            fontweight="bold",
        )
        ax.set_ylabel("Market-Adjusted Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Format x-axis
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    output_path = output_dir / "02_market_adjusted_returns_by_group.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_group_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot overall group comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Group Comparison: Average Returns Around Event", fontsize=16, fontweight="bold"
    )

    groups_order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    colors_map = {
        "Treatment Big": "#d62728",
        "Treatment Small": "#ff7f0e",
        "Control Big": "#2ca02c",
        "Control Small": "#1f77b4",
    }

    # Average Daily Return
    avg_daily = df.groupby("Group")["Daily_Return"].mean().reindex(groups_order) * 100
    ax1 = axes[0]
    bars1 = ax1.bar(
        range(len(avg_daily)),
        avg_daily,
        color=[colors_map[g] for g in groups_order],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_ylabel("Average Daily Return (%)", fontweight="bold")
    ax1.set_title("Daily Returns", fontweight="bold")
    ax1.set_xticks(range(len(avg_daily)))
    ax1.set_xticklabels(groups_order, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    # Average Market-Adjusted Return
    avg_adj = (
        df.groupby("Group")["Market_Adjusted_Return"].mean().reindex(groups_order) * 100
    )
    ax2 = axes[1]
    bars2 = ax2.bar(
        range(len(avg_adj)),
        avg_adj,
        color=[colors_map[g] for g in groups_order],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_ylabel("Average Market-Adjusted Return (%)", fontweight="bold")
    ax2.set_title("Market-Adjusted Returns", fontweight="bold")
    ax2.set_xticks(range(len(avg_adj)))
    ax2.set_xticklabels(groups_order, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = output_dir / "03_group_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of market-adjusted returns by date and group."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Pivot for heatmap
    pivot_data = (
        df.pivot_table(
            values="Market_Adjusted_Return",
            index="Group",
            columns="Date",
            aggfunc="mean",
        )
        * 100
    )

    # Reorder rows
    group_order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    pivot_data = pivot_data.reindex(group_order)

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Market-Adjusted Return (%)"},
        ax=ax,
        linewidths=1,
    )
    ax.set_title(
        "Market-Adjusted Returns by Group and Date\n(Negative = Underperformance vs Market)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Group", fontweight="bold")

    plt.tight_layout()
    output_path = output_dir / "04_heatmap_market_adjusted.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_individual_stocks_treatment_vs_control(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """Plot treatment vs control group stock comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Individual Stock Returns: Treatment vs Control by Exposure",
        fontsize=16,
        fontweight="bold",
    )

    # Treatment group
    treatment_data = df[df["Group"].str.contains("Treatment")]
    treatment_pivot = (
        treatment_data.pivot_table(
            values="Market_Adjusted_Return",
            index="Ticker",
            columns="Date",
            aggfunc="mean",
        )
        * 100
    )

    ax1 = axes[0]
    sns.heatmap(
        treatment_pivot.sort_values(treatment_pivot.columns[-1]),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Return (%)"},
        ax=ax1,
        linewidths=0.5,
    )
    ax1.set_title("Treatment Group (High-Exposure Banks)", fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Ticker")

    # Control group
    control_data = df[df["Group"].str.contains("Control")]
    control_pivot = (
        control_data.pivot_table(
            values="Market_Adjusted_Return",
            index="Ticker",
            columns="Date",
            aggfunc="mean",
        )
        * 100
    )

    ax2 = axes[1]
    sns.heatmap(
        control_pivot.sort_values(control_pivot.columns[-1]),
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Return (%)"},
        ax=ax2,
        linewidths=0.5,
    )
    ax2.set_title("Control Group (Low-Exposure / Large Financials)", fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Ticker")

    plt.tight_layout()
    output_path = output_dir / "05_individual_stocks_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_cumulative_returns(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot cumulative returns over the event window."""
    fig, ax = plt.subplots(figsize=(14, 7))

    groups_order = ["Treatment Big", "Treatment Small", "Control Big", "Control Small"]
    colors_map = {
        "Treatment Big": "#d62728",
        "Treatment Small": "#ff7f0e",
        "Control Big": "#2ca02c",
        "Control Small": "#1f77b4",
    }

    for group in groups_order:
        group_data = df[df["Group"] == group].sort_values("Date")
        cumulative_return = (1 + group_data["Market_Adjusted_Return"]).cumprod() - 1
        ax.plot(
            group_data["Date"],
            cumulative_return * 100,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=group,
            color=colors_map[group],
        )

    ax.axvline(
        x=pd.Timestamp("2023-03-10"),
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Event Date (SVB Collapse)",
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Date", fontweight="bold", fontsize=12)
    ax.set_ylabel(
        "Cumulative Market-Adjusted Return (%)", fontweight="bold", fontsize=12
    )
    ax.set_title(
        "Cumulative Market-Adjusted Returns Around SVB Collapse Event\n(Negative = Greater Underperformance)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_path = output_dir / "06_cumulative_returns.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {output_path}")
    plt.close()


def main() -> None:
    print("Loading event analysis data...")
    df = load_data()

    output_dir = Path(__file__).resolve().parent / "event_study_data" / "visualizations"
    output_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...\n")

    plot_daily_returns_by_group(df, output_dir)
    plot_market_adjusted_returns(df, output_dir)
    plot_group_comparison(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_individual_stocks_treatment_vs_control(df, output_dir)
    plot_cumulative_returns(df, output_dir)

    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
