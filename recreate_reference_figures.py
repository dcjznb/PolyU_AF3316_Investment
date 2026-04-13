from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
TABLE_10 = (
    BASE_DIR
    / "event_study_data"
    / "analysis_results"
    / "table_10_monthly_before_after_change_by_stock.csv"
)
TABLE_01 = (
    BASE_DIR
    / "event_study_data"
    / "analysis_results"
    / "table_01_daily_aar_by_group_date.csv"
)
OUT_DIR = BASE_DIR / "event_study_data" / "visualizations" / "report"


def _percent_axis(x: float, _pos: float) -> str:
    return f"{x * 100:.2f}%"


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    tickers = df["Ticker"].to_list()
    x = np.arange(len(tickers))
    width = 0.28

    before_vals = df["mean_return_before"].to_numpy()
    after_vals = df["mean_return_after"].to_numpy()
    vals = np.concatenate([before_vals, after_vals])

    ax.bar(
        x - width / 2,
        before_vals,
        width,
        color="#E8EA00",
        label="AVG Before",
        edgecolor="none",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        after_vals,
        width,
        color="#FF6A00",
        label="AVG After",
        edgecolor="none",
        zorder=3,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=50, ha="right", rotation_mode="anchor")

    ax.set_facecolor("#f2f2f2")
    ax.grid(axis="y", color="#c6c6c6", linewidth=1.0, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    tick_step = 0.01
    y_low = float(np.floor((np.min(vals) - 0.004) / tick_step) * tick_step)
    y_high = float(np.ceil((np.max(vals) + 0.004) / tick_step) * tick_step)
    y_low = min(y_low, 0.0)
    y_high = max(y_high, 0.0)
    ax.set_ylim(y_low, y_high)
    ax.set_yticks(np.arange(y_low, y_high + tick_step / 2, tick_step))
    ax.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    ax.axhline(0, color="#bfbfbf", linewidth=1.0, zorder=2)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.4,
        columnspacing=1.2,
    )


def plot_figure_01() -> Path:
    df = pd.read_csv(TABLE_10)

    treatment = df[df["Group"].str.contains("Treatment", na=False)].copy()
    control = df[df["Group"].str.contains("Control", na=False)].copy()

    # Keep a stable order matching the source table.
    treatment["_order"] = np.arange(len(treatment))
    control["_order"] = np.arange(len(control))
    treatment = treatment.sort_values("_order")
    control = control.sort_values("_order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    _plot_panel(axes[0], treatment, "Average Return of Treatment Stocks")
    _plot_panel(axes[1], control, "Average Return of Control Stocks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_01_avg_return_before_after_by_stock.png"
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _plot_beta_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    tickers = df["Ticker"].to_list()
    x = np.arange(len(tickers))
    width = 0.26

    before_vals = df["beta_before"].to_numpy()
    after_vals = df["beta_after"].to_numpy()

    ax.bar(
        x - width / 2,
        before_vals,
        width,
        color="#E8EA00",
        label="Beta Before",
        edgecolor="none",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        after_vals,
        width,
        color="#FF6A00",
        label="Beta After",
        edgecolor="none",
        zorder=3,
    )

    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)

    ax.set_facecolor("#f2f2f2")
    ax.grid(axis="y", color="#c6c6c6", linewidth=1.0, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    max_beta = float(np.nanmax(np.concatenate([before_vals, after_vals])))
    y_top = max(1.5, np.ceil(max_beta / 0.2) * 0.2)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(np.arange(0.0, y_top + 1e-9, 0.2))
    ax.tick_params(axis="x", length=0, labelsize=10, pad=6)
    ax.tick_params(axis="y", length=0)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        handlelength=0.6,
        handletextpad=0.4,
        columnspacing=1.2,
    )


def plot_figure_02() -> Path:
    df = pd.read_csv(TABLE_10)

    treatment = df[df["Group"].str.contains("Treatment", na=False)].copy()
    control = df[df["Group"].str.contains("Control", na=False)].copy()

    treatment["_order"] = np.arange(len(treatment))
    control["_order"] = np.arange(len(control))
    treatment = treatment.sort_values("_order")
    control = control.sort_values("_order")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.9), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    _plot_beta_panel(axes[0], treatment, "Beta for Treatment Stocks")
    _plot_beta_panel(axes[1], control, "Beta for Control Stocks")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_02_beta_before_after_by_stock.png"
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _return_axis_4dp(x: float, _pos: float) -> str:
    return f"{x:.4f}"


def _beta_axis_4dp(x: float, _pos: float) -> str:
    return f"{x:.4f}"


def _trimmed_decimal_axis(x: float, _pos: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _annotate_nonoverlap_labels(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    labels: list[str],
) -> None:
    points = np.column_stack([x_vals, y_vals])
    trans = ax.transData
    inv = trans.inverted()
    anchor = trans.transform(points)

    base_offsets = np.array(
        [
            [10.0, 8.0],
            [10.0, -8.0],
            [-10.0, 8.0],
            [-10.0, -8.0],
            [14.0, 0.0],
            [-14.0, 0.0],
            [0.0, 11.0],
            [0.0, -11.0],
        ]
    )
    text_pos = anchor + np.array(
        [base_offsets[i % len(base_offsets)] for i in range(len(labels))]
    )

    widths = np.array([max(24.0, len(t) * 6.6 + 10.0) for t in labels])
    heights = np.full(len(labels), 14.0)

    bbox = ax.get_window_extent()
    x_min, x_max = bbox.x0 + 6.0, bbox.x1 - 6.0
    y_min, y_max = bbox.y0 + 6.0, bbox.y1 - 6.0

    preferred = anchor + np.array(
        [base_offsets[i % len(base_offsets)] for i in range(len(labels))]
    )

    for _ in range(220):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                dx = text_pos[i, 0] - text_pos[j, 0]
                dy = text_pos[i, 1] - text_pos[j, 1]
                overlap_x = (widths[i] + widths[j]) * 0.5 - abs(dx)
                overlap_y = (heights[i] + heights[j]) * 0.5 - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    sign_x = 1.0 if dx >= 0 else -1.0
                    sign_y = 1.0 if dy >= 0 else -1.0
                    if abs(dx) < 1.0:
                        sign_x = 1.0 if i % 2 == 0 else -1.0
                    if abs(dy) < 1.0:
                        sign_y = 1.0 if i % 2 == 0 else -1.0

                    push_x = sign_x * overlap_x * 0.26
                    push_y = sign_y * overlap_y * 0.26
                    text_pos[i, 0] += push_x
                    text_pos[j, 0] -= push_x
                    text_pos[i, 1] += push_y
                    text_pos[j, 1] -= push_y
                    moved = True

        # Keep labels near anchor while allowing repulsion.
        text_pos += (preferred - text_pos) * 0.055

        text_pos[:, 0] = np.clip(text_pos[:, 0], x_min, x_max)
        text_pos[:, 1] = np.clip(text_pos[:, 1], y_min, y_max)

        if not moved:
            break

    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        tx, ty = inv.transform(text_pos[i])
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=8,
            color="#666666",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.72,
            },
            arrowprops={"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.6, "alpha": 0.8},
            zorder=4,
        )


def _plot_beta_return_panel(
    ax: plt.Axes,
    beta: np.ndarray,
    er: np.ndarray,
    labels: list[str],
    title: str,
    er_label: str,
) -> None:
    slope_er, intercept_er = np.polyfit(beta, er, 1)
    capm = intercept_er + slope_er * beta
    slope_capm, intercept_capm = np.polyfit(beta, capm, 1)

    ax.scatter(beta, er, s=36, color="#0B789B", label=er_label, zorder=3)
    ax.scatter(beta, capm, s=30, color="#FF6A00", label="CAPM", zorder=2)

    x_line = np.linspace(float(np.min(beta)), float(np.max(beta)), 120)
    ax.plot(
        x_line,
        intercept_er + slope_er * x_line,
        color="#5BA9C7",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.2,
        label=f"Linear ({er_label})",
        zorder=1,
    )
    ax.plot(
        x_line,
        intercept_capm + slope_capm * x_line,
        color="#FF9A66",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.2,
        label="Linear (CAPM)",
        zorder=1,
    )

    # Mark the orange fitted CAPM trend as the SML line.
    x_sml = float(x_line[int(len(x_line) * 0.8)])
    y_sml = float(intercept_capm + slope_capm * x_sml)
    ax.annotate(
        "SML Line",
        xy=(x_sml, y_sml),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=8,
        color="#FF8A4C",
        fontweight="bold",
    )

    x_min = float(np.min(beta))
    x_max = float(np.max(beta))
    x_pad = max(0.04, (x_max - x_min) * 0.12)

    y_min = float(np.min(np.concatenate([er, capm])))
    y_max = float(np.max(np.concatenate([er, capm])))
    y_pad = max(0.012, (y_max - y_min) * 0.24)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Beta")
    ax.set_ylabel("E(r)")

    ax.set_facecolor("#f2f2f2")
    ax.grid(color="#c6c6c6", linewidth=1.0, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c6c6c6")

    ax.xaxis.set_major_formatter(FuncFormatter(_beta_axis_4dp))
    ax.yaxis.set_major_formatter(FuncFormatter(_return_axis_4dp))
    ax.tick_params(axis="both", length=0)

    _annotate_nonoverlap_labels(ax, beta, er, labels)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.4,
        fontsize=10,
    )


def plot_figure_03() -> Path:
    df = pd.read_csv(TABLE_10).copy()

    beta_before = df["beta_before"].to_numpy(dtype=float)
    beta_after = df["beta_after"].to_numpy(dtype=float)
    er_before = df["mean_return_before"].to_numpy(dtype=float)
    er_after = df["mean_return_after"].to_numpy(dtype=float)
    labels = df["Ticker"].to_list()

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    _plot_beta_return_panel(
        axes[0],
        beta_before,
        er_before,
        labels,
        "Treatment & Control Before the Event",
        "E(r)",
    )
    _plot_beta_return_panel(
        axes[1],
        beta_after,
        er_after,
        labels,
        "Treatment & Control After the Event",
        "E(r)",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_03_beta_vs_expected_return_before_after.png"
    fig.tight_layout(w_pad=2.8)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def plot_figure_04() -> Path:
    df = pd.read_csv(TABLE_01)
    df["Date"] = pd.to_datetime(df["Date"])
    df["SuperGroup"] = np.where(
        df["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )

    df["weighted_return_sum"] = df["mean_daily_return"] * df["n_stocks"]
    grouped = (
        df.groupby(["Date", "SuperGroup"], as_index=False)
        .agg(
            weighted_return_sum=("weighted_return_sum", "sum"),
            total_n=("n_stocks", "sum"),
        )
        .assign(weighted_mean_return=lambda g: g["weighted_return_sum"] / g["total_n"])
    )

    pivot = (
        grouped.pivot(index="Date", columns="SuperGroup", values="weighted_mean_return")
        .sort_index()
        .fillna(0.0)
    )

    x = np.arange(len(pivot.index))
    width = 0.26

    treatment = pivot["Treatment"].to_numpy()
    control = pivot["Control"].to_numpy()
    x_labels = [f"{d.day}-{d.strftime('%b')}" for d in pivot.index]

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    fig.patch.set_facecolor("#f2f2f2")

    ax.bar(
        x - width / 2,
        treatment,
        width,
        color="#006C8F",
        label="AVG-Treatment",
        edgecolor="none",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        control,
        width,
        color="#FF6A00",
        label="AVG-Control",
        edgecolor="none",
        zorder=3,
    )

    ax.set_facecolor("#f2f2f2")
    ax.grid(axis="y", color="#c6c6c6", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Average Return of Both Groups", fontsize=20, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylim(-0.055, 0.012)
    ax.set_yticks(np.arange(-0.05, 0.011, 0.01))
    ax.yaxis.set_major_formatter(FuncFormatter(_trimmed_decimal_axis))
    ax.tick_params(axis="x", length=4, color="#d0d0d0", labelsize=16, pad=6)
    ax.tick_params(axis="y", length=0, labelsize=16)
    ax.axhline(0, color="#c6c6c6", linewidth=1.2, zorder=1)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        frameon=False,
        fontsize=15,
        handlelength=0.6,
        handletextpad=0.3,
        columnspacing=1.4,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_04_avg_return_both_groups_by_date.png"
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.30)
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _plot_group_period_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    beta_col: str,
    er_col: str,
    title: str,
) -> float:
    beta = df[beta_col].to_numpy(dtype=float)
    er = df[er_col].to_numpy(dtype=float)
    labels = df["Ticker"].to_list()

    # CAPM proxy: zero-risk-free approximation using period cross-sectional market premium.
    market_premium = float(np.mean(er) / np.mean(beta)) if np.mean(beta) != 0 else 0.0
    capm = beta * market_premium

    slope_er, intercept_er = np.polyfit(beta, er, 1)
    slope_capm, intercept_capm = np.polyfit(beta, capm, 1)

    ax.scatter(beta, er, s=30, color="#0B789B", label="E(r)", zorder=3)
    ax.scatter(beta, capm, s=26, color="#FF6A00", label="CAPM", zorder=2)

    x_line = np.linspace(float(np.min(beta)), float(np.max(beta)), 120)
    ax.plot(
        x_line,
        intercept_er + slope_er * x_line,
        color="#2D8DB5",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.0,
        label="Linear (E(r))",
        zorder=1,
    )
    ax.plot(
        x_line,
        intercept_capm + slope_capm * x_line,
        color="#FF8A4C",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.0,
        label="Linear (CAPM)",
        zorder=1,
    )

    # Mark the orange fitted CAPM trend as the SML line.
    x_sml = float(x_line[int(len(x_line) * 0.8)])
    y_sml = float(intercept_capm + slope_capm * x_sml)
    ax.annotate(
        "SML Line",
        xy=(x_sml, y_sml),
        xytext=(8, -8),
        textcoords="offset points",
        fontsize=8,
        color="#FF8A4C",
        fontweight="bold",
    )

    for x, y, t in zip(beta, er, labels):
        ax.annotate(
            t,
            xy=(x, y),
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=7,
            color="#666666",
        )

    x_min = float(np.min(beta))
    x_max = float(np.max(beta))
    x_pad = max(0.02, (x_max - x_min) * 0.06)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    y_all = np.concatenate([er, capm])
    y_min_raw = float(np.min(y_all))
    y_max_raw = float(np.max(y_all))
    y_pad = max(0.006, (y_max_raw - y_min_raw) * 0.12)
    y_min = y_min_raw - y_pad
    y_max = y_max_raw + y_pad
    ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Beta")
    ax.set_ylabel("E(r)")
    ax.set_facecolor("#f2f2f2")
    ax.grid(color="#c6c6c6", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c6c6c6")

    ax.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax.xaxis.set_major_formatter(FuncFormatter(_beta_axis_4dp))
    ax.tick_params(axis="both", length=0, labelsize=8)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
        fontsize=8,
    )

    ss_res = float(np.sum((er - capm) ** 2))
    ss_tot = float(np.sum((er - np.mean(er)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return r_squared


def _add_rsquare_row(ax: plt.Axes, left_value: float, right_value: float) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=[["R-Square", f"{left_value:.6f}", f"{right_value:.6f}"]],
        cellLoc="left",
        loc="center",
        colWidths=[0.12, 0.44, 0.44],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.6)

    for _, cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#444444")
        cell.set_facecolor("#f2f2f2")
        cell.PAD = 0.08


def plot_figure_05() -> Path:
    df = pd.read_csv(TABLE_10).copy()
    df["SuperGroup"] = np.where(
        df["Group"].str.contains("Treatment", na=False), "Treatment", "Control"
    )

    treatment = df[df["SuperGroup"] == "Treatment"].copy()
    control = df[df["SuperGroup"] == "Control"].copy()

    fig = plt.figure(figsize=(16.5, 12), facecolor="#f2f2f2")
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[13, 1.2, 13, 1.2],
        hspace=0.34,
        wspace=0.08,
    )

    ax_tb = fig.add_subplot(gs[0, 0])
    ax_ta = fig.add_subplot(gs[0, 1])
    ax_t_tbl = fig.add_subplot(gs[1, :])

    ax_cb = fig.add_subplot(gs[2, 0])
    ax_ca = fig.add_subplot(gs[2, 1])
    ax_c_tbl = fig.add_subplot(gs[3, :])

    r2_tb = _plot_group_period_scatter(
        ax_tb,
        treatment,
        "beta_before",
        "mean_return_before",
        "Treatment Before",
    )
    r2_ta = _plot_group_period_scatter(
        ax_ta,
        treatment,
        "beta_after",
        "mean_return_after",
        "Treatment After",
    )
    _add_rsquare_row(ax_t_tbl, r2_tb, r2_ta)

    r2_cb = _plot_group_period_scatter(
        ax_cb,
        control,
        "beta_before",
        "mean_return_before",
        "Control Before",
    )
    r2_ca = _plot_group_period_scatter(
        ax_ca,
        control,
        "beta_after",
        "mean_return_after",
        "Control After",
    )
    _add_rsquare_row(ax_c_tbl, r2_cb, r2_ca)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_05_four_panel_group_before_after_scatter.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _line_axis_fmt(x: float, _pos: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".")


def plot_figure_06() -> Path:
    df = pd.read_csv(TABLE_01)
    df["Date"] = pd.to_datetime(df["Date"])

    pivot = df.pivot_table(
        index="Date", columns="Group", values="mean_daily_return", aggfunc="mean"
    ).sort_index()

    dates = pivot.index.to_list()
    x_labels = [f"{d.day}-{d.strftime('%b')}" for d in dates]
    x = np.arange(len(dates))

    t_big = pivot["Treatment Big"].to_numpy(dtype=float)
    t_small = pivot["Treatment Small"].to_numpy(dtype=float)
    c_big = pivot["Control Big"].to_numpy(dtype=float)
    c_small = pivot["Control Small"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    ax_l, ax_r = axes

    ax_l.plot(
        x, t_big, color="#0B789B", linewidth=2.2, label="Average Return (Treatment Big)"
    )
    ax_l.plot(
        x,
        t_small,
        color="#E77732",
        linewidth=2.2,
        label="Average Return (Treatment Small)",
    )
    ax_l.set_title("Return of treatment groups", fontsize=17, pad=10)

    ax_r.plot(
        x, c_big, color="#0B789B", linewidth=2.2, label="Average Return (Control Big)"
    )
    ax_r.plot(
        x,
        c_small,
        color="#E77732",
        linewidth=2.2,
        label="Average Return (Control Small)",
    )
    ax_r.set_title("Return of control groups", fontsize=17, pad=10)

    for ax in axes:
        ax.set_facecolor("#f2f2f2")
        ax.grid(axis="y", color="#c6c6c6", linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0, color="#c6c6c6", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=40, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_formatter(FuncFormatter(_line_axis_fmt))
        ax.tick_params(axis="x", length=4, color="#d0d0d0", labelsize=14)
        ax.tick_params(axis="y", length=0, labelsize=13)

    y_min = float(np.min(np.concatenate([t_big, t_small, c_big, c_small])))
    y_max = float(np.max(np.concatenate([t_big, t_small, c_big, c_small])))
    pad = max(0.01, (y_max - y_min) * 0.18)
    step = 0.02
    low = float(np.floor((y_min - pad) / step) * step)
    high = float(np.ceil((y_max + pad) / step) * step)
    low = min(low, -0.02)
    high = max(high, 0.02)
    ax_l.set_ylim(low, high)
    ax_r.set_ylim(low, high)
    ax_l.set_yticks(np.arange(low, high + step / 2, step))
    ax_r.set_yticks(np.arange(low, high + step / 2, step))

    ax_l.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=11,
        handlelength=2.8,
        handletextpad=0.4,
        labelspacing=0.45,
    )
    ax_r.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=11,
        handlelength=2.8,
        handletextpad=0.4,
        labelspacing=0.45,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.12)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_06_line_return_by_group_big_small.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _plot_treatment_big_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    beta_col: str,
    er_col: str,
    title: str,
) -> None:
    beta = df[beta_col].to_numpy(dtype=float)
    er = df[er_col].to_numpy(dtype=float)
    labels = df["Ticker"].to_list()

    market_premium = float(np.mean(er) / np.mean(beta)) if np.mean(beta) != 0 else 0.0
    capm = beta * market_premium

    slope_er, intercept_er = np.polyfit(beta, er, 1)
    slope_capm, intercept_capm = np.polyfit(beta, capm, 1)

    ax.scatter(beta, er, s=42, color="#0B789B", label="E(r)", zorder=3)
    ax.scatter(beta, capm, s=46, color="#FF6A00", label="CAPM", zorder=2)

    x_line = np.linspace(float(np.min(beta)), float(np.max(beta)), 120)
    ax.plot(
        x_line,
        intercept_er + slope_er * x_line,
        color="#1886AD",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.6,
        label="Linear (E(r))",
        zorder=1,
    )
    ax.plot(
        x_line,
        intercept_capm + slope_capm * x_line,
        color="#FF8A4C",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.6,
        label="Linear (CAPM)",
        zorder=1,
    )

    x_sml = float(x_line[int(len(x_line) * 0.8)])
    y_sml = float(intercept_capm + slope_capm * x_sml)
    ax.annotate(
        "SML Line",
        xy=(x_sml, y_sml),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=10,
        color="#FF8A4C",
        fontweight="bold",
    )

    for x, y, t in zip(beta, er, labels):
        ax.annotate(
            t,
            xy=(x, y),
            xytext=(0, -10) if y < 0 else (0, 8),
            textcoords="offset points",
            fontsize=10,
            color="#0B3470",
            fontweight="bold",
            ha="center",
        )

    x_low = float(np.floor((np.min(beta) - 0.05) / 0.1) * 0.1)
    x_high = float(np.ceil((np.max(beta) + 0.05) / 0.1) * 0.1)
    y_low = float(np.floor((min(np.min(er), np.min(capm)) - 0.01) / 0.01) * 0.01)
    y_high = float(np.ceil((max(np.max(er), np.max(capm)) + 0.01) / 0.01) * 0.01)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_title(title, fontsize=19, pad=20)
    ax.set_xlabel("Beta", fontsize=14)
    ax.set_ylabel("E(r)", fontsize=14)

    ax.set_facecolor("#f2f2f2")
    ax.grid(color="#c6c6c6", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c6c6c6")

    ax.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax.xaxis.set_major_formatter(FuncFormatter(_beta_axis_4dp))
    ax.tick_params(axis="both", length=0, labelsize=11)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=4,
        frameon=False,
        fontsize=13,
        handlelength=3.0,
        columnspacing=1.5,
        handletextpad=0.45,
    )


def plot_figure_07() -> Path:
    df = pd.read_csv(TABLE_10)
    treatment_big = df[df["Group"] == "Treatment Big"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    _plot_treatment_big_panel(
        axes[0],
        treatment_big,
        "beta_before",
        "mean_return_before",
        "Treatment Big before event",
    )
    _plot_treatment_big_panel(
        axes[1],
        treatment_big,
        "beta_after",
        "mean_return_after",
        "Treatment Big after event",
    )

    fig.text(
        0.02,
        0.96,
        "Treatment Big Group",
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        family="serif",
    )

    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5],
            [0.12, 0.80],
            transform=fig.transFigure,
            color="#c6c6c6",
            linewidth=2.0,
        )
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.68, bottom=0.24, wspace=0.34)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_07_treatment_big_before_after_scatter.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def _plot_treatment_small_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    beta_col: str,
    er_col: str,
    title: str,
) -> None:
    beta = df[beta_col].to_numpy(dtype=float)
    er = df[er_col].to_numpy(dtype=float)
    labels = df["Ticker"].to_list()

    market_premium = float(np.mean(er) / np.mean(beta)) if np.mean(beta) != 0 else 0.0
    capm = beta * market_premium

    slope_er, intercept_er = np.polyfit(beta, er, 1)
    slope_capm, intercept_capm = np.polyfit(beta, capm, 1)

    ax.scatter(beta, er, s=42, color="#0B789B", label="E(r)", zorder=3)
    ax.scatter(beta, capm, s=46, color="#FF6A00", label="CAPM", zorder=2)

    x_line = np.linspace(float(np.min(beta)), float(np.max(beta)), 120)
    ax.plot(
        x_line,
        intercept_er + slope_er * x_line,
        color="#1886AD",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.6,
        label="Linear (E(r))",
        zorder=1,
    )
    ax.plot(
        x_line,
        intercept_capm + slope_capm * x_line,
        color="#FF8A4C",
        linestyle=(0, (1.2, 1.2)),
        linewidth=1.6,
        label="Linear (CAPM)",
        zorder=1,
    )

    x_sml = float(x_line[int(len(x_line) * 0.8)])
    y_sml = float(intercept_capm + slope_capm * x_sml)
    ax.annotate(
        "SML Line",
        xy=(x_sml, y_sml),
        xytext=(8, -10),
        textcoords="offset points",
        fontsize=10,
        color="#FF8A4C",
        fontweight="bold",
    )

    for x, y, t in zip(beta, er, labels):
        ax.annotate(
            t,
            xy=(x, y),
            xytext=(0, -10) if y < 0 else (0, 8),
            textcoords="offset points",
            fontsize=10,
            color="#0B3470",
            fontweight="bold",
            ha="center",
        )

    x_low = float(np.floor((np.min(beta) - 0.05) / 0.1) * 0.1)
    x_high = float(np.ceil((np.max(beta) + 0.05) / 0.1) * 0.1)
    y_low = float(np.floor((min(np.min(er), np.min(capm)) - 0.01) / 0.01) * 0.01)
    y_high = float(np.ceil((max(np.max(er), np.max(capm)) + 0.01) / 0.01) * 0.01)

    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.set_title(title, fontsize=19, pad=20)
    ax.set_xlabel("Beta", fontsize=14)
    ax.set_ylabel("E(r)", fontsize=14)

    ax.set_facecolor("#f2f2f2")
    ax.grid(color="#c6c6c6", linewidth=1.2, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#c6c6c6")

    ax.yaxis.set_major_formatter(FuncFormatter(_percent_axis))
    ax.xaxis.set_major_formatter(FuncFormatter(_beta_axis_4dp))
    ax.tick_params(axis="both", length=0, labelsize=11)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=4,
        frameon=False,
        fontsize=13,
        handlelength=3.0,
        columnspacing=1.5,
        handletextpad=0.45,
    )


def plot_figure_08() -> Path:
    df = pd.read_csv(TABLE_10)
    treatment_small = df[df["Group"] == "Treatment Small"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), sharey=False)
    fig.patch.set_facecolor("#f2f2f2")

    _plot_treatment_small_panel(
        axes[0],
        treatment_small,
        "beta_before",
        "mean_return_before",
        "Treatment small before event",
    )
    _plot_treatment_small_panel(
        axes[1],
        treatment_small,
        "beta_after",
        "mean_return_after",
        "Treatment small after event",
    )

    fig.text(
        0.02,
        0.96,
        "Treatment Small Group",
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        family="serif",
    )

    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5],
            [0.12, 0.80],
            transform=fig.transFigure,
            color="#c6c6c6",
            linewidth=2.0,
        )
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.68, bottom=0.24, wspace=0.34)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "Figure_08_treatment_small_before_after_scatter.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_file


def main() -> None:
    out_file_01 = plot_figure_01()
    out_file_02 = plot_figure_02()
    out_file_03 = plot_figure_03()
    out_file_04 = plot_figure_04()
    out_file_05 = plot_figure_05()
    out_file_06 = plot_figure_06()
    out_file_07 = plot_figure_07()
    out_file_08 = plot_figure_08()
    print(f"Saved: {out_file_01}")
    print(f"Saved: {out_file_02}")
    print(f"Saved: {out_file_03}")
    print(f"Saved: {out_file_04}")
    print(f"Saved: {out_file_05}")
    print(f"Saved: {out_file_06}")
    print(f"Saved: {out_file_07}")
    print(f"Saved: {out_file_08}")


if __name__ == "__main__":
    main()
