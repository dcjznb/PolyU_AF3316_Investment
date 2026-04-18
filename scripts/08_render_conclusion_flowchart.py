from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


try:
    font_manager.findfont("Noto Sans", fallback_to_default=False)
    plt.rcParams["font.family"] = "Noto Sans"
    plt.rcParams["font.sans-serif"] = ["Noto Sans", "DejaVu Sans"]
except Exception:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["font.size"] = 10


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "visualizations" / "report"


def _box(ax, x, y, w, h, text, facecolor, edgecolor, textcolor="#111111"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.6,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=14,
        color=textcolor,
        fontweight="semibold",
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, color="#4b5563"):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color=color,
    )
    ax.add_patch(arrow)


def build_conclusion_flowchart(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "How the SELECT announcement reshaped valuation",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.5,
        0.885,
        "A concise causal path from event shock to longer-run repricing",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#555555",
    )

    top_y = 0.67
    box_w = 0.18
    box_h = 0.16
    gap = 0.055
    xs = [0.05, 0.05 + box_w + gap, 0.05 + 2 * (box_w + gap), 0.05 + 3 * (box_w + gap)]

    box_specs = [
        (
            "SELECT\nannouncement",
            "#ffffff",
            "#111111",
            "#111111",
        ),
        (
            "Immediate repricing\nof obesity-exposed\nmedical device stocks",
            "#f8f8f8",
            "#333333",
            "#111111",
        ),
        (
            "Post-event pattern\nlooks persistent, not\njust temporary liquidity",
            "#f8f8f8",
            "#333333",
            "#111111",
        ),
        (
            "Beta explains less\nof cross-sectional\nreturns over time",
            "#ffffff",
            "#111111",
            "#111111",
        ),
    ]

    for x, (text, face, edge, tcolor) in zip(xs, box_specs):
        _box(ax, x, top_y, box_w, box_h, text, face, edge, tcolor)

    for i in range(3):
        _arrow(
            ax,
            xs[i] + box_w,
            top_y + box_h / 2,
            xs[i + 1],
            top_y + box_h / 2,
        )

    bottom_y = 0.34
    _box(
        ax,
        0.17,
        bottom_y,
        0.66,
        0.16,
        "Event-specific exposure became more important than standard market risk\nfor treatment stocks",
        "#f2f2f2",
        "#111111",
        "#111111",
    )

    _arrow(ax, 0.5, top_y, 0.5, bottom_y + 0.16, color="#111111")

    ax.text(
        0.5,
        0.16,
        "Overall: GLP-1 developments appear to have reset\nexpectations of future demand, reshaping healthcare valuation.",
        ha="center",
        va="center",
        fontsize=14.5,
        color="#111111",
        fontweight="semibold",
        wrap=True,
    )

    png_path = output_dir / "flowchart_conclusion.png"
    pdf_path = output_dir / "flowchart_conclusion.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = build_conclusion_flowchart()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
