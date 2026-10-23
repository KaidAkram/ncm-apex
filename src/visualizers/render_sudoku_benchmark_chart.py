"""
NCM-Apex: Sudoku Three-Paradigm Benchmark Visualizer (Bento-Box Dark Aesthetic)
Produces publication-grade graphics (4K 16:9 PNG + SVG):
- assets/diagrams/sudoku_benchmark_results.png
- assets/diagrams/sudoku_benchmark_results.svg
Compares Solved Boards (%) and NFE across Easy, Medium, and Hard tiers:
AR vs. MDLM vs. DUO vs. FMLM+ vs. NCM (Ours).
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

COLOR_BG = "#080B10"
COLOR_CARD = "#0F141F"
COLOR_BORDER = "#1E293B"
COLOR_TEXT_WHITE = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_CYAN = "#00F0FF"
COLOR_ROSE = "#FF2A85"
COLOR_AMBER = "#F59E0B"
COLOR_PURPLE = "#A855F7"
COLOR_TEAL = "#14B8A6"
COLOR_GREEN = "#10B981"


def render_sudoku_benchmark_chart(output_dir: str = "assets/diagrams"):
    """
    Renders the 4K Bento-box comparison chart of Sudoku solve rates across paradigms.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(26, 14), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.0], wspace=0.18, left=0.05, right=0.95, top=0.88, bottom=0.10)

    # Main Title
    plt.figtext(
        0.05, 0.95,
        "NCM-Apex: Non-Autoregressive Spatial Reasoning on Sudoku",
        fontsize=28, fontweight="bold", color=COLOR_TEXT_WHITE, ha="left",
    )
    plt.figtext(
        0.05, 0.915,
        "Paper Table 3 Benchmark: Next-Concept Modeling (NCM) vs. Autoregressive (AR) & Diffusion Models across Easy, Medium, Hard splits",
        fontsize=15, color=COLOR_CYAN, ha="left",
    )

    # ================= LEFT: GROUPED BAR CHART =================
    ax_bar = fig.add_subplot(gs[0, 0], facecolor=COLOR_CARD)

    # Categories & Data
    tiers = ["Easy (40 clues)", "Medium (35 clues)", "Hard (30 clues)"]
    models = [
        ("Autoregressive (AR)", [14.6, 5.1, 1.0], COLOR_ROSE, "128 NFE (Sequential Collapse)"),
        ("MDLM (Diffusion)", [77.1, 45.9, 30.2], COLOR_AMBER, "128 NFE (Full Diffusion)"),
        ("DUO (Diffusion)", [84.0, 58.7, 42.1], COLOR_PURPLE, "128 NFE (Iterative Denoising)"),
        ("FMLM+ (Diffusion)", [88.5, 64.2, 50.8], COLOR_TEAL, "128 NFE (Flow Matching)"),
        ("NCM (Ours)", [99.5, 96.5, 86.1], COLOR_CYAN, "12 NFE (Coarse-to-Fine Parallel)"),
    ]

    x = np.arange(len(tiers))
    total_models = len(models)
    bar_width = 0.15
    offsets = np.linspace(-bar_width * 2, bar_width * 2, total_models)

    for i, (name, values, color, tag) in enumerate(models):
        pos = x + offsets[i]
        is_ncm = ("NCM" in name)
        edge_col = "#FFFFFF" if is_ncm else color
        lw = 2.0 if is_ncm else 1.0

        bars = ax_bar.bar(
            pos, values,
            width=bar_width * 0.92,
            label=name,
            color=color,
            edgecolor=edge_col,
            linewidth=lw,
            alpha=0.95 if is_ncm else 0.85,
            zorder=3,
        )

        # Labels on top of bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height() + 1.5
            fontweight = "bold" if is_ncm else "normal"
            text_color = COLOR_CYAN if is_ncm else COLOR_TEXT_MUTED
            font_size = 11 if is_ncm else 9.5
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=font_size, fontweight=fontweight, color=text_color,
            )

    # Axis formatting
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(tiers, fontsize=15, fontweight="bold", color=COLOR_TEXT_WHITE)
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Solved Boards (%)", fontsize=15, fontweight="bold", color=COLOR_TEXT_WHITE, labelpad=12)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25, color=COLOR_BORDER, zorder=0)

    for spine in ax_bar.spines.values():
        spine.set_color(COLOR_BORDER)
        spine.set_linewidth(1.2)

    legend = ax_bar.legend(
        loc="upper right",
        frameon=True,
        facecolor=COLOR_BG,
        edgecolor=COLOR_BORDER,
        fontsize=12,
        ncol=2,
    )
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT_WHITE)

    # ================= RIGHT: TELEMETRY HUD & THEORETICAL CALLOUTS =================
    ax_hud = fig.add_subplot(gs[0, 1], facecolor=COLOR_CARD)
    ax_hud.axis("off")

    # Outer border for HUD card
    hud_border = patches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=COLOR_CARD, edgecolor=COLOR_BORDER, linewidth=1.5
    )
    ax_hud.add_patch(hud_border)

    # Header Card: Key Takeaways
    ax_hud.text(0.08, 0.92, "COMPUTATIONAL EFFICIENCY & METRICS", fontsize=16, fontweight="bold", color=COLOR_CYAN)

    # Metric Box 1: Function Evaluations (NFE)
    box1 = patches.FancyBboxPatch((0.08, 0.74), 0.84, 0.15, boxstyle="round,pad=0.015", facecolor="#141E33", edgecolor=COLOR_CYAN, linewidth=1.5)
    ax_hud.add_patch(box1)
    ax_hud.text(0.12, 0.84, "SAMPLING EFFICIENCY (NFE)", fontsize=11, fontweight="bold", color=COLOR_TEXT_MUTED)
    ax_hud.text(0.12, 0.79, "NCM: 12 Passes  vs.  Diffusion: 128 Passes", fontsize=14, fontweight="bold", color=COLOR_TEXT_WHITE)
    ax_hud.text(0.12, 0.755, "10.7x Fewer Function Evaluations | 21.3x Faster Latency on RTX 3070 Ti", fontsize=10.5, color=COLOR_CYAN)

    # Metric Box 2: Hard Accuracy Gap
    box2 = patches.FancyBboxPatch((0.08, 0.55), 0.84, 0.15, boxstyle="round,pad=0.015", facecolor="#261224", edgecolor=COLOR_ROSE, linewidth=1.5)
    ax_hud.add_patch(box2)
    ax_hud.text(0.12, 0.65, "HARD SUDOKU ACCURACY (30 CLUES)", fontsize=11, fontweight="bold", color=COLOR_TEXT_MUTED)
    ax_hud.text(0.12, 0.60, "NCM: 86.1%  vs.  AR: 1.0%  (86x Accuracy Gain)", fontsize=14, fontweight="bold", color=COLOR_TEXT_WHITE)
    ax_hud.text(0.12, 0.565, "AR collapses due to greedy commitment without backtracking", fontsize=10.5, color=COLOR_ROSE)

    # Theory Box: The "No Backtracking" Trap
    box3 = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.42, boxstyle="round,pad=0.015", facecolor="#0B132B", edgecolor=COLOR_BORDER, linewidth=1.2)
    ax_hud.add_patch(box3)
    ax_hud.text(0.12, 0.45, "THEORETICAL INSIGHT: THE BACKTRACKING TRAP", fontsize=12, fontweight="bold", color=COLOR_GREEN)

    theory_text = (
        "1. Why Autoregressive Models Collapse on 2D Grids:\n"
        "   - AR models generate cell-by-cell left-to-right (0 to 80).\n"
        "   - Sudoku requires bidirectional global consistency: cell (8, 8)\n"
        "     constrains cell (0, 8) and subgrid (2, 2).\n"
        "   - An AR model with no backtracking makes irreversible greedy\n"
        "     choices at cell 10 that violate unseen constraints at cell 60.\n\n"
        "2. Why Diffusion is Compute-Heavy:\n"
        "   - Denoising all 81 tokens across 128 iterations burns enormous\n"
        "     FLOPs and frequently settles in conflicting local modes (30.2%).\n\n"
        "3. How NCM Solves It in 12 Passes:\n"
        "   - Coarse scales (1-4) establish global digit balance across 3x3 boxes.\n"
        "   - Intermediate scales (5-8) enforce row/col parity simultaneously.\n"
        "   - Fine scales (9-12) lock in exact numbers in parallel!"
    )
    ax_hud.text(0.12, 0.26, theory_text, fontsize=10.5, color=COLOR_TEXT_MUTED, va="center", linespacing=1.35)

    # Save outputs
    out_png = os.path.join(output_dir, "sudoku_benchmark_results.png")
    out_svg = os.path.join(output_dir, "sudoku_benchmark_results.svg")

    plt.savefig(out_png, dpi=300, facecolor=COLOR_BG, edgecolor="none", bbox_inches="tight")
    plt.savefig(out_svg, facecolor=COLOR_BG, edgecolor="none", bbox_inches="tight")
    plt.close()

    print(f"[Rendered] {out_png}")
    print(f"[Rendered] {out_svg}")


if __name__ == "__main__":
    render_sudoku_benchmark_chart()
