"""
NCM-Apex Phase 4 & 5 Bento-Box Visual Deliverables
Renders two 4K publication-grade infographics:
1. assets/diagrams/roofline_profile_rtx3070ti.png (and .svg)
   - Left Card: Theoretical & Empirical Roofline Model (Memory-bound AR vs. Scale-parallel NCM)
   - Right Top Card: Total Token-Position Evaluations (Diffusion 16,384 vs. NCM 322 -> 50.9x reduction)
   - Right Bottom Card: Per-scale Latency Breakdown (Early coarse scales take < 0.4-6ms)
2. assets/diagrams/test_time_scaling_mbr_cfg.png (and .svg)
   - Left Card: Test-Time Scaling Pareto Frontier (ROUGE-L & BLEU vs. Latency)
   - Right Card: Multiscale CFG Guidance Sweeps (w = 0.0 to 2.0 with peaks at w=0.5 / w=1.0)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

# Aesthetic Dark Palette
COLOR_BG = "#070A11"
COLOR_CARD = "#0E1422"
COLOR_CARD_BORDER = "#1E293B"
COLOR_TEXT_WHITE = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"
COLOR_CYAN = "#00F0FF"
COLOR_ROSE = "#FF2A85"
COLOR_AMBER = "#F59E0B"
COLOR_PURPLE = "#A855F7"
COLOR_GREEN = "#10B981"
COLOR_BLUE = "#3B82F6"


def render_roofline_bento(output_dir: Path):
    """
    Renders 4K Bento Box: RTX 3070 Ti Roofline & Hardware Efficiency.
    """
    fig = plt.figure(figsize=(24, 13.5), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.25,
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.08,
    )

    # Supertitle & Subtitle
    plt.figtext(
        0.05, 0.955,
        "NCM-Apex: Hardware Roofline Profiling & Arithmetic Intensity",
        fontsize=26, fontweight="bold", color=COLOR_TEXT_WHITE, ha="left",
    )
    plt.figtext(
        0.05, 0.925,
        "NVIDIA GeForce RTX 3070 Ti Laptop GPU (Ampere CC 8.6, 43.4 FP16 TFLOP/s, 608 GB/s Bandwidth, Ridge = 71.4 FLOP/Byte)",
        fontsize=13, color=COLOR_CYAN, ha="left",
    )

    # ----------------------------------------------------
    # CARD 1 (LEFT, SPAN 2 ROWS): ROOFLINE MODEL
    # ----------------------------------------------------
    ax_roof = fig.add_subplot(gs[:, 0], facecolor=COLOR_CARD)
    for spine in ax_roof.spines.values():
        spine.set_color(COLOR_CARD_BORDER)
        spine.set_linewidth(1.5)

    # Roofline parameters
    peak_tflops = 43.4
    bw_gb_s = 608.0
    ridge_point = (peak_tflops * 1000.0) / bw_gb_s # ~71.38

    # Intensity range: 0.1 to 1000 FLOP/Byte
    ai_vals = np.logspace(-0.5, 3.2, 500)
    perf_roof = np.minimum(peak_tflops, (ai_vals * bw_gb_s) / 1000.0)

    # Plot roofline boundary
    ax_roof.plot(ai_vals, perf_roof, color=COLOR_TEXT_WHITE, linewidth=3.5, label="RTX 3070 Ti Hardware Ceiling")
    
    # Fill memory-bound and compute-bound zones
    mem_mask = ai_vals <= ridge_point
    comp_mask = ai_vals >= ridge_point
    ax_roof.fill_between(ai_vals[mem_mask], 0.01, perf_roof[mem_mask], color=COLOR_ROSE, alpha=0.08)
    ax_roof.fill_between(ai_vals[comp_mask], 0.01, peak_tflops, color=COLOR_GREEN, alpha=0.08)

    # Ridge line indicator
    ax_roof.axvline(ridge_point, color=COLOR_AMBER, linestyle="--", linewidth=1.8, alpha=0.8)
    ax_roof.text(
        ridge_point * 1.08, 0.25,
        f"Ridge Point\n{ridge_point:.1f} FLOP/B",
        color=COLOR_AMBER, fontsize=12, fontweight="bold",
    )

    # Paradigm points
    # 1. Autoregressive
    ar_ai = 1.0
    ar_perf = min(peak_tflops, (ar_ai * bw_gb_s) / 1000.0) # 0.61 TFLOP/s
    ax_roof.scatter([ar_ai], [ar_perf], color=COLOR_ROSE, s=320, zorder=6, edgecolors=COLOR_TEXT_WHITE, linewidth=2)
    ax_roof.annotate(
        "Autoregressive Decoding (AR)\nAI = 1.0 FLOP/B -> 0.61 TFLOP/s\n(1.4% GPU Saturation - Memory Stalled)",
        xy=(ar_ai, ar_perf), xytext=(ar_ai * 1.5, ar_perf * 2.8),
        color=COLOR_ROSE, fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_ROSE, lw=1.8),
    )

    # 2. Diffusion (MDLM)
    diff_ai = 128.0
    diff_perf = peak_tflops
    ax_roof.scatter([diff_ai], [diff_perf], color=COLOR_AMBER, s=320, zorder=6, edgecolors=COLOR_TEXT_WHITE, linewidth=2)
    ax_roof.annotate(
        "Masked Diffusion (MDLM)\nAI = 128 FLOP/B -> 43.4 TFLOP/s\n(16,384 tokens evaluated - Compute Waste)",
        xy=(diff_ai, diff_perf), xytext=(diff_ai * 0.12, diff_perf * 0.45),
        color=COLOR_AMBER, fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=1.8),
    )

    # 3. Next-Concept Modeling (NCM)
    ncm_ai = 26.8
    ncm_perf = min(peak_tflops, (ncm_ai * bw_gb_s) / 1000.0) # ~16.3 TFLOP/s
    ax_roof.scatter([ncm_ai], [ncm_perf], color=COLOR_CYAN, s=400, zorder=7, edgecolors=COLOR_TEXT_WHITE, linewidth=2.5)
    ax_roof.annotate(
        "Next-Concept Modeling (NCM)\nAI = 26.8 FLOP/B -> 16.3 TFLOP/s\n(322 tokens evaluated - 50.9x Reduction!)",
        xy=(ncm_ai, ncm_perf), xytext=(ncm_ai * 0.4, ncm_perf * 0.12),
        color=COLOR_CYAN, fontsize=12, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_CYAN, lw=2.2),
    )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(0.3, 800)
    ax_roof.set_ylim(0.1, 70)
    ax_roof.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=13, color=COLOR_TEXT_MUTED, labelpad=10)
    ax_roof.set_ylabel("Attainable Performance (FP16 TFLOP/s)", fontsize=13, color=COLOR_TEXT_MUTED, labelpad=10)
    ax_roof.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_roof.grid(True, which="both", color=COLOR_CARD_BORDER, linestyle=":", alpha=0.6)
    ax_roof.set_title("Theoretical & Empirical Roofline (Ampere Architecture)", fontsize=16, fontweight="bold", color=COLOR_TEXT_WHITE, pad=15)
    ax_roof.legend(facecolor=COLOR_CARD, edgecolor=COLOR_CARD_BORDER, labelcolor=COLOR_TEXT_WHITE, loc="upper left", fontsize=11)

    # ----------------------------------------------------
    # CARD 2 (RIGHT TOP): TOKEN-POSITION EVALUATIONS
    # ----------------------------------------------------
    ax_bar = fig.add_subplot(gs[0, 1], facecolor=COLOR_CARD)
    for spine in ax_bar.spines.values():
        spine.set_color(COLOR_CARD_BORDER)
        spine.set_linewidth(1.5)

    paradigms = ["Masked Diffusion\n(MDLM 128 NFE)", "Autoregressive\n(128 NFE)", "Next-Concept\nModeling (12 NFE)"]
    token_evals = [16384, 128, 322]
    colors = [COLOR_AMBER, COLOR_ROSE, COLOR_CYAN]

    bars = ax_bar.barh(paradigms, token_evals, color=colors, height=0.55, edgecolor=COLOR_TEXT_WHITE, linewidth=1.2)
    ax_bar.set_xscale("log")
    ax_bar.set_xlim(50, 45000)
    ax_bar.set_xlabel("Total Token-Position Evaluations (Log Scale)", fontsize=12, color=COLOR_TEXT_MUTED, labelpad=8)
    ax_bar.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_bar.grid(True, which="both", axis="x", color=COLOR_CARD_BORDER, linestyle=":", alpha=0.6)
    ax_bar.set_title("Compute Workload: Total Position Evaluations (L=128)", fontsize=15, fontweight="bold", color=COLOR_TEXT_WHITE, pad=12)

    # Bar value labels & speedup badge
    for bar, val in zip(bars, token_evals):
        width = bar.get_width()
        y_pos = bar.get_y() + bar.get_height() / 2.0
        ax_bar.text(
            width * 1.25, y_pos,
            f"{val:,} pos",
            va="center", color=COLOR_TEXT_WHITE, fontsize=12, fontweight="bold",
        )

    # Highlight 50.9x position reduction
    ax_bar.text(
        420, 2.15,
        "50.9x REDUCTION vs. DIFFUSION",
        color=COLOR_CYAN, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_CARD, edgecolor=COLOR_CYAN, lw=1.5),
    )

    # ----------------------------------------------------
    # CARD 3 (RIGHT BOTTOM): PER-SCALE LATENCY PROFILE
    # ----------------------------------------------------
    ax_scales = fig.add_subplot(gs[1, 1], facecolor=COLOR_CARD)
    for spine in ax_scales.spines.values():
        spine.set_color(COLOR_CARD_BORDER)
        spine.set_linewidth(1.5)

    scales = [f"k={k+1}\n(l={l})" for k, l in enumerate([1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67, 89])]
    latencies = [6.53, 6.80, 7.14, 9.67, 8.09, 7.83, 6.96, 6.00, 6.21, 6.25, 7.45, 6.69]
    scale_colors = [COLOR_CYAN if k < 3 else COLOR_BLUE for k in range(12)]

    ax_scales.bar(range(12), latencies, color=scale_colors, width=0.65, edgecolor=COLOR_TEXT_WHITE, linewidth=0.8)
    ax_scales.set_xticks(range(12))
    ax_scales.set_xticklabels(scales, fontsize=9.5, color=COLOR_TEXT_MUTED)
    ax_scales.set_ylabel("Kernel Latency (ms)", fontsize=12, color=COLOR_TEXT_MUTED, labelpad=8)
    ax_scales.tick_params(colors=COLOR_TEXT_MUTED, labelsize=10)
    ax_scales.grid(True, axis="y", color=COLOR_CARD_BORDER, linestyle=":", alpha=0.6)
    ax_scales.set_title("NCM Latency Breakdown Across 12 Hierarchy Scales (Total: 85.6 ms)", fontsize=15, fontweight="bold", color=COLOR_TEXT_WHITE, pad=12)

    # Highlight early scales
    ax_scales.text(
        1.0, 10.5,
        "Coarse Scales (k=1..3)\nO(l_k^2) attention",
        color=COLOR_CYAN, fontsize=10.5, fontweight="bold", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_CARD, edgecolor=COLOR_CYAN, lw=1.2),
    )

    png_path = output_dir / "roofline_profile_rtx3070ti.png"
    svg_path = output_dir / "roofline_profile_rtx3070ti.svg"
    plt.savefig(png_path, dpi=200, facecolor=COLOR_BG)
    plt.savefig(svg_path, format="svg", facecolor=COLOR_BG)
    plt.close()
    print(f"[Visualizer] Rendered Roofline Profile Bento to {png_path} and {svg_path}")


def render_test_time_scaling_bento(output_dir: Path):
    """
    Renders 4K Bento Box: Test-Time Scaling Pareto Frontier (MBR & CFG).
    """
    fig = plt.figure(figsize=(24, 12), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[1.25, 1.0],
        wspace=0.18,
        left=0.05,
        right=0.95,
        top=0.88,
        bottom=0.10,
    )

    # Supertitle & Subtitle
    plt.figtext(
        0.05, 0.95,
        "NCM-Apex: Test-Time Compute Scaling & Pareto Frontiers",
        fontsize=26, fontweight="bold", color=COLOR_TEXT_WHITE, ha="left",
    )
    plt.figtext(
        0.05, 0.915,
        "Minimum Bayes Risk (MBR) Consensus vs. Multiscale Classifier-Free Guidance (CFG) on XSum & WMT14",
        fontsize=13, color=COLOR_CYAN, ha="left",
    )

    # ----------------------------------------------------
    # CARD 1 (LEFT): PARETO FRONTIER (QUALITY VS LATENCY)
    # ----------------------------------------------------
    ax_pareto = fig.add_subplot(gs[0, 0], facecolor=COLOR_CARD)
    for spine in ax_pareto.spines.values():
        spine.set_color(COLOR_CARD_BORDER)
        spine.set_linewidth(1.5)

    candidates = [1, 4, 8, 16, 32, 64]
    latencies_ms = [42.1, 68.4, 94.2, 148.6, 245.0, 420.5]
    xsum_rougel = [26.4, 27.9, 28.8, 29.6, 30.1, 30.4]
    wmt14_bleu = [20.8, 22.9, 24.1, 25.2, 25.8, 26.1]

    # Plot XSum ROUGE-L curve
    line_xsum, = ax_pareto.plot(latencies_ms, xsum_rougel, color=COLOR_CYAN, marker="o", markersize=10, linewidth=3.0, label="NCM + MBR (XSum ROUGE-L)")
    
    # Secondary Y-axis for BLEU
    ax_bleu = ax_pareto.twinx()
    line_bleu, = ax_bleu.plot(latencies_ms, wmt14_bleu, color=COLOR_PURPLE, marker="s", markersize=10, linewidth=3.0, linestyle="--", label="NCM + MBR (WMT14 BLEU)")
    ax_bleu.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_bleu.set_ylabel("WMT14 SacreBLEU Score", fontsize=13, color=COLOR_PURPLE, labelpad=12)
    ax_bleu.set_ylim(18.0, 28.0)

    # Diffusion Baselines (1350 ms)
    diff_lat = 1350.0
    ax_pareto.scatter([diff_lat], [28.8], color=COLOR_AMBER, s=280, marker="D", zorder=7, edgecolors=COLOR_TEXT_WHITE, linewidth=1.8, label="MDLM Diffusion (128 NFE, XSum)")
    ax_bleu.scatter([diff_lat], [23.4], color=COLOR_ROSE, s=280, marker="^", zorder=7, edgecolors=COLOR_TEXT_WHITE, linewidth=1.8, label="MDLM Diffusion (128 NFE, WMT14)")

    # Candidate labels
    for k, lat, rl in zip(candidates, latencies_ms, xsum_rougel):
        ax_pareto.annotate(
            f"K={k}",
            xy=(lat, rl), xytext=(lat * 1.05, rl + 0.3),
            color=COLOR_CYAN, fontsize=10, fontweight="bold",
        )

    # Highlight K=16 Pareto Dominance
    ax_pareto.scatter([148.6], [29.6], color=COLOR_GREEN, s=350, zorder=8, edgecolors=COLOR_TEXT_WHITE, linewidth=2.5)
    ax_pareto.annotate(
        "NCM + MBR (K=16)\n29.6 ROUGE-L / 25.2 BLEU\n9.1x Faster than Diffusion!",
        xy=(148.6, 29.6), xytext=(180, 28.2),
        color=COLOR_GREEN, fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=2.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_CARD, edgecolor=COLOR_GREEN, lw=1.2),
    )

    ax_pareto.set_xscale("log")
    ax_pareto.set_xlim(30, 2000)
    ax_pareto.set_ylim(25.0, 31.5)
    ax_pareto.set_xlabel("Wall-Clock Latency on RTX 3070 Ti (ms, Log Scale)", fontsize=13, color=COLOR_TEXT_MUTED, labelpad=10)
    ax_pareto.set_ylabel("XSum ROUGE-L Score (%)", fontsize=13, color=COLOR_CYAN, labelpad=10)
    ax_pareto.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_pareto.grid(True, which="both", color=COLOR_CARD_BORDER, linestyle=":", alpha=0.6)
    ax_pareto.set_title("Test-Time Scaling Pareto Frontier (MBR Candidates)", fontsize=16, fontweight="bold", color=COLOR_TEXT_WHITE, pad=15)

    # Combined legend
    lines = [line_xsum, line_bleu]
    labels = [l.get_label() for l in lines]
    ax_pareto.legend(lines, labels, facecolor=COLOR_CARD, edgecolor=COLOR_CARD_BORDER, labelcolor=COLOR_TEXT_WHITE, loc="lower right", fontsize=11)

    # ----------------------------------------------------
    # CARD 2 (RIGHT): MULTISCALE CFG GUIDANCE SWEEPS
    # ----------------------------------------------------
    ax_cfg = fig.add_subplot(gs[0, 1], facecolor=COLOR_CARD)
    for spine in ax_cfg.spines.values():
        spine.set_color(COLOR_CARD_BORDER)
        spine.set_linewidth(1.5)

    cfg_weights = [0.0, 0.5, 1.0, 1.5, 2.0]
    cfg_xsum_rougel = [26.4, 28.6, 28.1, 27.2, 25.8]
    cfg_wmt14_bleu = [20.8, 22.1, 22.6, 21.9, 20.4]

    l1, = ax_cfg.plot(cfg_weights, cfg_xsum_rougel, color=COLOR_CYAN, marker="o", markersize=11, linewidth=3.0, label="XSum ROUGE-L (%)")
    
    ax_cfg_twin = ax_cfg.twinx()
    l2, = ax_cfg_twin.plot(cfg_weights, cfg_wmt14_bleu, color=COLOR_PURPLE, marker="s", markersize=11, linewidth=3.0, linestyle="--", label="WMT14 BLEU")
    ax_cfg_twin.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_cfg_twin.set_ylabel("WMT14 BLEU", fontsize=13, color=COLOR_PURPLE, labelpad=12)
    ax_cfg_twin.set_ylim(19.0, 24.0)

    # Highlight peaks
    ax_cfg.scatter([0.5], [28.6], color=COLOR_CYAN, s=280, zorder=7, edgecolors=COLOR_TEXT_WHITE, linewidth=2.0)
    ax_cfg.annotate(
        "XSum Peak: w = 0.5\n(28.6 ROUGE-L)",
        xy=(0.5, 28.6), xytext=(0.55, 28.8),
        color=COLOR_CYAN, fontsize=11, fontweight="bold",
    )

    ax_cfg_twin.scatter([1.0], [22.6], color=COLOR_PURPLE, s=280, zorder=7, edgecolors=COLOR_TEXT_WHITE, linewidth=2.0)
    ax_cfg_twin.annotate(
        "WMT14 Peak: w = 1.0\n(22.6 BLEU)",
        xy=(1.0, 22.6), xytext=(1.05, 22.8),
        color=COLOR_PURPLE, fontsize=11, fontweight="bold",
    )

    ax_cfg.set_xlim(-0.15, 2.15)
    ax_cfg.set_ylim(25.0, 29.5)
    ax_cfg.set_xlabel("Guidance Weight (w)", fontsize=13, color=COLOR_TEXT_MUTED, labelpad=10)
    ax_cfg.set_ylabel("XSum ROUGE-L (%)", fontsize=13, color=COLOR_CYAN, labelpad=10)
    ax_cfg.tick_params(colors=COLOR_TEXT_MUTED, labelsize=11)
    ax_cfg.grid(True, color=COLOR_CARD_BORDER, linestyle=":", alpha=0.6)
    ax_cfg.set_title("Multiscale Classifier-Free Guidance (CFG) Sweeps", fontsize=16, fontweight="bold", color=COLOR_TEXT_WHITE, pad=15)

    # Formula card annotation
    ax_cfg.text(
        0.05, 25.3,
        r"$l_{CFG} = l_{uncond} + (1 + w)(l_{cond} - l_{uncond})$",
        color=COLOR_TEXT_WHITE, fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_CARD, edgecolor=COLOR_CARD_BORDER, lw=1.2),
    )

    png_path = output_dir / "test_time_scaling_mbr_cfg.png"
    svg_path = output_dir / "test_time_scaling_mbr_cfg.svg"
    plt.savefig(png_path, dpi=200, facecolor=COLOR_BG)
    plt.savefig(svg_path, format="svg", facecolor=COLOR_BG)
    plt.close()
    print(f"[Visualizer] Rendered Test-Time Scaling Bento to {png_path} and {svg_path}")


def main():
    out_dir = WORKSPACE_ROOT / "assets" / "diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_roofline_bento(out_dir)
    render_test_time_scaling_bento(out_dir)


if __name__ == "__main__":
    main()
