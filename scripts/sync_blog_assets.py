"""
NCM-Apex Phase 9 Asset Sync and Pipeline Migration Script
Generates all 7 publication diagrams and copies diagrams, videos, and JSON traces to MYBLOG.
"""

import os
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BLOG_PUBLIC_NCM = Path("C:/Users/Akram KAID/Desktop/MYBLOG/public/assets/ncm")
DIAGRAMS_DIR = WORKSPACE_ROOT / "assets" / "diagrams"
VIDEOS_DIR = WORKSPACE_ROOT / "assets" / "videos"
TRAJECTORIES_DIR = WORKSPACE_ROOT / "assets" / "trajectories"
BENCHMARKS_DIR = WORKSPACE_ROOT / "assets" / "benchmarks"

def generate_all_diagrams():
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Generating Diagram 1: Stage-1 Multiscale RVQ ---")
    from src.visualizers.render_rvq_ladder import generate_architecture_diagram
    generate_architecture_diagram()

    print("\n--- Generating Diagram 2: Stage-2 Transformer Prior ---")
    from src.visualizers.render_stage2_visuals import render_stage2_flowchart
    render_stage2_flowchart(str(DIAGRAMS_DIR))

    print("\n--- Generating Diagram 3: Sudoku 3-Way Benchmark Results ---")
    from src.visualizers.render_sudoku_benchmark_chart import render_sudoku_benchmark_chart
    render_sudoku_benchmark_chart(str(DIAGRAMS_DIR))

    print("\n--- Generating Diagram 4 & 5: Roofline Profile & Test-Time Scaling ---")
    from src.visualizers.render_phase_4_5_visuals import render_roofline_bento, render_test_time_scaling_bento
    render_roofline_bento(DIAGRAMS_DIR)
    render_test_time_scaling_bento(DIAGRAMS_DIR)

    print("\n--- Generating Diagram 6: Latent Concept Probing ---")
    from src.visualizers.render_concept_probing_visuals import render_concept_probing_bento
    render_concept_probing_bento(DIAGRAMS_DIR)

    print("\n--- Generating Diagram 7: PhD Extensions Ablation ---")
    from src.visualizers.render_phd_extensions_visuals import render_phd_extensions_bento
    render_phd_extensions_bento(DIAGRAMS_DIR)


def sync_assets_to_blog():
    BLOG_PUBLIC_NCM.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Syncing assets to target blog: {BLOG_PUBLIC_NCM} ---")

    # 1. Bento Diagrams
    diagram_files = [
        "ncm_stage1_multiscale_rvq.svg", "ncm_stage1_multiscale_rvq.png",
        "ncm_stage2_transformer_prior.svg", "ncm_stage2_transformer_prior.png",
        "sudoku_benchmark_results.svg", "sudoku_benchmark_results.png",
        "roofline_profile_rtx3070ti.svg", "roofline_profile_rtx3070ti.png",
        "test_time_scaling_mbr_cfg.svg", "test_time_scaling_mbr_cfg.png",
        "latent_concept_probing.svg", "latent_concept_probing.png",
        "phd_extensions_ablation.svg", "phd_extensions_ablation.png",
    ]

    for fname in diagram_files:
        src = DIAGRAMS_DIR / fname
        if src.exists():
            dst = BLOG_PUBLIC_NCM / fname
            shutil.copy2(src, dst)
            print(f"  [Copied Diagram] {fname} ({src.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [Warning] Diagram {fname} not found in {DIAGRAMS_DIR}")

    # 2. Keynote Motion Design Videos
    video_files = [
        "sudoku_race_60fps.mp4", "sudoku_race_preview.webm",
        "translation_trajectory.mp4", "translation_trajectory_preview.webm",
    ]

    for vname in video_files:
        src = VIDEOS_DIR / vname
        if src.exists():
            dst = BLOG_PUBLIC_NCM / vname
            shutil.copy2(src, dst)
            print(f"  [Copied Video] {vname} ({src.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"  [Warning] Video {vname} not found in {VIDEOS_DIR}")

    # 3. Empirical JSON Payloads
    json_files = [
        (TRAJECTORIES_DIR / "sudoku_hard_trace.json", "sudoku_hard_trace.json"),
        (TRAJECTORIES_DIR / "translation_trajectory.json", "translation_trajectory.json"),
        (TRAJECTORIES_DIR / "story_algebra_traces.json", "story_algebra_traces.json"),
        (BENCHMARKS_DIR / "phd_extensions_results.json", "phd_extensions_results.json"),
    ]

    for src, jname in json_files:
        if src.exists():
            dst = BLOG_PUBLIC_NCM / jname
            shutil.copy2(src, dst)
            print(f"  [Copied JSON] {jname} ({src.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  [Warning] JSON trace {jname} not found at {src}")

    print("\n[Asset Sync Complete] All assets successfully staged into blog public directory!")


if __name__ == "__main__":
    generate_all_diagrams()
    sync_assets_to_blog()
