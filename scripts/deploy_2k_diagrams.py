"""
NCM-Apex: Deploy 2K High-Definition Diagrams to MYBLOG
Replaces programmatic SVG diagrams with the authentic 2K Keynote renders.
"""

import os
import shutil
from pathlib import Path

SOURCE_DIR = Path(r"c:\Users\Akram KAID\Desktop\NCM\assets\diagrams")
PUBLIC_TARGET = Path(r"C:\Users\Akram KAID\Desktop\MYBLOG\public\assets\ncm")
POST_TARGET = Path(r"C:\Users\Akram KAID\Desktop\MYBLOG\src\content\blog\mastering-next-concept-modeling\assets")

PUBLIC_TARGET.mkdir(parents=True, exist_ok=True)
POST_TARGET.mkdir(parents=True, exist_ok=True)

# Find files matching patterns
source_files = list(SOURCE_DIR.iterdir())

def find_file(pattern):
    for f in source_files:
        if pattern.lower() in f.name.lower() and f.name.endswith(".jpg"):
            return f
    raise FileNotFoundError(f"Could not find .jpg matching {pattern}")

DIAGRAM_MAPPINGS = {
    "ncm_stage1_multiscale_rvq.jpg": find_file("Multiscale_residual_VQ_VAE"),
    "ncm_stage2_transformer_prior.jpg": find_file("ncm_stage2_transformer_prior"),
    "attention_patterns_comparison.jpg": find_file("attention_patterns_comparison"),
    "sudoku_benchmark_results.jpg": find_file("sudoku_benchmark_results"),
    "roofline_profile_rtx3070ti.jpg": find_file("roofline_profile_rtx3070ti"),
    "test_time_scaling_mbr_cfg.jpg": find_file("test_time_scaling_mbr_cfg"),
    "latent_concept_probing.jpg": find_file("latent_concept_probing"),
    "phd_extensions_ablation.jpg": find_file("phd_extensions_ablation"),
}

print("[1] Copying 2K diagram renders...")
for target_name, src_path in DIAGRAM_MAPPINGS.items():
    print(f"  Mapping {src_path.name} -> {target_name} ({src_path.stat().st_size / 1024 / 1024:.2f} MB)")
    shutil.copy2(src_path, PUBLIC_TARGET / target_name)
    shutil.copy2(src_path, POST_TARGET / target_name)

print("[2] Removing obsolete SVG diagram files as requested...")
obsolete_svgs = [
    "ncm_stage1_multiscale_rvq.svg",
    "ncm_stage2_transformer_prior.svg",
    "sudoku_benchmark_results.svg",
    "roofline_profile_rtx3070ti.svg",
    "test_time_scaling_mbr_cfg.svg",
    "latent_concept_probing.svg",
    "phd_extensions_ablation.svg",
    "ncm_stage1_multiscale_rvq.png",
    "ncm_stage2_transformer_prior.png",
    "sudoku_benchmark_results.png",
    "roofline_profile_rtx3070ti.png",
    "test_time_scaling_mbr_cfg.png",
    "latent_concept_probing.png",
    "phd_extensions_ablation.png",
]

for svg_name in obsolete_svgs:
    p1 = PUBLIC_TARGET / svg_name
    p2 = POST_TARGET / svg_name
    if p1.exists():
        p1.unlink()
        print(f"  Removed {p1.name} from public")
    if p2.exists():
        p2.unlink()
        print(f"  Removed {p2.name} from post assets")

print("[SUCCESS] All 8 2K diagrams deployed!")
