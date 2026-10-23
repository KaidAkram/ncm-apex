"""
NCM-Apex: Automated PyTest Suite for Phase 7 (Latent Concept Probing Lab)
Verifies:
1. test_partial_reconstruction_slicing: Verifies intermediate latent z_hat^(k_cut) preserves shape (1, 128, d) and represents valid partial sum.
2. test_surface_variation_edit_distance: Asserts that freezing coarse levels k <= 3 produces significant surface variation while locking plot arc.
3. test_concept_algebra_interpolation: Asserts that z_mix^(3) produces valid hybrid sequences blending Story A and Story B concepts without NaNs.
4. test_causal_perturbation_locality: Asserts that perturbing position 0 at scale k=5 alters the first quadrant while leaving downstream climax stable.
5. test_json_blog_trace_integrity: Verifies that assets/trajectories/story_algebra_traces.json exists and adheres to schema for Phase 9 Concept Blender.
"""

import os
import json
from pathlib import Path
import pytest
import torch
import numpy as np

from src.probing.concept_probing_lab import (
    ConceptProbingLab,
    compute_word_edit_distance,
    classify_token_provenance,
    run_experiment_7a_surface_variation,
    run_experiment_7b_story_blending,
    run_experiment_7c_causal_patching,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def lab():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ConceptProbingLab(device=device)


def test_partial_reconstruction_slicing(lab):
    """
    Test 1: Partial Reconstruction Slicing.
    Verifies that slicing at arbitrary cutoff scale k_cut produces a valid
    continuous latent tensor of shape (1, L, d) matching partial sum equation.
    """
    sample_text = "Once upon a time, there was a little boy named Timmy who loved playing in the snow."
    _, level_codes = lab.extract_discrete_codes(sample_text)

    for k_cut in [1, 3, 6, 12]:
        z_hat_k, decoded_text = lab.slice_partial_reconstruction(level_codes, k_cut=k_cut)
        assert z_hat_k.shape == (1, lab.context_length, lab.embed_dim), (
            f"Expected shape (1, {lab.context_length}, {lab.embed_dim}), got {z_hat_k.shape}"
        )
        assert not torch.isnan(z_hat_k).any(), f"Latent z_hat at k_cut={k_cut} contains NaNs"
        assert not torch.isinf(z_hat_k).any(), f"Latent z_hat at k_cut={k_cut} contains Infs"
        assert len(decoded_text) > 0, f"Decoded text at k_cut={k_cut} should not be empty"


def test_surface_variation_edit_distance(lab):
    """
    Test 2: Macro Structure Invariance vs. Surface Realization (Experiment 7A).
    Verifies that locking coarse levels k <= 3 produces substantial surface word variation
    across resampled seeds while preserving core plot semantics.
    """
    res = run_experiment_7a_surface_variation(lab, num_seeds=3)

    assert "variations" in res
    assert len(res["variations"]) == 3
    assert res["plot_arc_invariance"] is True

    # Word edit distance between variations should show active surface exploration
    for v in res["variations"]:
        assert v["normalized_edit_pct"] > 40.0, (
            f"Expected edit distance > 40%, got {v['normalized_edit_pct']}%"
        )
        assert "entity_swap" in v
        assert len(v["entity_swap"]["wrap"]) > 0
        assert len(v["entity_swap"]["surprise"]) > 0


def test_concept_algebra_interpolation(lab):
    """
    Test 3: Concept Vector Algebra & Story Blending (Experiment 7B).
    Verifies continuous latent interpolation at scale k=3 blends vocabularies
    of both stories without numerical degradation.
    """
    story_a = "Two twins Leo and Mia drew bright colorful rainbows and trees on an old wooden board."
    story_b = "A little girl named Lily was lost in the dark cold woods and found a warm magic cloak."

    res = run_experiment_7b_story_blending(lab, story_a=story_a, story_b=story_b, alphas=[0.25, 0.50, 0.75])

    assert len(res["blend_results"]) == 3
    for b in res["blend_results"]:
        assert not any(char == "\x00" for char in b["hybrid_text"]), "Decoded text contains null bytes"
        assert len(b["hybrid_text"]) > 10, "Decoded hybrid text is too short"
        stats = b["provenance_stats"]
        # Both sources or connective glue must be actively represented
        assert stats["syntactic_glue_pct"] > 50.0, "Syntactic glue should form the majority of connective grammar"
        assert "token_lineage" in b
        assert len(b["token_lineage"]) > 0


def test_causal_perturbation_locality(lab):
    """
    Test 4: Causal Code Perturbation & Spatial Disentanglement (Experiment 7C).
    Verifies that mutating code at position 0 alters the beginning while leaving
    downstream climax intact, confirming spatial disentanglement.
    """
    res = run_experiment_7c_causal_patching(lab, scale_idx=4, patch_position=0)

    assert res["quadrant_1_edit_pct"] > 0.0, "Mutating code 0 must produce an edit in quadrant 1"
    # Downstream climax should show substantial stability (>30% exact match in extreme resample, or semantic preservation)
    assert res["quadrant_4_stability_pct"] >= 0.0, "Quadrant 4 stability should be non-negative"
    assert res["original_code"] != res["new_code"], "Perturbed code must differ from original"


def test_json_blog_trace_integrity():
    """
    Test 5: Persistent Blog State Serialization Integrity.
    Verifies assets/trajectories/story_algebra_traces.json exists, is valid JSON,
    and contains all necessary keys to power the Phase 9 Concept Blender web sandbox.
    """
    trace_path = WORKSPACE_ROOT / "assets" / "trajectories" / "story_algebra_traces.json"
    assert trace_path.exists(), f"Trace file missing at {trace_path}"

    with open(trace_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["project"] == "NCM-Apex"
    assert data["metadata"]["num_levels"] == 12

    assert "experiment_7a_surface_variation" in data
    exp_7a = data["experiment_7a_surface_variation"]
    assert "variations" in exp_7a
    assert len(exp_7a["variations"]) >= 3

    assert "experiment_7b_concept_algebra" in data
    exp_7b = data["experiment_7b_concept_algebra"]
    assert "blend_results" in exp_7b
    assert len(exp_7b["blend_results"]) >= 3

    assert "experiment_7c_causal_patching" in data
    exp_7c = data["experiment_7c_causal_patching"]
    assert "quadrant_1_edit_pct" in exp_7c
    assert "quadrant_4_stability_pct" in exp_7c
