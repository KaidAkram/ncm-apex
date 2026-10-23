"""
Test Suite: Phase 6 - Flagship Motion Design Engine & Video Integrity Suite
Tests trajectory completeness, 60fps video export properties, and frame rendering determinism.
"""

import json
from pathlib import Path
import pytest
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="module")
def trajectories_dir():
    return WORKSPACE_ROOT / "assets" / "trajectories"

@pytest.fixture(scope="module")
def videos_dir():
    return WORKSPACE_ROOT / "assets" / "videos"

def test_trajectory_data_completeness(trajectories_dir):
    """Validate empirical trajectory data completeness for AR, MDLM, and NCM."""
    sudoku_path = trajectories_dir / "sudoku_hard_trace.json"
    assert sudoku_path.exists(), f"Sudoku trajectory missing at {sudoku_path}"

    with open(sudoku_path, "r", encoding="utf-8") as f:
        sudoku_data = json.load(f)

    # Check keys
    assert "ncm_trace" in sudoku_data
    assert "ar_trace" in sudoku_data
    assert "mdlm_trace" in sudoku_data
    assert "metadata" in sudoku_data

    # Check step counts
    assert len(sudoku_data["ar_trace"]) == 81, f"Expected 81 AR steps, found {len(sudoku_data['ar_trace'])}"
    assert len(sudoku_data["mdlm_trace"]) == 128, f"Expected 128 MDLM steps, found {len(sudoku_data['mdlm_trace'])}"
    assert len(sudoku_data["ncm_trace"]) == 12, f"Expected 12 NCM steps, found {len(sudoku_data['ncm_trace'])}"

    # Check NCM progression & step-6 self-repair (21 -> 19 -> 24)
    ncm_trace = sudoku_data["ncm_trace"]
    assert ncm_trace[0]["correct_cells"] == 2
    assert ncm_trace[4]["correct_cells"] == 21  # Step 5
    assert ncm_trace[5]["correct_cells"] == 19  # Step 6: dynamic undo / self-repair
    assert ncm_trace[6]["correct_cells"] == 24  # Step 7
    assert ncm_trace[-1]["correct_cells"] == 81 # Step 12: 100% solved

    # Check board states
    for step in ncm_trace:
        assert len(step["board_state"]) == 81
        assert "row_violations" in step
        assert "col_violations" in step
        assert "box_violations" in step

    # Check WMT14 translation trajectory
    trans_path = trajectories_dir / "translation_trajectory.json"
    assert trans_path.exists(), f"Translation trajectory missing at {trans_path}"

    with open(trans_path, "r", encoding="utf-8") as f:
        trans_data = json.load(f)

    steps = trans_data["steps"]
    assert len(steps) == 11, f"Expected 11 translation steps, got {len(steps)}"
    assert steps[0]["bleu_score"] == 6.0
    assert steps[-1]["bleu_score"] == 77.8

    # Monotonic BLEU improvement assertion
    for i in range(len(steps) - 1):
        assert steps[i+1]["bleu_score"] >= steps[i]["bleu_score"], (
            f"BLEU must not degrade: Step {i} ({steps[i]['bleu_score']}) -> Step {i+1} ({steps[i+1]['bleu_score']})"
        )

def test_video_export_properties(videos_dir):
    """Verify exported MP4 files exist, have valid video streams, and match target resolutions."""
    sudoku_mp4 = videos_dir / "sudoku_race_60fps.mp4"
    trans_mp4 = videos_dir / "translation_trajectory.mp4"

    assert sudoku_mp4.exists(), f"Sudoku video missing at {sudoku_mp4}"
    assert trans_mp4.exists(), f"Translation video missing at {trans_mp4}"

    # File size > 1 MB check
    assert sudoku_mp4.stat().st_size > 1 * 1024 * 1024, f"Sudoku video too small: {sudoku_mp4.stat().st_size} bytes"
    assert trans_mp4.stat().st_size > 1 * 1024 * 1024, f"Translation video too small: {trans_mp4.stat().st_size} bytes"

    # Inspect Sudoku Video Stream
    cap_sudoku = cv2.VideoCapture(str(sudoku_mp4))
    assert cap_sudoku.isOpened(), "Could not open sudoku_race_60fps.mp4"
    sw = int(cap_sudoku.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap_sudoku.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sfps = cap_sudoku.get(cv2.CAP_PROP_FPS)
    scount = int(cap_sudoku.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_sudoku.release()

    assert sw == 1920, f"Expected 1920 width, got {sw}"
    assert sh == 1080, f"Expected 1080 height, got {sh}"
    assert sfps >= 59.0, f"Expected 60 FPS, got {sfps}"
    assert scount >= 720, f"Expected at least 720 frames (12s @ 60fps), got {scount}"
    duration_s = scount / sfps
    assert 12.0 <= duration_s <= 20.0, f"Duration out of range (12s - 20s): {duration_s:.2f}s"

    # Inspect Translation Video Stream
    cap_trans = cv2.VideoCapture(str(trans_mp4))
    assert cap_trans.isOpened(), "Could not open translation_trajectory.mp4"
    tw = int(cap_trans.get(cv2.CAP_PROP_FRAME_WIDTH))
    th = int(cap_trans.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tfps = cap_trans.get(cv2.CAP_PROP_FPS)
    tcount = int(cap_trans.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_trans.release()

    assert tw == 1080, f"Expected 1080 width, got {tw}"
    assert th == 1080, f"Expected 1080 height, got {th}"
    assert tfps >= 59.0, f"Expected 60 FPS, got {tfps}"
    assert tcount >= 720, f"Expected at least 720 frames, got {tcount}"
    tduration_s = tcount / tfps
    assert 12.0 <= tduration_s <= 20.0, f"Duration out of range (12s - 20s): {tduration_s:.2f}s"

def test_frame_render_determinism(videos_dir):
    """Re-render keyframes and assert pixel similarity > 99.5% against golden keynote references."""
    sudoku_golden_path = videos_dir / "golden_keyframe_sudoku_final.png"
    trans_golden_path = videos_dir / "golden_keyframe_wmt14_final.png"

    assert sudoku_golden_path.exists(), f"Sudoku golden reference missing at {sudoku_golden_path}"
    assert trans_golden_path.exists(), f"Translation golden reference missing at {trans_golden_path}"

    golden_sudoku = cv2.imread(str(sudoku_golden_path))
    golden_trans = cv2.imread(str(trans_golden_path))
    assert golden_sudoku is not None and golden_trans is not None, "Failed to load golden reference images"

    from scripts.render_motion_assets import MotionRenderer
    renderer = MotionRenderer()

    # Re-render Sudoku Final Frame (899)
    re_sudoku = renderer.render_sudoku_frame(899)
    assert golden_sudoku.shape == re_sudoku.shape
    diff_sudoku = np.abs(golden_sudoku.astype(np.int32) - re_sudoku.astype(np.int32))
    sim_sudoku = (np.sum(diff_sudoku <= 2) / diff_sudoku.size) * 100.0
    assert sim_sudoku >= 99.5, f"Sudoku render determinism violated: {sim_sudoku:.2f}%"

    # Re-render Translation Final Frame (839)
    re_trans = renderer.render_translation_frame(839)
    assert golden_trans.shape == re_trans.shape
    diff_trans = np.abs(golden_trans.astype(np.int32) - re_trans.astype(np.int32))
    sim_trans = (np.sum(diff_trans <= 2) / diff_trans.size) * 100.0
    assert sim_trans >= 99.5, f"Translation render determinism violated: {sim_trans:.2f}%"

def test_webm_preview_export(videos_dir):
    """Verify lightweight WebM preview assets exist for Astro blog embedding."""
    sudoku_webm = videos_dir / "sudoku_race_preview.webm"
    trans_webm = videos_dir / "translation_trajectory_preview.webm"

    assert sudoku_webm.exists(), f"Sudoku WebM preview missing at {sudoku_webm}"
    assert trans_webm.exists(), f"Translation WebM preview missing at {trans_webm}"
    assert sudoku_webm.stat().st_size > 50 * 1024, "Sudoku WebM preview suspiciously small"
    assert trans_webm.stat().st_size > 50 * 1024, "Translation WebM preview suspiciously small"

def test_keynote_polish_specifications(videos_dir):
    """Verify that safe-zones and Paper Table 12 parity are strictly enforced."""
    trans_golden_path = videos_dir / "golden_keyframe_wmt14_final.png"
    assert trans_golden_path.exists()

    with open(WORKSPACE_ROOT / "assets" / "trajectories" / "translation_trajectory.json", "r", encoding="utf-8") as f:
        tdata = json.load(f)

    # Table 12 exact wording check
    assert "enterprises will remain vulnerable" in tdata["metadata"]["reference_translation"]
    assert "enterprises will remain vulnerable" in tdata["steps"][-1]["text"]
