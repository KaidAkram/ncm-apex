"""
NCM-Apex Motion Design Engine - Trajectory Data Loader & Timeline Interpolator
Provides frame-accurate, normalized trajectory states for 60 FPS video synthesis.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class MotionDataLoader:
    def __init__(
        self,
        sudoku_path: Optional[str] = None,
        translation_path: Optional[str] = None
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.sudoku_path = Path(sudoku_path) if sudoku_path else base_dir / "assets" / "trajectories" / "sudoku_hard_trace.json"
        self.translation_path = Path(translation_path) if translation_path else base_dir / "assets" / "trajectories" / "translation_trajectory.json"
        
        self.sudoku_data: Dict[str, Any] = {}
        self.translation_data: Dict[str, Any] = {}
        self.load_data()

    def load_data(self) -> None:
        if not self.sudoku_path.exists():
            raise FileNotFoundError(f"Sudoku trajectory not found at {self.sudoku_path}")
        with open(self.sudoku_path, "r", encoding="utf-8") as f:
            self.sudoku_data = json.load(f)

        if not self.translation_path.exists():
            raise FileNotFoundError(f"Translation trajectory not found at {self.translation_path}")
        with open(self.translation_path, "r", encoding="utf-8") as f:
            self.translation_data = json.load(f)

    def get_sudoku_frame_state(self, time_sec: float) -> Dict[str, Any]:
        """
        Interpolates Sudoku states across AR, MDLM, and NCM at 60 FPS (15s total = 900 frames).
        - 0.0s - 1.5s: Intro presentation (clues only, 30 clues locked in slate)
        - 1.5s - 11.5s: Active generation showdown
        - 11.5s - 15.0s: Final outcomes, emerald shockwave for NCM, failure inspection for AR/MDLM
        """
        total_duration = 15.0
        t = max(0.0, min(float(time_sec), total_duration))
        race_start = 1.5
        race_end = 11.5
        race_duration = race_end - race_start

        if t < race_start:
            intro_progress = t / race_start
            return {
                "phase": "intro",
                "intro_progress": intro_progress,
                "time_sec": t,
                "ar": self.sudoku_data["ar_trace"][0],
                "ar_failed": False,
                "mdlm": self.sudoku_data["mdlm_trace"][0],
                "ncm": self.sudoku_data["ncm_trace"][0],
                "ncm_shockwave": 0.0,
                "camera_scale": 0.96 + 0.04 * intro_progress,
                "is_self_repair": False,
            }

        race_t = min(t - race_start, race_duration)
        race_progress = race_t / race_duration

        # AR Progress (81 steps mapped over 8.0s)
        ar_duration = 8.0
        ar_progress = min(1.0, race_t / ar_duration)
        ar_idx = min(len(self.sudoku_data["ar_trace"]) - 1, int(ar_progress * (len(self.sudoku_data["ar_trace"]) - 1)))
        ar_state = self.sudoku_data["ar_trace"][ar_idx]
        ar_failed = ar_idx >= 53 or race_t >= 7.0

        # MDLM Progress (128 steps mapped over 10.0s)
        mdlm_progress = min(1.0, race_t / race_duration)
        mdlm_idx = min(len(self.sudoku_data["mdlm_trace"]) - 1, int(mdlm_progress * (len(self.sudoku_data["mdlm_trace"]) - 1)))
        mdlm_state = self.sudoku_data["mdlm_trace"][mdlm_idx]

        # NCM Progress (12 steps with dynamic self-repair at Step 6)
        if race_progress < 0.20:
            ncm_idx = int((race_progress / 0.20) * 3)
        elif race_progress < 0.40:
            ncm_idx = 3 + int(((race_progress - 0.20) / 0.20) * 2)
        elif race_progress < 0.60:
            ncm_idx = 5  # Step 6: 21 -> 19 -> 24 dynamic undo
        elif race_progress < 0.90:
            ncm_idx = 6 + int(((race_progress - 0.60) / 0.30) * 5)
        else:
            ncm_idx = 11  # Step 12: 81/81 100% solved
        ncm_idx = min(len(self.sudoku_data["ncm_trace"]) - 1, max(0, ncm_idx))
        ncm_state = self.sudoku_data["ncm_trace"][ncm_idx]

        # Shockwave trigger
        ncm_shockwave = 0.0
        if t >= 10.7:
            ncm_shockwave = min(1.0, (t - 10.7) / 1.5)

        phase = "outro" if t >= race_end else "race"

        return {
            "phase": phase,
            "race_progress": race_progress,
            "time_sec": t,
            "ar": ar_state,
            "ar_failed": ar_failed,
            "mdlm": mdlm_state,
            "ncm": ncm_state,
            "ncm_shockwave": ncm_shockwave,
            "is_self_repair": ncm_idx == 5,
            "camera_scale": 1.0,
        }

    def get_translation_frame_state(self, time_sec: float) -> Dict[str, Any]:
        """
        Interpolates WMT14 Translation states at 60 FPS (14s total = 840 frames).
        11 hierarchical concept steps smoothly morphing.
        """
        total_duration = 14.0
        t = max(0.0, min(float(time_sec), total_duration))
        steps = self.translation_data["steps"]
        num_steps = len(steps)

        if t < 1.2:
            step_idx = 0
            step_prog = t / 1.2
        elif t >= 12.0:
            step_idx = num_steps - 1
            step_prog = 1.0
        else:
            active_w = (t - 1.2) / (12.0 - 1.2)
            float_step = active_w * (num_steps - 1)
            step_idx = int(float_step)
            step_prog = float_step - step_idx

        current_step = steps[step_idx]
        next_step = steps[min(num_steps - 1, step_idx + 1)]

        interp_bleu = current_step["bleu_score"] + (next_step["bleu_score"] - current_step["bleu_score"]) * step_prog
        interp_scale = current_step["code_length"] + (next_step["code_length"] - current_step["code_length"]) * step_prog

        return {
            "time_sec": t,
            "step_index": step_idx + 1,
            "total_steps": num_steps,
            "current_step": current_step,
            "next_step": next_step,
            "step_progress": step_prog,
            "interpolated_bleu": interp_bleu,
            "interpolated_scale": interp_scale,
            "source_sentence": self.translation_data["metadata"]["source_sentence"],
            "reference_translation": self.translation_data["metadata"]["reference_translation"],
            "is_final_lock": step_idx == num_steps - 1,
        }
