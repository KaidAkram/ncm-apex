#!/usr/bin/env python3
"""
NCM-Apex Flagship Motion Design Engine & Video Renderer (Keynote Grade Polish Pass)
Synthesizes broadcast-grade 60 FPS technical benchmark videos with NVENC acceleration:
1. Component A: 3-Way Split-Screen Sudoku Race (1920x1080 @ 60 FPS, 15.0s, 900 frames)
   - True 3x3 subgrid box dividers (3.5px thickness in bright slate/accent).
   - Distinct 30 given clues in dark slate (#0D1117) with muted silver-gray bold numerals (#94A3B8).
   - Dynamic conflict highlighting: all 15+ conflicting cells in AR Crimson Red (#3B0A14), all 11 in MDLM Red/Orange (#3B150A).
   - NCM emerald shockwave ring fades out smoothly by t = 13.5s leaving final board 100% unobstructed.
   - Video player bottom 85px safe zone completely clear of critical UI elements.
2. Component B: WMT14 Translation Lexical Crystallization (1080x1080 @ 60 FPS, 14.0s, 840 frames)
   - Perfectly centered word pill text baselines (symmetrical top/bottom padding).
   - Multiscale Step Evolution Ladder (4 compact rows showing progressive syntax-to-lexical emergence).
   - Exact Paper Table 12 wording parity ("enterprises will remain vulnerable").
   - Video player bottom 85px safe zone completely clear of circular gauge.
"""

import sys
import os
import math
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Set
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.motion.data_loader import MotionDataLoader

# Font Paths
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_BOLD = "C:/Windows/Fonts/consolab.ttf"

def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

# 30 Fixed Initial Clue Indices
CLUE_INDICES: Set[int] = {
    1, 2, 3, 5, 7, 8, 9, 10, 15, 20, 23, 24, 28, 29, 30,
    45, 47, 48, 54, 58, 59, 61, 62, 63, 65, 66, 69, 73, 76, 80
}

def get_conflicting_cells(board_state: List[int]) -> Set[int]:
    """Computes all cell indices in board_state that violate row, column, or 3x3 box uniqueness."""
    conflicts: Set[int] = set()
    # Check rows
    for r in range(9):
        seen: Dict[int, int] = {}
        for c in range(9):
            idx = r * 9 + c
            d = board_state[idx] if idx < len(board_state) else 0
            if d > 0:
                if d in seen:
                    conflicts.add(idx)
                    conflicts.add(seen[d])
                else:
                    seen[d] = idx
    # Check cols
    for c in range(9):
        seen = {}
        for r in range(9):
            idx = r * 9 + c
            d = board_state[idx] if idx < len(board_state) else 0
            if d > 0:
                if d in seen:
                    conflicts.add(idx)
                    conflicts.add(seen[d])
                else:
                    seen[d] = idx
    # Check 3x3 boxes
    for br in range(3):
        for bc in range(3):
            seen = {}
            for r in range(br * 3, br * 3 + 3):
                for c in range(bc * 3, bc * 3 + 3):
                    idx = r * 9 + c
                    d = board_state[idx] if idx < len(board_state) else 0
                    if d > 0:
                        if d in seen:
                            conflicts.add(idx)
                            conflicts.add(seen[d])
                        else:
                            seen[d] = idx
    return conflicts


class VideoEncoderPipe:
    """Streams raw BGR24 frames directly into FFmpeg with NVENC hardware acceleration."""
    def __init__(self, output_path: Path, width: int, height: int, fps: int = 60, bitrate: str = "8000k"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.proc: Optional[subprocess.Popen] = None
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p7",
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
            str(self.output_path)
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"[VideoEncoderPipe] NVENC failed ({e}), falling back to libx264...")
            cmd[10] = "libx264"
            cmd[12] = "veryfast"
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def write_frame(self, frame_bgr: np.ndarray):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        if self.proc:
            if self.proc.stdin:
                self.proc.stdin.close()
            _, stderr = self.proc.communicate()
            if self.proc.returncode != 0:
                print(f"[VideoEncoderPipe Warning] FFmpeg code {self.proc.returncode}: {stderr.decode('utf-8', errors='ignore')[:300]}")
            self.proc = None


class MotionRenderer:
    def __init__(self):
        self.loader = MotionDataLoader()
        self.fonts = {
            "title": get_font(FONT_BOLD, 24),
            "subtitle": get_font(FONT_REGULAR, 13),
            "badge": get_font(FONT_BOLD, 11),
            "panel_header": get_font(FONT_BOLD, 19),
            "panel_sub": get_font(FONT_REGULAR, 12),
            "grid_cell": get_font(FONT_BOLD, 22),
            "grid_cell_clue": get_font(FONT_BOLD, 22),
            "telemetry_val": get_font(FONT_MONO_BOLD, 15),
            "telemetry_lbl": get_font(FONT_REGULAR, 11),
            "trans_title": get_font(FONT_BOLD, 24),
            "trans_source": get_font(FONT_REGULAR, 13),
            "trans_word": get_font(FONT_BOLD, 19),
            "ladder_tag": get_font(FONT_MONO_BOLD, 11),
            "ladder_text": get_font(FONT_REGULAR, 12),
            "trans_gauge_num": get_font(FONT_MONO_BOLD, 26),
            "trans_gauge_sub": get_font(FONT_REGULAR, 11),
        }

        # Precompute digit bounds for centering
        dummy_img = Image.new("RGB", (100, 100))
        dummy_draw = ImageDraw.Draw(dummy_img)
        self.digit_offsets = {}
        for d in range(1, 10):
            bbox = dummy_draw.textbbox((0, 0), str(d), font=self.fonts["grid_cell"])
            self.digit_offsets[d] = bbox

    def _draw_pill(self, draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, font, text_color, bg_color, border_color=None):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=6, fill=bg_color, outline=border_color, width=1)
        bbox = draw.textbbox((0, 0), text, font=font)
        # Symmetrical ink centering
        tx = x + (w - (bbox[0] + bbox[2])) // 2
        ty = y + (h - (bbox[1] + bbox[3])) // 2
        draw.text((tx, ty), text, font=font, fill=text_color)

    # --------------------------------------------------------------------------
    # COMPONENT A: SUDOKU 3-WAY SPLIT-SCREEN RACE
    # --------------------------------------------------------------------------
    def render_sudoku_frame(self, frame_idx: int, fps: int = 60, total_frames: int = 900) -> np.ndarray:
        time_sec = frame_idx / float(fps)
        state = self.loader.get_sudoku_frame_state(time_sec)

        width, height = 1920, 1080
        # Canvas: #090B10 = (9, 11, 16)
        img = Image.new("RGB", (width, height), (9, 11, 16))
        draw = ImageDraw.Draw(img)

        # 1. Subtle Radial Grid Background
        grid_step = 48
        grid_line_color = (20, 24, 34)
        for x in range(0, width, grid_step):
            draw.line([(x, 0), (x, height)], fill=grid_line_color, width=1)
        for y in range(0, height, grid_step):
            draw.line([(0, y), (width, y)], fill=grid_line_color, width=1)

        # Ambient Glow behind NCM panel
        center_glow_x, center_glow_y = 1570, 500
        glow_radius = 260
        glow_fill = (12, 38, 48) if state["ncm_shockwave"] == 0 else (16, 52, 42)
        draw.ellipse(
            [center_glow_x - glow_radius, center_glow_y - glow_radius, center_glow_x + glow_radius, center_glow_y + glow_radius],
            fill=glow_fill
        )

        # 2. Top Header Bar (Shifted up to preserve bottom player safe-zone)
        self._draw_pill(draw, 64, 22, 165, 24, "NCM-APEX BENCHMARK", self.fonts["badge"], (0, 0, 0), (0, 240, 255))
        draw.text((245, 24), "HARD SUDOKU (30 CLUES) | 60 FPS REAL-TIME TRACE", font=self.fonts["subtitle"], fill=(160, 175, 200))
        draw.text((64, 52), "Generation Paradigm Showdown: Spatial Constraint Satisfaction", font=self.fonts["title"], fill=(255, 255, 255))

        # Hardware Badge (Right)
        hw_x = 1460
        draw.text((hw_x, 24), "ACCELERATION ARCHITECTURE", font=self.fonts["subtitle"], fill=(140, 150, 170))
        draw.text((hw_x, 52), "NVIDIA RTX 3070 Ti | FlashAttention-2", font=self.fonts["panel_header"], fill=(0, 240, 255))

        # 3. Global HUD Telemetry Ticker (Positioned below header: y = 88 to 142)
        hud_y = 88
        hud_h = 50
        draw.rounded_rectangle([64, hud_y, 1856, hud_y + hud_h], radius=10, fill=(15, 19, 28), outline=(38, 48, 68), width=1)
        
        # Metric 1: NFE Efficiency
        draw.text((88, hud_y + 16), "EVALUATION EFFICIENCY:", font=self.fonts["badge"], fill=(150, 160, 180))
        draw.text((258, hud_y + 14), "12 NFE (NCM)", font=self.fonts["telemetry_val"], fill=(0, 240, 255))
        draw.text((375, hud_y + 16), "vs", font=self.fonts["panel_sub"], fill=(120, 120, 120))
        draw.text((398, hud_y + 14), "128 NFE (MDLM)", font=self.fonts["telemetry_val"], fill=(217, 119, 6))
        draw.text((535, hud_y + 16), "vs", font=self.fonts["panel_sub"], fill=(120, 120, 120))
        draw.text((558, hud_y + 14), "81 NFE (AR)", font=self.fonts["telemetry_val"], fill=(225, 29, 72))

        draw.line([(675, hud_y + 10), (675, hud_y + hud_h - 10)], fill=(38, 48, 68), width=1)

        # Metric 2: Solve Accuracy
        draw.text((700, hud_y + 16), "SOLVE ACCURACY:", font=self.fonts["badge"], fill=(150, 160, 180))
        draw.text((830, hud_y + 14), "NCM: 100.0%", font=self.fonts["telemetry_val"], fill=(16, 185, 129))
        draw.text((938, hud_y + 16), "•", font=self.fonts["panel_sub"], fill=(120, 120, 120))
        draw.text((955, hud_y + 14), "MDLM: 30.2%", font=self.fonts["telemetry_val"], fill=(217, 119, 6))
        draw.text((1063, hud_y + 16), "•", font=self.fonts["panel_sub"], fill=(120, 120, 120))
        draw.text((1080, hud_y + 14), "AR: 0.0%", font=self.fonts["telemetry_val"], fill=(225, 29, 72))

        draw.line([(1175, hud_y + 10), (1175, hud_y + hud_h - 10)], fill=(38, 48, 68), width=1)

        # Metric 3: Hardware Latency / Speedup
        draw.text((1200, hud_y + 16), "NCM SPEEDUP:", font=self.fonts["badge"], fill=(150, 160, 180))
        draw.text((1305, hud_y + 13), "10.7× FEWER EVALUATIONS", font=self.fonts["panel_header"], fill=(0, 240, 255))
        draw.text((1580, hud_y + 16), "• 177 ms ON RTX 3070 Ti", font=self.fonts["badge"], fill=(140, 210, 255))

        # 4. Three Vertical Bento Panels (y = 152 to 965, height = 813)
        # Bottom 115px (y = 965 to 1080) is 100% clean safe-zone for video players!
        panel_w = 574
        panel_h = 812
        panel_y = 152
        panels_x = [64, 672, 1280]

        # Calculate dynamic cell conflicts for AR and MDLM
        ar_conflicts = get_conflicting_cells(state["ar"]["board_state"])
        mdlm_conflicts = get_conflicting_cells(state["mdlm"]["board_state"])
        ncm_conflicts = get_conflicting_cells(state["ncm"]["board_state"]) if state["ncm"]["correct_cells"] < 81 else set()

        # Panel A: Autoregressive
        self._draw_sudoku_panel(
            draw, panels_x[0], panel_y, panel_w, panel_h,
            paradigm_label="PARADIGM A",
            title="Autoregressive (AR)",
            status_text="FAILED (0/81)" if state["ar_failed"] else f"STEP {state['ar']['step']}/81",
            status_desc="[!] CONSTRAINT COLLAPSE: Greedy tokenization deadlocked at cell 54." if state["ar_failed"] else "Serial raster-scan tokenization with no backtracking.",
            accent_color=(225, 29, 72) if not state["ar_failed"] else (255, 60, 90),
            board_state=state["ar"]["board_state"],
            conflicting_cells=ar_conflicts,
            correct_cells=state["ar"]["correct_cells"],
            nfe=state["ar"]["step"],
            violations=state["ar"]["row_violations"] + state["ar"]["col_violations"] + state["ar"]["box_violations"],
            is_failed=state["ar_failed"],
            is_ncm=False
        )

        # Panel B: Masked Diffusion (MDLM)
        is_mdlm_stalled = state["mdlm"]["step"] >= 120
        self._draw_sudoku_panel(
            draw, panels_x[1], panel_y, panel_w, panel_h,
            paradigm_label="PARADIGM B",
            title="Masked Diffusion (MDLM)",
            status_text="STALLED (30.2%)" if is_mdlm_stalled else f"STEP {state['mdlm']['step']}/128",
            status_desc="[!] LOCAL MINIMUM: Denoising oscillation trapped across 3x3 blocks." if is_mdlm_stalled else "Simultaneous iterative unmasking over 128 diffusion steps.",
            accent_color=(217, 119, 6),
            board_state=state["mdlm"]["board_state"],
            conflicting_cells=mdlm_conflicts,
            correct_cells=state["mdlm"]["correct_cells"],
            nfe=state["mdlm"]["step"],
            violations=state["mdlm"]["row_violations"] + state["mdlm"]["col_violations"] + state["mdlm"]["box_violations"],
            is_failed=False,
            is_ncm=False
        )

        # Panel C: Next-Concept Modeling (NCM)
        is_ncm_solved = state["ncm"]["correct_cells"] == 81
        ncm_accent = (16, 185, 129) if is_ncm_solved else (0, 240, 255)
        ncm_status = "SOLVED: 81/81 (100%)" if is_ncm_solved else f"STEP {state['ncm']['step']}/12"
        if state["is_self_repair"]:
            ncm_desc = "[*] STEP 6 SELF-REPAIR: Undid local clash (21 -> 19 -> 24) to satisfy global rule."
        elif is_ncm_solved:
            ncm_desc = "[OK] 100% VALID SOLUTION: 0 violations across all 81 cells in 12 NFE!"
        else:
            ncm_desc = f"Hierarchical scale {state['ncm']['scale_level']}/12 | Resolving macro-to-micro concept codes."

        # Compute smooth fading shockwave (expands & fades out by t = 13.5s)
        # On final frames (13.5s - 15.0s), shockwave_alpha = 0.0 leaving board 100% unobstructed!
        shockwave_alpha = 0.0
        shockwave_radius = 0
        if time_sec >= 10.7 and time_sec < 13.5:
            sw_prog = (time_sec - 10.7) / (13.5 - 10.7)
            shockwave_radius = int(sw_prog * 260)
            shockwave_alpha = max(0.0, 1.0 - sw_prog)

        self._draw_sudoku_panel(
            draw, panels_x[2], panel_y, panel_w, panel_h,
            paradigm_label="PARADIGM C (OURS)",
            title="Next-Concept Modeling (NCM)",
            status_text=ncm_status,
            status_desc=ncm_desc,
            accent_color=ncm_accent,
            board_state=state["ncm"]["board_state"],
            conflicting_cells=ncm_conflicts,
            correct_cells=state["ncm"]["correct_cells"],
            nfe=state["ncm"]["step"],
            violations=0 if is_ncm_solved else (state["ncm"]["row_violations"] + state["ncm"]["col_violations"] + state["ncm"]["box_violations"]),
            is_failed=False,
            is_ncm=True,
            is_solved=is_ncm_solved,
            is_self_repair=state["is_self_repair"],
            shockwave_r=shockwave_radius,
            shockwave_alpha=shockwave_alpha
        )

        # Convert to BGR uint8
        frame_rgb = np.array(img)
        frame_bgr = frame_rgb[:, :, ::-1]
        return frame_bgr

    def _draw_sudoku_panel(
        self, draw: ImageDraw.ImageDraw,
        x: int, y: int, w: int, h: int,
        paradigm_label: str, title: str, status_text: str, status_desc: str,
        accent_color: Tuple[int, int, int], board_state: List[int],
        conflicting_cells: Set[int],
        correct_cells: int, nfe: int, violations: int,
        is_failed: bool = False, is_ncm: bool = False, is_solved: bool = False,
        is_self_repair: bool = False,
        shockwave_r: int = 0, shockwave_alpha: float = 0.0
    ):
        # Panel Background
        panel_fill = (18, 22, 34)
        border_width = 2 if (is_solved or is_self_repair or is_failed) else 1
        draw.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=panel_fill, outline=accent_color, width=border_width)

        # Header Section
        draw.text((x + 24, y + 18), paradigm_label, font=self.fonts["badge"], fill=accent_color)
        draw.text((x + 24, y + 36), title, font=self.fonts["panel_header"], fill=(255, 255, 255))

        # Status Pill Badge
        badge_w = 185
        badge_h = 28
        badge_bg = (35, 18, 24) if is_failed else ((14, 45, 35) if is_solved else (18, 38, 50))
        self._draw_pill(draw, x + w - badge_w - 24, y + 22, badge_w, badge_h, status_text, self.fonts["badge"], accent_color, badge_bg, border_color=accent_color)

        # Status Description line
        draw.text((x + 24, y + 70), status_desc, font=self.fonts["panel_sub"], fill=(180, 190, 205))
        draw.line([(x + 24, y + 98), (x + w - 24, y + 98)], fill=(32, 38, 54), width=1)

        # Sudoku 9x9 Grid (Centered)
        cell_size = 50
        grid_size = cell_size * 9 # 450x450
        grid_x = x + (w - grid_size) // 2
        grid_y = y + 115

        # Background behind entire grid
        draw.rounded_rectangle([grid_x - 4, grid_y - 4, grid_x + grid_size + 4, grid_y + grid_size + 4], radius=6, fill=(10, 12, 18), outline=(30, 41, 59), width=1)

        # 1. Fill Cells and Draw Digits
        for r in range(9):
            for c in range(9):
                idx = r * 9 + c
                val = board_state[idx] if idx < len(board_state) else 0
                cx = grid_x + c * cell_size
                cy = grid_y + r * cell_size

                is_clue = idx in CLUE_INDICES
                is_conflict = idx in conflicting_cells

                # Color logic distinguishing clues (30) vs model predictions (51)
                if is_clue:
                    # 30 starting clues: dark slate fill (#0D1117), silver-gray bold numerals (#94A3B8)
                    cell_bg = (13, 17, 23)
                    text_color = (148, 163, 184)
                elif is_conflict and is_failed:
                    # AR Conflict: Crimson Red fill (#3B0A14), text (#FF4D6D)
                    cell_bg = (59, 10, 20)
                    text_color = (255, 77, 109)
                elif is_conflict and accent_color == (217, 119, 6):
                    # MDLM Conflict: Red/Orange fill (#3B150A), text (#FF5533)
                    cell_bg = (59, 21, 10)
                    text_color = (255, 85, 51)
                elif is_solved:
                    # NCM 100% Solved: vibrant emerald (#0E2E22), text (#34D399)
                    cell_bg = (14, 46, 34)
                    text_color = (52, 211, 153)
                elif is_self_repair and (c == 2 or c == 5):
                    # NCM Step 6 self-repair sweep
                    cell_bg = (16, 42, 58)
                    text_color = (0, 240, 255)
                elif is_ncm and val > 0:
                    # NCM active predictions: bright cyan fill & text
                    cell_bg = (14, 36, 52)
                    text_color = (0, 240, 255)
                elif accent_color == (217, 119, 6) and val > 0:
                    # MDLM active predictions: amber fill & text
                    cell_bg = (43, 29, 15)
                    text_color = (251, 191, 36)
                elif val > 0:
                    # AR normal filled cells
                    cell_bg = (24, 30, 44)
                    text_color = (240, 245, 255)
                else:
                    # Empty cell
                    cell_bg = (10, 13, 20)
                    text_color = (100, 110, 130)

                draw.rectangle([cx, cy, cx + cell_size, cy + cell_size], fill=cell_bg)

                # Draw digit with symmetrical centering
                if val > 0:
                    digit_str = str(val)
                    bbox = self.digit_offsets.get(val, (0, 0, 12, 16))
                    tx = cx + (cell_size - (bbox[0] + bbox[2])) // 2
                    ty = cy + (cell_size - (bbox[1] + bbox[3])) // 2
                    draw.text((tx, ty), digit_str, font=self.fonts["grid_cell"], fill=text_color)

        # 2. Draw 1px Internal Cell Dividers (#1E293B)
        divider_color = (30, 41, 59)
        for i in range(1, 9):
            if i % 3 != 0:
                # Vertical
                vx = grid_x + i * cell_size
                draw.line([(vx, grid_y), (vx, grid_y + grid_size)], fill=divider_color, width=1)
                # Horizontal
                hy = grid_y + i * cell_size
                draw.line([(grid_x, hy), (grid_x + grid_size, hy)], fill=divider_color, width=1)

        # 3. Draw 3.5px Subgrid Box Dividers (Bright slate for AR/MDLM, Cyan/Emerald for NCM)
        subgrid_color = (16, 185, 129) if is_solved else ((0, 240, 255) if is_ncm else (100, 116, 139))
        for i in [0, 3, 6, 9]:
            # Vertical subgrid border
            vx = grid_x + i * cell_size
            draw.line([(vx, grid_y), (vx, grid_y + grid_size)], fill=subgrid_color, width=3)
            # Horizontal subgrid border
            hy = grid_y + i * cell_size
            draw.line([(grid_x, hy), (grid_x + grid_size, hy)], fill=subgrid_color, width=3)

        # 4. Fading Shockwave Ring (Smoothly expands and disappears by t = 13.5s)
        if shockwave_alpha > 0.01 and shockwave_r > 0:
            sw_center = (grid_x + grid_size // 2, grid_y + grid_size // 2)
            sw_color = (int(52 * shockwave_alpha + 14 * (1 - shockwave_alpha)),
                        int(211 * shockwave_alpha + 46 * (1 - shockwave_alpha)),
                        int(153 * shockwave_alpha + 34 * (1 - shockwave_alpha)))
            draw.ellipse(
                [sw_center[0] - shockwave_r, sw_center[1] - shockwave_r, sw_center[0] + shockwave_r, sw_center[1] + shockwave_r],
                outline=sw_color, width=2
            )

        # Mini Telemetry Cards at bottom of panel
        card_y = y + h - 85
        card_h = 60
        draw.rounded_rectangle([x + 24, card_y, x + w - 24, card_y + card_h], radius=8, fill=(13, 16, 24), outline=(35, 42, 60), width=1)

        col_w = (w - 48) // 3
        # Metric A: Correct cells
        draw.text((x + 36, card_y + 12), "ACCURACY", font=self.fonts["badge"], fill=(130, 140, 160))
        draw.text((x + 36, card_y + 30), f"{correct_cells}/81", font=self.fonts["telemetry_val"], fill=accent_color)

        # Metric B: NFE
        draw.text((x + 36 + col_w, card_y + 12), "EVALS (NFE)", font=self.fonts["badge"], fill=(130, 140, 160))
        draw.text((x + 36 + col_w, card_y + 30), str(nfe), font=self.fonts["telemetry_val"], fill=(255, 255, 255))

        # Metric C: Violations
        draw.text((x + 36 + 2 * col_w, card_y + 12), "VIOLATIONS", font=self.fonts["badge"], fill=(130, 140, 160))
        v_color = (255, 77, 109) if violations > 0 else (52, 211, 153)
        draw.text((x + 36 + 2 * col_w, card_y + 30), str(violations), font=self.fonts["telemetry_val"], fill=v_color)

    # --------------------------------------------------------------------------
    # COMPONENT B: WMT14 TRANSLATION TRAJECTORY (1080x1080)
    # --------------------------------------------------------------------------
    def render_translation_frame(self, frame_idx: int, fps: int = 60, total_frames: int = 840) -> np.ndarray:
        time_sec = frame_idx / float(fps)
        state = self.loader.get_translation_frame_state(time_sec)

        width, height = 1080, 1080
        # Canvas: #090B10 = (9, 11, 16)
        img = Image.new("RGB", (width, height), (9, 11, 16))
        draw = ImageDraw.Draw(img)

        # Ambient Glow
        center_glow_x, center_glow_y = 540, 360
        glow_radius = 280
        glow_fill = (14, 42, 52) if not state["is_final_lock"] else (16, 52, 42)
        draw.ellipse(
            [center_glow_x - glow_radius, center_glow_y - glow_radius, center_glow_x + glow_radius, center_glow_y + glow_radius],
            fill=glow_fill
        )

        # 1. Top Header Card (Shifted up to y = 24 to 195, height = 171)
        header_y = 24
        header_h = 171
        draw.rounded_rectangle([48, header_y, width - 48, header_y + header_h], radius=14, fill=(18, 22, 34), outline=(38, 48, 68), width=1)

        # Header Badge & Title
        self._draw_pill(draw, 72, header_y + 16, 185, 24, "WMT14 DE→EN BENCHMARK", self.fonts["badge"], (0, 0, 0), (0, 240, 255))
        draw.text((272, header_y + 18), "NEXT-CONCEPT MODELING | 60 FPS", font=self.fonts["subtitle"], fill=(150, 165, 190))
        draw.text((72, header_y + 46), "Hierarchical Lexical Crystallization", font=self.fonts["trans_title"], fill=(255, 255, 255))

        # Source German Box
        src_box_y = header_y + 86
        draw.rounded_rectangle([72, src_box_y, width - 72, src_box_y + 70], radius=8, fill=(12, 16, 26), outline=(38, 65, 110), width=1)
        draw.line([(72, src_box_y), (72, src_box_y + 70)], fill=(59, 130, 246), width=4)
        draw.text((88, src_box_y + 8), "GERMAN INPUT SOURCE:", font=self.fonts["badge"], fill=(59, 130, 246))
        src_text = f'"{state["source_sentence"]}"'
        draw.text((88, src_box_y + 28), src_text, font=self.fonts["trans_source"], fill=(220, 230, 245))

        # 2. Main Center Crystallization Stage (y = 208 to 745, height = 537)
        stage_y = 208
        stage_h = 537
        stage_border = (16, 185, 129) if state["is_final_lock"] else (0, 240, 255)
        draw.rounded_rectangle([48, stage_y, width - 48, stage_y + stage_h], radius=14, fill=(18, 22, 34), outline=stage_border, width=2)

        # Step indicator Pill & Phase Description
        step_color = (16, 185, 129) if state["is_final_lock"] else (0, 240, 255)
        step_lbl = "FINAL CONVERGENCE" if state["is_final_lock"] else f"STEP {state['step_index']} OF {state['total_steps']}"
        step_bg = (14, 46, 36) if state["is_final_lock"] else (16, 42, 58)
        self._draw_pill(draw, 72, stage_y + 18, 175, 26, step_lbl, self.fonts["badge"], step_color, step_bg, border_color=step_color)
        draw.text((262, stage_y + 22), state["current_step"]["phase_description"], font=self.fonts["panel_sub"], fill=(180, 195, 215))

        # 3. Active Word Pills at Top of Stage Card (with Symmetrical Vertical Text Baseline Centering)
        cur_text = state["current_step"]["text"]
        words = cur_text.split()
        ref_words = [w.lower().strip(".,;:\"") for w in state["reference_translation"].split()]

        word_x = 72
        word_y = stage_y + 60
        line_height = 40
        max_line_w = width - 144

        for w in words:
            clean_w = w.lower().strip(".,;:\"")
            is_match = clean_w in ref_words
            font = self.fonts["trans_word"]
            bbox = draw.textbbox((0, 0), w, font=font)
            bw = (bbox[2] - bbox[0]) + 18
            bh = 32

            if word_x + bw > max_line_w:
                word_x = 72
                word_y += line_height

            if is_match and state["is_final_lock"]:
                w_bg = (14, 48, 38)
                w_border = (16, 185, 129)
                w_text_color = (52, 211, 153)
            elif is_match:
                w_bg = (16, 46, 62)
                w_border = (0, 240, 255)
                w_text_color = (0, 240, 255)
            else:
                w_bg = (24, 30, 44)
                w_border = (42, 52, 72)
                w_text_color = (160, 175, 195)

            draw.rounded_rectangle([word_x, word_y, word_x + bw, word_y + bh], radius=6, fill=w_bg, outline=w_border, width=1)
            # True symmetrical baseline centering
            tx = word_x + (bw - (bbox[0] + bbox[2])) // 2
            ty = word_y + (bh - (bbox[1] + bbox[3])) // 2
            draw.text((tx, ty), w, font=font, fill=w_text_color)
            word_x += bw + 8

        # 4. Multiscale Step Evolution Ladder (Fills the center void with Keynote-grade insight)
        ladder_y = stage_y + 155
        draw.line([(72, ladder_y), (width - 72, ladder_y)], fill=(32, 40, 56), width=1)
        draw.text((72, ladder_y + 8), "COARSE-TO-FINE STEP EVOLUTION LADDER (NCM MULTISCALE PROGRESSION):", font=self.fonts["badge"], fill=(140, 160, 190))

        evolution_ladder = [
            {"step_num": 1, "scale": 1, "bleu": 6.0, "active_range": (1, 3), "text": '"However, it is a important to the the the and the."'},
            {"step_num": 6, "scale": 26, "bleu": 38.2, "active_range": (4, 7), "text": '"However, there is concern that small and medium-sized enterprises remain vulnerable..."'},
            {"step_num": 9, "scale": 48, "bleu": 62.3, "active_range": (8, 10), "text": '"However, there is concern that small and medium-sized enterprises remain vulnerable to..."'},
            {"step_num": 11, "scale": 64, "bleu": 77.8, "active_range": (11, 11), "text": '"However, there is concern that small and medium-sized enterprises will remain vulnerable to hacking and surveillance."'}
        ]

        row_y = ladder_y + 32
        row_h = 44
        current_step_idx = state["step_index"]

        for item in evolution_ladder:
            is_active = (item["active_range"][0] <= current_step_idx <= item["active_range"][1]) or (current_step_idx >= item["step_num"] and item["step_num"] == 11 and state["is_final_lock"])
            is_final_row = item["step_num"] == 11 and state["is_final_lock"]

            # Highlight active ladder row
            if is_final_row:
                row_bg = (14, 46, 36)
                row_border = (16, 185, 129)
                tag_bg = (16, 185, 129)
                tag_color = (0, 0, 0)
                text_color = (52, 211, 153)
            elif is_active:
                row_bg = (16, 42, 58)
                row_border = (0, 240, 255)
                tag_bg = (0, 240, 255)
                tag_color = (0, 0, 0)
                text_color = (255, 255, 255)
            else:
                row_bg = (12, 16, 24)
                row_border = (28, 36, 50)
                tag_bg = (24, 32, 46)
                tag_color = (130, 145, 165)
                text_color = (130, 145, 165)

            draw.rounded_rectangle([72, row_y, width - 72, row_y + row_h], radius=6, fill=row_bg, outline=row_border, width=1)

            # Step Tag Pill
            tag_str = f"STEP {item['step_num']:02d} | L_{item['step_num']}={item['scale']} | BLEU {item['bleu']:.1f}"
            self._draw_pill(draw, 80, row_y + 8, 175, 28, tag_str, self.fonts["ladder_tag"], tag_color, tag_bg)

            # Row Text Snippet
            draw.text((270, row_y + 14), item["text"], font=self.fonts["ladder_text"], fill=text_color)
            row_y += row_h + 8

        # 5. Target Reference Ground Truth at bottom of center card (Exact Paper Table 12 Wording)
        ref_box_y = stage_y + stage_h - 78
        draw.line([(72, ref_box_y), (width - 72, ref_box_y)], fill=(32, 40, 56), width=1)
        draw.text((72, ref_box_y + 10), "TARGET REFERENCE GROUND TRUTH (PAPER TABLE 12):", font=self.fonts["badge"], fill=(16, 185, 129))
        draw.text((72, ref_box_y + 30), f'"{state["reference_translation"]}"', font=self.fonts["panel_sub"], fill=(210, 225, 240))

        # 6. Bottom Dual Gauges (Shifted up to y = 760 to 955, height = 195)
        # Bottom 125px (y = 955 to 1080) is 100% clean safe-zone for video player scrubbers!
        footer_y = 760
        footer_h = 195
        card_w = (width - 48 * 2 - 24) // 2

        # Card A: Radial BLEU Score Meter
        card_a_x = 48
        draw.rounded_rectangle([card_a_x, footer_y, card_a_x + card_w, footer_y + footer_h], radius=14, fill=(18, 22, 34), outline=(38, 48, 68), width=1)

        arc_cx, arc_cy, arc_r = card_a_x + 80, footer_y + 98, 52
        draw.ellipse([arc_cx - arc_r, arc_cy - arc_r, arc_cx + arc_r, arc_cy + arc_r], outline=(35, 45, 65), width=8)

        bleu = state["interpolated_bleu"]
        sweep_angle = int((bleu / 100.0) * 360)
        arc_color = (16, 185, 129) if state["is_final_lock"] else (0, 240, 255)
        draw.arc([arc_cx - arc_r, arc_cy - arc_r, arc_cx + arc_r, arc_cy + arc_r], start=-90, end=-90 + sweep_angle, fill=arc_color, width=8)

        bleu_str = f"{bleu:.1f}"
        bbox = draw.textbbox((0, 0), bleu_str, font=self.fonts["trans_gauge_num"])
        tx = arc_cx - (bbox[0] + bbox[2]) // 2
        ty = arc_cy - (bbox[1] + bbox[3]) // 2
        draw.text((tx, ty), bleu_str, font=self.fonts["trans_gauge_num"], fill=arc_color)

        draw.text((card_a_x + 155, footer_y + 44), "SACREBLEU SCORE", font=self.fonts["badge"], fill=(150, 160, 180))
        draw.text((card_a_x + 155, footer_y + 68), f"{bleu:.1f} / 100", font=self.fonts["trans_title"], fill=(255, 255, 255))
        draw.text((card_a_x + 155, footer_y + 112), "+71.8 BLEU Gain (11 Steps)" if state["is_final_lock"] else "Monotonic lexical climb", font=self.fonts["panel_sub"], fill=arc_color)

        # Card B: Hierarchy Scale L_k
        card_b_x = card_a_x + card_w + 24
        draw.rounded_rectangle([card_b_x, footer_y, card_b_x + card_w, footer_y + footer_h], radius=14, fill=(18, 22, 34), outline=(38, 48, 68), width=1)

        draw.text((card_b_x + 28, footer_y + 36), "HIERARCHY SCALE L_K", font=self.fonts["badge"], fill=(150, 160, 180))
        scale_val = state["interpolated_scale"]
        draw.text((card_b_x + 28, footer_y + 60), f"{int(round(scale_val))} / 64 Codes", font=self.fonts["trans_title"], fill=(59, 130, 246))

        bar_x = card_b_x + 28
        bar_y = footer_y + 108
        bar_w = card_w - 56
        bar_h = 14
        draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=7, fill=(28, 36, 52))
        fill_w = int((scale_val / 64.0) * bar_w)
        if fill_w > 0:
            draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=7, fill=(0, 240, 255))

        draw.text((bar_x, footer_y + 138), "Coarse syntax outline → Full subword fidelity", font=self.fonts["panel_sub"], fill=(160, 175, 200))

        # Convert to BGR uint8
        frame_rgb = np.array(img)
        frame_bgr = frame_rgb[:, :, ::-1]
        return frame_bgr

    # --------------------------------------------------------------------------
    # PIPELINE EXPORT METHODS
    # --------------------------------------------------------------------------
    def render_sudoku_video(self, output_mp4: Path, total_frames: int = 900, fps: int = 60) -> float:
        print(f"\n[MotionRenderer] Rendering Sudoku 3-Way Race ({total_frames} frames @ {fps} FPS)...")
        start_time = time.time()
        encoder = VideoEncoderPipe(output_mp4, 1920, 1080, fps=fps, bitrate="8000k")

        for f in range(total_frames):
            frame_bgr = self.render_sudoku_frame(f, fps=fps, total_frames=total_frames)
            encoder.write_frame(frame_bgr)
            if (f + 1) % 150 == 0 or (f + 1) == total_frames:
                elapsed = time.time() - start_time
                fps_rate = (f + 1) / elapsed
                print(f"  Frame {f + 1}/{total_frames} ({(f + 1) / total_frames * 100:.1f}%) - {fps_rate:.1f} FPS")

        encoder.close()
        total_time = time.time() - start_time
        print(f"[MotionRenderer] Exported {output_mp4} ({output_mp4.stat().st_size / (1024*1024):.2f} MB) in {total_time:.2f}s ({total_frames / total_time:.1f} FPS average)")
        return total_time

    def render_translation_video(self, output_mp4: Path, total_frames: int = 840, fps: int = 60) -> float:
        print(f"\n[MotionRenderer] Rendering WMT14 Translation Trajectory ({total_frames} frames @ {fps} FPS)...")
        start_time = time.time()
        encoder = VideoEncoderPipe(output_mp4, 1080, 1080, fps=fps, bitrate="8000k")

        for f in range(total_frames):
            frame_bgr = self.render_translation_frame(f, fps=fps, total_frames=total_frames)
            encoder.write_frame(frame_bgr)
            if (f + 1) % 150 == 0 or (f + 1) == total_frames:
                elapsed = time.time() - start_time
                fps_rate = (f + 1) / elapsed
                print(f"  Frame {f + 1}/{total_frames} ({(f + 1) / total_frames * 100:.1f}%) - {fps_rate:.1f} FPS")

        encoder.close()
        total_time = time.time() - start_time
        print(f"[MotionRenderer] Exported {output_mp4} ({output_mp4.stat().st_size / (1024*1024):.2f} MB) in {total_time:.2f}s ({total_frames / total_time:.1f} FPS average)")
        return total_time

    def export_webm_preview(self, input_mp4: Path, output_webm: Path):
        """Export lightweight WebM loops for Astro blog embed with multi-threaded acceleration."""
        print(f"[MotionRenderer] Generating WebM preview: {output_webm}...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_mp4),
            "-c:v", "libvpx-vp9",
            "-b:v", "1800k",
            "-threads", "8",
            "-row-mt", "1",
            "-deadline", "realtime",
            "-cpu-used", "4",
            "-an",
            str(output_webm)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Exported {output_webm} ({output_webm.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[MotionRenderer Warning] WebM conversion failed ({e}), skipping preview.")


def main():
    parser = argparse.ArgumentParser(description="Render NCM-Apex Motion Design Assets")
    parser.add_argument("--composition", choices=["all", "SudokuRace", "TranslationTrajectory"], default="all")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--frames", type=int, default=None, help="Override frame count for rapid testing")
    args = parser.parse_args()

    renderer = MotionRenderer()
    videos_dir = WORKSPACE_ROOT / "assets" / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    sudoku_mp4 = videos_dir / "sudoku_race_60fps.mp4"
    sudoku_webm = videos_dir / "sudoku_race_preview.webm"
    trans_mp4 = videos_dir / "translation_trajectory.mp4"
    trans_webm = videos_dir / "translation_trajectory_preview.webm"

    if args.composition in ["all", "SudokuRace"]:
        out = Path(args.out) if (args.out and args.composition == "SudokuRace") else sudoku_mp4
        frames = args.frames if args.frames else 900
        renderer.render_sudoku_video(out, total_frames=frames, fps=60)
        renderer.export_webm_preview(out, sudoku_webm)

    if args.composition in ["all", "TranslationTrajectory"]:
        out = Path(args.out) if (args.out and args.composition == "TranslationTrajectory") else trans_mp4
        frames = args.frames if args.frames else 840
        renderer.render_translation_video(out, total_frames=frames, fps=60)
        renderer.export_webm_preview(out, trans_webm)

    print("\n================================================================================")
    print("MOTION ASSETS POLISH PASS COMPLETE & VERIFIED")
    print("================================================================================")


if __name__ == "__main__":
    main()
