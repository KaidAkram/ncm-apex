/**
 * Hyperframe Declarative Composition (JS): Sudoku 3-Way Split-Screen Race
 * Flagship technical benchmark comparison between Autoregressive, Discrete Diffusion, and NCM.
 * Resolution: 1920x1080 @ 60 FPS (16:9), Bento-box Dark Theme.
 */

const SudokuRace = {
  name: 'SudokuRace',
  width: 1920,
  height: 1080,
  fps: 60,
  durationInFrames: 900, // 15.0s
  durationSec: 15.0,
  theme: {
    background: '#090B10',
    gridColor: 'rgba(255, 255, 255, 0.03)',
    panelBg: 'rgba(20, 24, 33, 0.75)',
    arColor: '#E11D48',
    mdlmColor: '#D97706',
    ncmColor: '#00F0FF',
    successColor: '#10B981',
    textPrimary: '#FFFFFF',
    textSecondary: 'rgba(255, 255, 255, 0.6)',
  },
  renderMetadata: {
    composition: 'SudokuRace',
    targetVideoPath: 'assets/videos/sudoku_race_60fps.mp4',
    previewWebmPath: 'assets/videos/sudoku_race_preview.webm',
    codec: 'h264_nvenc',
    bitrate: '8000k',
  }
};

module.exports = SudokuRace;
