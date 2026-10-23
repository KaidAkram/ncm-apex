/**
 * Hyperframe Declarative Composition (JS): WMT14 Hierarchical Translation Emergence
 * Coarse-to-fine lexical crystallization trajectory (BLEU 6.0 -> 77.8).
 * Resolution: 1080x1080 Square @ 60 FPS (1:1), Dark Bento Tech Theme.
 */

const TranslationTrajectory = {
  name: 'TranslationTrajectory',
  width: 1080,
  height: 1080,
  fps: 60,
  durationInFrames: 840, // 14.0s
  durationSec: 14.0,
  theme: {
    background: '#090B10',
    panelBg: 'rgba(20, 24, 33, 0.75)',
    accentCyan: '#00F0FF',
    accentBlue: '#3B82F6',
    accentGreen: '#10B981',
    textPrimary: '#FFFFFF',
    textSecondary: 'rgba(255, 255, 255, 0.6)',
  },
  renderMetadata: {
    composition: 'TranslationTrajectory',
    targetVideoPath: 'assets/videos/translation_trajectory.mp4',
    previewWebmPath: 'assets/videos/translation_trajectory_preview.webm',
    codec: 'h264_nvenc',
    bitrate: '8000k',
  }
};

module.exports = TranslationTrajectory;
