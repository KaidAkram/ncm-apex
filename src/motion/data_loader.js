/**
 * NCM-Apex Motion Design Engine - Trajectory Data Loader & Timeline Interpolator
 * Normalizes empirical generation traces across AR, MDLM, and NCM paradigms for 60 FPS playback.
 */

const fs = require('fs');
const path = require('path');

class MotionDataLoader {
  constructor(sudokuPath, translationPath) {
    this.sudokuPath = sudokuPath || path.resolve(__dirname, '../../assets/trajectories/sudoku_hard_trace.json');
    this.translationPath = translationPath || path.resolve(__dirname, '../../assets/trajectories/translation_trajectory.json');
    this.sudokuData = null;
    this.translationData = null;
    this.loadData();
  }

  loadData() {
    if (fs.existsSync(this.sudokuPath)) {
      this.sudokuData = JSON.parse(fs.readFileSync(this.sudokuPath, 'utf8'));
    } else {
      throw new Error(`Sudoku trajectory not found at ${this.sudokuPath}`);
    }

    if (fs.existsSync(this.translationPath)) {
      this.translationData = JSON.parse(fs.readFileSync(this.translationPath, 'utf8'));
    } else {
      throw new Error(`Translation trajectory not found at ${this.translationPath}`);
    }
  }

  /**
   * Interpolate Sudoku states for a given playback time in seconds.
   * Total animation duration: 15.0 seconds at 60 FPS (900 frames).
   * - 0.0s to 1.5s: Intro & problem presentation (clues only)
   * - 1.5s to 11.5s: Active race execution (10.0s race window)
   *   * AR finishes/fails at ~9.5s
   *   * MDLM iterates all 128 steps across 1.5s - 11.5s
   *   * NCM executes 12 concept levels with scale-dependent timing:
   *     Steps 1-4 (coarse): quick exploration (~2.0s)
   *     Steps 5-7 (self-repair window): deliberate resolution (~3.5s)
   *     Steps 8-12 (fine crystallization): rapid convergence to 100% (~4.5s)
   * - 11.5s to 15.0s: Final celebration & outcome inspection (emerald shockwave lock)
   */
  getSudokuFrameState(timeSec) {
    const totalDuration = 15.0;
    const t = Math.max(0, Math.min(timeSec, totalDuration));
    const raceStart = 1.5;
    const raceEnd = 11.5;
    const raceDuration = raceEnd - raceStart;

    // Intro state (clues only)
    if (t < raceStart) {
      const introProgress = t / raceStart;
      return {
        phase: 'intro',
        introProgress,
        timeSec: t,
        ar: this.sudokuData.ar_trace[0],
        mdlm: this.sudokuData.mdlm_trace[0],
        ncm: this.sudokuData.ncm_trace[0],
        ncmShockwave: 0.0,
        cameraScale: 0.95 + 0.05 * introProgress,
      };
    }

    // Active race phase
    const raceT = Math.min(t - raceStart, raceDuration);
    const raceProgress = raceT / raceDuration; // 0.0 to 1.0

    // AR progress: 81 steps mapped over 8.0s (1.5s to 9.5s)
    const arDuration = 8.0;
    const arProgress = Math.min(1.0, raceT / arDuration);
    const arStepIdx = Math.min(
      this.sudokuData.ar_trace.length - 1,
      Math.floor(arProgress * (this.sudokuData.ar_trace.length - 1))
    );
    const arState = this.sudokuData.ar_trace[arStepIdx];
    const arFailed = arStepIdx >= 53 || raceT >= 7.0;

    // MDLM progress: 128 steps mapped over full race duration (10.0s)
    const mdlmProgress = Math.min(1.0, raceT / raceDuration);
    const mdlmStepIdx = Math.min(
      this.sudokuData.mdlm_trace.length - 1,
      Math.floor(mdlmProgress * (this.sudokuData.mdlm_trace.length - 1))
    );
    const mdlmState = this.sudokuData.mdlm_trace[mdlmStepIdx];

    // NCM progress: 12 steps with non-linear non-autoregressive concept timing
    // Step 5->6 self-repair happens around raceProgress 0.40 - 0.55
    // Full lock at raceProgress = 0.92 (approx 10.7s)
    let ncmStepIdx = 0;
    if (raceProgress < 0.20) {
      // Steps 1 to 3
      ncmStepIdx = Math.floor((raceProgress / 0.20) * 3);
    } else if (raceProgress < 0.40) {
      // Steps 4 to 5
      ncmStepIdx = 3 + Math.floor(((raceProgress - 0.20) / 0.20) * 2);
    } else if (raceProgress < 0.60) {
      // Step 6 (Dynamic self-repair window: 21 -> 19 -> 24)
      ncmStepIdx = 5;
    } else if (raceProgress < 0.90) {
      // Steps 7 to 11
      ncmStepIdx = 6 + Math.floor(((raceProgress - 0.60) / 0.30) * 5);
    } else {
      // Step 12 (100% Solved lock-in)
      ncmStepIdx = 11;
    }
    ncmStepIdx = Math.min(this.sudokuData.ncm_trace.length - 1, Math.max(0, ncmStepIdx));
    const ncmState = this.sudokuData.ncm_trace[ncmStepIdx];

    // Shockwave trigger after lock-in
    let ncmShockwave = 0.0;
    if (t >= 10.7) {
      ncmShockwave = Math.min(1.0, (t - 10.7) / 1.5);
    }

    const phase = t >= raceEnd ? 'outro' : 'race';

    return {
      phase,
      raceProgress,
      timeSec: t,
      ar: arState,
      arFailed,
      mdlm: mdlmState,
      ncm: ncmState,
      ncmShockwave,
      isSelfRepairWindow: ncmStepIdx === 5,
      cameraScale: 1.0,
    };
  }

  /**
   * Interpolate WMT14 Translation states for a given playback time in seconds.
   * Total animation duration: 14.0 seconds at 60 FPS (840 frames).
   * 11 hierarchical concept steps smoothly morphing.
   */
  getTranslationFrameState(timeSec) {
    const totalDuration = 14.0;
    const t = Math.max(0, Math.min(timeSec, totalDuration));
    const steps = this.translationData.steps;
    const numSteps = steps.length; // 11

    // Intro banner: 0.0s to 1.2s
    // 11 steps spread across 1.2s to 12.0s (~0.98s per step)
    // Outro celebration: 12.0s to 14.0s
    let currentStepIdx = 0;
    let stepProgress = 0.0;

    if (t < 1.2) {
      currentStepIdx = 0;
      stepProgress = t / 1.2;
    } else if (t >= 12.0) {
      currentStepIdx = numSteps - 1;
      stepProgress = 1.0;
    } else {
      const activeWindow = (t - 1.2) / (12.0 - 1.2);
      const floatStep = activeWindow * (numSteps - 1);
      currentStepIdx = Math.floor(floatStep);
      stepProgress = floatStep - currentStepIdx;
    }

    const currentStep = steps[currentStepIdx];
    const nextStep = steps[Math.min(numSteps - 1, currentStepIdx + 1)];

    // Interpolate BLEU score smoothly
    const interpolatedBleu = currentStep.bleu_score + (nextStep.bleu_score - currentStep.bleu_score) * stepProgress;
    const interpolatedScale = currentStep.code_length + (nextStep.code_length - currentStep.code_length) * stepProgress;

    return {
      timeSec: t,
      stepIndex: currentStepIdx + 1,
      totalSteps: numSteps,
      currentStep,
      nextStep,
      stepProgress,
      interpolatedBleu,
      interpolatedScale,
      sourceSentence: this.translationData.metadata.source_sentence,
      referenceTranslation: this.translationData.metadata.reference_translation,
      isFinalLock: currentStepIdx === numSteps - 1,
    };
  }
}

module.exports = MotionDataLoader;
