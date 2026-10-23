/**
 * Hyperframe Declarative Composition: WMT14 Hierarchical Translation Emergence
 * Coarse-to-fine lexical crystallization trajectory (BLEU 6.0 -> 77.8).
 * Resolution: 1080x1080 Square @ 60 FPS (1:1), Dark Bento Tech Theme.
 */

import React from 'react';

export interface TranslationTrajectoryProps {
  frame: number;
  fps: number;
  durationInFrames: number;
  data: {
    timeSec: number;
    stepIndex: number;
    totalSteps: number;
    currentStep: {
      step: number;
      scale_level: number;
      code_length: number;
      bleu_score: number;
      text: string;
      emerged_words: string[];
      phase_description: string;
    };
    nextStep: {
      step: number;
      scale_level: number;
      code_length: number;
      bleu_score: number;
      text: string;
      emerged_words: string[];
      phase_description: string;
    };
    stepProgress: number;
    interpolatedBleu: number;
    interpolatedScale: number;
    sourceSentence: string;
    referenceTranslation: string;
    isFinalLock: boolean;
  };
}

export const TranslationTrajectory: React.FC<TranslationTrajectoryProps> = ({ frame, fps, durationInFrames, data }) => {
  const {
    timeSec,
    stepIndex,
    totalSteps,
    currentStep,
    interpolatedBleu,
    interpolatedScale,
    sourceSentence,
    referenceTranslation,
    isFinalLock,
  } = data;

  const referenceWords = referenceTranslation.split(' ');
  const currentWords = currentStep.text.split(' ');

  return (
    <div
      style={{
        width: 1080,
        height: 1080,
        backgroundColor: '#090B10',
        color: '#FFFFFF',
        fontFamily: "'Segoe UI', 'Inter', -apple-system, sans-serif",
        position: 'relative',
        overflow: 'hidden',
        padding: 56,
        boxSizing: 'border-box',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
      }}
    >
      {/* Background Radial Glow */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'radial-gradient(circle at 50% 25%, rgba(0, 240, 255, 0.08) 0%, transparent 60%), radial-gradient(circle at 80% 80%, rgba(16, 185, 129, 0.06) 0%, transparent 50%)',
          pointerEvents: 'none',
        }}
      />

      {/* Top Header Card */}
      <header
        style={{
          background: 'rgba(20, 24, 33, 0.75)',
          backdropFilter: 'blur(16px)',
          borderRadius: 16,
          border: '1px solid rgba(255, 255, 255, 0.1)',
          padding: '24px 28px',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
          <span
            style={{
              background: 'linear-gradient(90deg, #00F0FF, #3B82F6)',
              color: '#000',
              fontSize: 12,
              fontWeight: 800,
              padding: '3px 8px',
              borderRadius: 4,
              letterSpacing: '0.08em',
            }}
          >
            WMT14 DE&rarr;EN BENCHMARK
          </span>
          <span style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: 13 }}>
            NEXT-CONCEPT MODELING &bull; 60 FPS
          </span>
        </div>

        <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 14px', letterSpacing: '-0.02em' }}>
          Hierarchical Lexical Crystallization
        </h1>

        {/* Source German Sentence Box */}
        <div
          style={{
            background: 'rgba(0, 0, 0, 0.3)',
            borderRadius: 8,
            padding: '12px 16px',
            borderLeft: '3px solid #3B82F6',
            fontSize: 13,
            lineHeight: 1.5,
          }}
        >
          <div style={{ color: '#3B82F6', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
            GERMAN INPUT SOURCE:
          </div>
          <div style={{ color: 'rgba(255, 255, 255, 0.85)', fontStyle: 'italic' }}>
            &ldquo;{sourceSentence}&rdquo;
          </div>
        </div>
      </header>

      {/* Center Crystallization Stage */}
      <main
        style={{
          background: 'rgba(20, 24, 33, 0.75)',
          backdropFilter: 'blur(16px)',
          borderRadius: 16,
          border: `1.5px solid ${isFinalLock ? '#10B981' : '#00F0FF'}`,
          padding: '32px 36px',
          boxShadow: isFinalLock
            ? '0 0 45px rgba(16, 185, 129, 0.3), inset 0 0 20px rgba(16, 185, 129, 0.1)'
            : '0 0 35px rgba(0, 240, 255, 0.25), inset 0 0 15px rgba(0, 240, 255, 0.08)',
          position: 'relative',
        }}
      >
        {/* Step Badge & Phase Description */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div
            style={{
              fontSize: 12,
              fontWeight: 800,
              padding: '4px 10px',
              borderRadius: 6,
              background: isFinalLock ? 'rgba(16, 185, 129, 0.2)' : 'rgba(0, 240, 255, 0.15)',
              color: isFinalLock ? '#34D399' : '#00F0FF',
              border: `1px solid ${isFinalLock ? '#10B981' : '#00F0FF'}`,
              letterSpacing: '0.05em',
            }}
          >
            {isFinalLock ? 'FINAL CONVERGENCE' : `STEP ${stepIndex} OF ${totalSteps}`}
          </div>
          <div style={{ fontSize: 13, color: 'rgba(255, 255, 255, 0.6)', fontWeight: 500 }}>
            {currentStep.phase_description}
          </div>
        </div>

        {/* Dynamic Morphing Translation Text Box */}
        <div
          style={{
            minHeight: 120,
            fontSize: 22,
            lineHeight: 1.6,
            fontWeight: 500,
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px 8px',
            alignItems: 'center',
          }}
        >
          {currentWords.map((word, wIdx) => {
            const cleanWord = word.replace(/[^a-zA-Z-]/g, '').toLowerCase();
            const isMatch = referenceWords.some(
              (rw) => rw.replace(/[^a-zA-Z-]/g, '').toLowerCase() === cleanWord
            );

            return (
              <span
                key={`${stepIndex}-${wIdx}-${word}`}
                style={{
                  display: 'inline-block',
                  padding: '2px 6px',
                  borderRadius: 4,
                  background: isMatch
                    ? isFinalLock
                      ? 'rgba(16, 185, 129, 0.25)'
                      : 'rgba(0, 240, 255, 0.18)'
                    : 'rgba(255, 255, 255, 0.05)',
                  color: isMatch ? (isFinalLock ? '#34D399' : '#00F0FF') : 'rgba(255, 255, 255, 0.4)',
                  border: isMatch
                    ? `1px solid ${isFinalLock ? '#10B981' : '#00F0FF'}`
                    : '1px solid transparent',
                  fontWeight: isMatch ? 700 : 400,
                  transition: 'all 0.3s ease',
                }}
              >
                {word}
              </span>
            );
          })}
        </div>

        {/* Reference Target Ground Truth Subtext */}
        <div
          style={{
            marginTop: 24,
            paddingTop: 16,
            borderTop: '1px solid rgba(255, 255, 255, 0.1)',
            fontSize: 13,
            color: 'rgba(255, 255, 255, 0.5)',
          }}
        >
          <strong style={{ color: '#10B981' }}>TARGET REFERENCE: </strong>
          &ldquo;{referenceTranslation}&rdquo;
        </div>
      </main>

      {/* Bottom Gauges Container */}
      <footer
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 20,
        }}
      >
        {/* Radial Arc Gauge: BLEU Metric */}
        <div
          style={{
            background: 'rgba(20, 24, 33, 0.75)',
            backdropFilter: 'blur(16px)',
            borderRadius: 16,
            border: '1px solid rgba(255, 255, 255, 0.1)',
            padding: 24,
            display: 'flex',
            alignItems: 'center',
            gap: 20,
          }}
        >
          {/* Radial Arc SVG representation */}
          <div style={{ position: 'relative', width: 80, height: 80, flexShrink: 0 }}>
            <svg width="80" height="80" viewBox="0 0 80 80">
              <circle
                cx="40"
                cy="40"
                r="34"
                stroke="rgba(255, 255, 255, 0.1)"
                strokeWidth="6"
                fill="none"
              />
              <circle
                cx="40"
                cy="40"
                r="34"
                stroke={isFinalLock ? '#10B981' : '#00F0FF'}
                strokeWidth="6"
                strokeDasharray={`${(interpolatedBleu / 100) * 213} 213`}
                strokeLinecap="round"
                fill="none"
                transform="rotate(-90 40 40)"
              />
            </svg>
            <div
              style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: 13,
                fontWeight: 800,
                color: isFinalLock ? '#34D399' : '#00F0FF',
              }}
            >
              {interpolatedBleu.toFixed(1)}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.4)', fontWeight: 700, letterSpacing: '0.05em' }}>
              SACREBLEU SCORE
            </div>
            <div style={{ fontSize: 22, fontWeight: 800, color: isFinalLock ? '#34D399' : '#FFFFFF' }}>
              {interpolatedBleu.toFixed(1)} <span style={{ fontSize: 13, color: 'rgba(255, 255, 255, 0.4)' }}>/ 100</span>
            </div>
            <div style={{ fontSize: 11, color: isFinalLock ? '#34D399' : '#00F0FF' }}>
              {isFinalLock ? '+71.8 BLEU gain (11 steps)' : 'Monotonic climb'}
            </div>
          </div>
        </div>

        {/* Code Resolution Gauge */}
        <div
          style={{
            background: 'rgba(20, 24, 33, 0.75)',
            backdropFilter: 'blur(16px)',
            borderRadius: 16,
            border: '1px solid rgba(255, 255, 255, 0.1)',
            padding: 24,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.4)', fontWeight: 700, letterSpacing: '0.05em' }}>
              HIERARCHY SCALE L_K
            </span>
            <span style={{ fontSize: 14, fontWeight: 700, color: '#3B82F6' }}>
              {Math.round(interpolatedScale)} / 64 Codes
            </span>
          </div>

          {/* Progress Bar */}
          <div
            style={{
              height: 10,
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: 5,
              overflow: 'hidden',
              marginBottom: 8,
            }}
          >
            <div
              style={{
                height: '100%',
                width: `${(interpolatedScale / 64) * 100}%`,
                background: 'linear-gradient(90deg, #3B82F6, #00F0FF)',
                borderRadius: 5,
                transition: 'width 0.1s linear',
              }}
            />
          </div>

          <div style={{ fontSize: 11, color: 'rgba(255, 255, 255, 0.5)' }}>
            Coarse macro syntax &rarr; Subword lexical fidelity
          </div>
        </div>
      </footer>
    </div>
  );
};

export default TranslationTrajectory;
