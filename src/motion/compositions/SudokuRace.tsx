/**
 * Hyperframe Declarative Composition: Sudoku 3-Way Split-Screen Race
 * Flagship technical benchmark comparison between Autoregressive, Discrete Diffusion, and NCM.
 * Resolution: 1920x1080 @ 60 FPS (16:9), Bento-box Dark Theme.
 */

import React from 'react';

export interface SudokuRaceProps {
  frame: number;
  fps: number;
  durationInFrames: number;
  data: {
    phase: 'intro' | 'race' | 'outro';
    timeSec: number;
    ar: {
      step: number;
      cell_index?: number;
      placed_digit?: number;
      board_state: number[];
      correct_cells: number;
      row_violations: number;
      col_violations: number;
      box_violations: number;
    };
    arFailed: boolean;
    mdlm: {
      step: number;
      revealed_cells?: number;
      board_state: number[];
      correct_cells: number;
      row_violations: number;
      col_violations: number;
      box_violations: number;
    };
    ncm: {
      step: number;
      scale_level: number;
      codes_count: number;
      board_state: number[];
      correct_cells: number;
      row_violations: number;
      col_violations: number;
      box_violations: number;
    };
    ncmShockwave: number;
    isSelfRepair: boolean;
    cameraScale: number;
  };
}

export const SudokuRace: React.FC<SudokuRaceProps> = ({ frame, fps, durationInFrames, data }) => {
  const { phase, timeSec, ar, arFailed, mdlm, ncm, ncmShockwave, isSelfRepair, cameraScale } = data;

  return (
    <div
      style={{
        width: 1920,
        height: 1080,
        backgroundColor: '#090B10',
        color: '#FFFFFF',
        fontFamily: "'Segoe UI', 'Inter', -apple-system, sans-serif",
        position: 'relative',
        overflow: 'hidden',
        transform: `scale(${cameraScale})`,
        transformOrigin: 'center center',
      }}
    >
      {/* Background Radial Tech Grid */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'radial-gradient(circle at 50% 30%, rgba(0, 240, 255, 0.08) 0%, transparent 60%), linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px)',
          backgroundSize: '100% 100%, 48px 48px, 48px 48px',
          opacity: 0.8,
        }}
      />

      {/* Top Header Bar */}
      <header
        style={{
          position: 'absolute',
          top: 36,
          left: 64,
          right: 64,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
          paddingBottom: 18,
          zIndex: 10,
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
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
              NCM-APEX BENCHMARK
            </span>
            <span style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: 13, letterSpacing: '0.05em' }}>
              HARD SUDOKU (30 CLUES) &bull; 60 FPS REAL-TIME TRACE
            </span>
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: '6px 0 0', letterSpacing: '-0.02em' }}>
            Generation Paradigm Showdown: Spatial Constraint Satisfaction
          </h1>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: 13, color: 'rgba(255, 255, 255, 0.5)' }}>ACCELERATION ARCHITECTURE</div>
          <div style={{ fontSize: 15, fontWeight: 600, color: '#00F0FF' }}>NVIDIA RTX 3070 Ti &bull; FlashAttention-2</div>
        </div>
      </header>

      {/* Main 3-Way Bento Panels Container */}
      <main
        style={{
          position: 'absolute',
          top: 150,
          left: 64,
          right: 64,
          bottom: 120,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: 28,
        }}
      >
        {/* PANEL 1: Autoregressive (AR) */}
        <section
          style={{
            background: 'rgba(20, 24, 33, 0.7)',
            backdropFilter: 'blur(16px)',
            borderRadius: 16,
            border: `1px solid ${arFailed ? '#E11D48' : 'rgba(225, 29, 72, 0.4)'}`,
            boxShadow: arFailed
              ? '0 0 35px rgba(225, 29, 72, 0.35), inset 0 0 20px rgba(225, 29, 72, 0.15)'
              : '0 8px 32px rgba(0, 0, 0, 0.4)',
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <div style={{ color: '#E11D48', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em' }}>PARADIGM A</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>Autoregressive (AR)</div>
            </div>
            <div
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 4,
                background: arFailed ? 'rgba(225, 29, 72, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                color: arFailed ? '#FF4D6D' : 'rgba(255, 255, 255, 0.7)',
                border: `1px solid ${arFailed ? '#E11D48' : 'rgba(255, 255, 255, 0.1)'}`,
                fontWeight: 600,
              }}
            >
              {arFailed ? 'FAILED (0/81)' : `STEP ${ar.step}/81`}
            </div>
          </div>

          {/* Subtitle / Status */}
          <div style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.6)', minHeight: 32 }}>
            {arFailed ? (
              <span style={{ color: '#FF4D6D', fontWeight: 600 }}>
                &times; CONSTRAINT COLLAPSE: Greedy tokenization deadlocked at cell 54.
              </span>
            ) : (
              'Serial raster-scan tokenization with no backtracking.'
            )}
          </div>

          {/* Sudoku Grid Area */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <SudokuGrid board={ar.board_state} accentColor="#E11D48" isFailed={arFailed} />
          </div>

          {/* Mini Telemetry Bar */}
          <div
            style={{
              marginTop: 12,
              padding: '10px 14px',
              borderRadius: 8,
              background: 'rgba(0, 0, 0, 0.4)',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
            }}
          >
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Solved: </span>
              <strong style={{ color: arFailed ? '#E11D48' : '#FFF' }}>{ar.correct_cells}/81</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>NFE: </span>
              <strong>{ar.step}</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Violations: </span>
              <strong style={{ color: '#E11D48' }}>{ar.row_violations + ar.col_violations + ar.box_violations}</strong>
            </div>
          </div>
        </section>

        {/* PANEL 2: Masked Diffusion (MDLM) */}
        <section
          style={{
            background: 'rgba(20, 24, 33, 0.7)',
            backdropFilter: 'blur(16px)',
            borderRadius: 16,
            border: '1px solid rgba(217, 119, 6, 0.45)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), inset 0 0 15px rgba(217, 119, 6, 0.08)',
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <div style={{ color: '#D97706', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em' }}>PARADIGM B</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>Masked Diffusion (MDLM)</div>
            </div>
            <div
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 4,
                background: 'rgba(217, 119, 6, 0.15)',
                color: '#FBBF24',
                border: '1px solid rgba(217, 119, 6, 0.3)',
                fontWeight: 600,
              }}
            >
              {mdlm.step >= 120 ? 'STALLED (30.2%)' : `STEP ${mdlm.step}/128`}
            </div>
          </div>

          {/* Subtitle / Status */}
          <div style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.6)', minHeight: 32 }}>
            {mdlm.step >= 120 ? (
              <span style={{ color: '#FBBF24' }}>
                ! LOCAL MINIMUM: Denoising oscillation trapped across 3x3 blocks.
              </span>
            ) : (
              'Simultaneous iterative unmasking over 128 diffusion steps.'
            )}
          </div>

          {/* Sudoku Grid Area */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <SudokuGrid board={mdlm.board_state} accentColor="#D97706" />
          </div>

          {/* Mini Telemetry Bar */}
          <div
            style={{
              marginTop: 12,
              padding: '10px 14px',
              borderRadius: 8,
              background: 'rgba(0, 0, 0, 0.4)',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
            }}
          >
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Solved: </span>
              <strong style={{ color: '#FBBF24' }}>{mdlm.correct_cells}/81</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>NFE: </span>
              <strong>{mdlm.step}</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Violations: </span>
              <strong style={{ color: '#FBBF24' }}>{mdlm.row_violations + mdlm.col_violations + mdlm.box_violations}</strong>
            </div>
          </div>
        </section>

        {/* PANEL 3: Next-Concept Modeling (NCM) */}
        <section
          style={{
            background: 'rgba(20, 24, 33, 0.75)',
            backdropFilter: 'blur(16px)',
            borderRadius: 16,
            border: `1.5px solid ${ncmShockwave > 0 ? '#10B981' : '#00F0FF'}`,
            boxShadow: ncmShockwave > 0
              ? '0 0 45px rgba(16, 185, 129, 0.45), inset 0 0 25px rgba(16, 185, 129, 0.2)'
              : '0 0 35px rgba(0, 240, 255, 0.35), inset 0 0 20px rgba(0, 240, 255, 0.1)',
            padding: 20,
            display: 'flex',
            flexDirection: 'column',
            position: 'relative',
          }}
        >
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <div style={{ color: '#00F0FF', fontSize: 11, fontWeight: 800, letterSpacing: '0.1em' }}>PARADIGM C (OURS)</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>Next-Concept Modeling (NCM)</div>
            </div>
            <div
              style={{
                fontSize: 11,
                padding: '4px 8px',
                borderRadius: 4,
                background: ncm.correct_cells === 81 ? 'rgba(16, 185, 129, 0.25)' : 'rgba(0, 240, 255, 0.15)',
                color: ncm.correct_cells === 81 ? '#34D399' : '#00F0FF',
                border: `1px solid ${ncm.correct_cells === 81 ? '#10B981' : '#00F0FF'}`,
                fontWeight: 700,
              }}
            >
              {ncm.correct_cells === 81 ? 'SOLVED: 81/81 (100%)' : `STEP ${ncm.step}/12`}
            </div>
          </div>

          {/* Subtitle / Status */}
          <div style={{ fontSize: 12, color: 'rgba(255, 255, 255, 0.6)', minHeight: 32 }}>
            {isSelfRepair ? (
              <span style={{ color: '#38BDF8', fontWeight: 600 }}>
                &bull; STEP 6 SELF-REPAIR: Model undid local clash (21 &rarr; 19 &rarr; 24) to satisfy global column.
              </span>
            ) : ncm.correct_cells === 81 ? (
              <span style={{ color: '#34D399', fontWeight: 700 }}>
                &check; 100% VALID SOLUTION: 0 violations across all 81 cells in 12 NFE!
              </span>
            ) : (
              `Hierarchical scale ${ncm.scale_level}/12 &bull; Resolving macro-to-micro concept codes.`
            )}
          </div>

          {/* Sudoku Grid Area */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <SudokuGrid board={ncm.board_state} accentColor={ncm.correct_cells === 81 ? '#10B981' : '#00F0FF'} isSolved={ncm.correct_cells === 81} />
          </div>

          {/* Mini Telemetry Bar */}
          <div
            style={{
              marginTop: 12,
              padding: '10px 14px',
              borderRadius: 8,
              background: 'rgba(0, 0, 0, 0.4)',
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: 12,
            }}
          >
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Solved: </span>
              <strong style={{ color: ncm.correct_cells === 81 ? '#34D399' : '#00F0FF' }}>{ncm.correct_cells}/81</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>NFE: </span>
              <strong style={{ color: '#00F0FF' }}>{ncm.step}</strong>
            </div>
            <div>
              <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>Violations: </span>
              <strong style={{ color: ncm.correct_cells === 81 ? '#34D399' : '#FFF' }}>
                {ncm.correct_cells === 81 ? 0 : ncm.row_violations + ncm.col_violations + ncm.box_violations}
              </strong>
            </div>
          </div>
        </section>
      </main>

      {/* Bottom Global HUD Telemetry Ticker */}
      <footer
        style={{
          position: 'absolute',
          bottom: 24,
          left: 64,
          right: 64,
          height: 68,
          background: 'rgba(15, 18, 26, 0.85)',
          backdropFilter: 'blur(20px)',
          borderRadius: 12,
          border: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-around',
          padding: '0 24px',
          fontSize: 13,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>EVALUATION EFFICIENCY:</span>
          <span style={{ color: '#00F0FF', fontWeight: 700 }}>12 NFE (NCM)</span>
          <span style={{ color: 'rgba(255, 255, 255, 0.3)' }}>vs</span>
          <span style={{ color: '#D97706' }}>128 NFE (MDLM)</span>
          <span style={{ color: 'rgba(255, 255, 255, 0.3)' }}>vs</span>
          <span style={{ color: '#E11D48' }}>81 NFE (AR)</span>
        </div>

        <div style={{ height: 24, width: 1, background: 'rgba(255, 255, 255, 0.1)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>SOLVE ACCURACY:</span>
          <span style={{ color: '#34D399', fontWeight: 700 }}>NCM: 100.0%</span>
          <span style={{ color: 'rgba(255, 255, 255, 0.3)' }}>&bull;</span>
          <span style={{ color: '#FBBF24' }}>MDLM: 30.2%</span>
          <span style={{ color: 'rgba(255, 255, 255, 0.3)' }}>&bull;</span>
          <span style={{ color: '#FF4D6D' }}>AR: 0.0%</span>
        </div>

        <div style={{ height: 24, width: 1, background: 'rgba(255, 255, 255, 0.1)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: 'rgba(255, 255, 255, 0.4)' }}>NCM SPEEDUP:</span>
          <span style={{ color: '#00F0FF', fontWeight: 700, fontSize: 15 }}>10.7&times; lower NFE vs MDLM</span>
        </div>
      </footer>
    </div>
  );
};

interface SudokuGridProps {
  board: number[];
  accentColor: string;
  isFailed?: boolean;
  isSolved?: boolean;
}

const SudokuGrid: React.FC<SudokuGridProps> = ({ board, accentColor, isFailed, isSolved }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(9, 36px)',
        gridTemplateRows: 'repeat(9, 36px)',
        gap: 2,
        padding: 6,
        background: 'rgba(10, 12, 18, 0.9)',
        borderRadius: 8,
        border: `1px solid ${accentColor}44`,
      }}
    >
      {board.map((val, idx) => {
        const row = Math.floor(idx / 9);
        const col = idx % 9;
        const isBoxRight = col === 2 || col === 5;
        const isBoxBottom = row === 2 || row === 5;
        const hasVal = val > 0;

        return (
          <div
            key={idx}
            style={{
              width: 36,
              height: 36,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 16,
              fontWeight: hasVal ? 700 : 400,
              color: isSolved
                ? '#34D399'
                : isFailed
                ? '#FF4D6D'
                : hasVal
                ? '#FFFFFF'
                : 'rgba(255, 255, 255, 0.1)',
              background: 'rgba(255, 255, 255, 0.02)',
              borderRadius: 3,
              marginRight: isBoxRight ? 4 : 0,
              marginBottom: isBoxBottom ? 4 : 0,
            }}
          >
            {hasVal ? val : ''}
          </div>
        );
      })}
    </div>
  );
};

export default SudokuRace;
