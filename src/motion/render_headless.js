/**
 * Hyperframe Headless Renderer Runner (Node.js)
 * Orchestrates programmatic motion rendering pipelines and connects with NVENC/FFmpeg.
 */

const { spawn } = require('child_process');
const path = require('path');
const MotionDataLoader = require('./data_loader');
const SudokuRace = require('./compositions/SudokuRace');
const TranslationTrajectory = require('./compositions/TranslationTrajectory');

function parseArgs() {
  const args = process.argv.slice(2);
  const options = {
    composition: 'all',
    out: null,
  };

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--composition' && args[i + 1]) {
      options.composition = args[i + 1];
      i++;
    } else if (args[i] === '--out' && args[i + 1]) {
      options.out = args[i + 1];
      i++;
    }
  }
  return options;
}

async function runPythonNVENCPipeline(composition, outPath) {
  const scriptPath = path.resolve(__dirname, '../../scripts/render_motion_assets.py');
  const pythonCmd = 'C:\\anaconda3\\envs\\gaussian_splatting\\python.exe';
  const args = [scriptPath];

  if (composition && composition !== 'all') {
    args.push('--composition', composition);
  }
  if (outPath) {
    args.push('--out', outPath);
  }

  console.log(`[Hyperframe Node Runner] Dispatching to Hardware NVENC Pipeline: ${composition}...`);
  return new Promise((resolve, reject) => {
    const proc = spawn(pythonCmd, args, { stdio: 'inherit' });
    proc.on('close', (code) => {
      if (code === 0) {
        console.log(`[Hyperframe Node Runner] Successfully completed ${composition}.`);
        resolve(code);
      } else {
        reject(new Error(`Rendering failed with exit code ${code}`));
      }
    });
  });
}

async function main() {
  const options = parseArgs();
  console.log('================================================================================');
  console.log('HYPERFRAME HEADLESS MOTION DESIGN ENGINE - 60 FPS NVENC EXPORTER');
  console.log('================================================================================');

  try {
    const loader = new MotionDataLoader();
    console.log('[Hyperframe] Validated empirical trajectory data assets:');
    console.log(`  - Sudoku Trajectory: AR(${loader.sudokuData.ar_trace.length}), MDLM(${loader.sudokuData.mdlm_trace.length}), NCM(${loader.sudokuData.ncm_trace.length})`);
    console.log(`  - Translation Trajectory: ${loader.translationData.steps.length} Steps (BLEU 6.0 -> 77.8)`);

    await runPythonNVENCPipeline(options.composition, options.out);
  } catch (err) {
    console.error('[Hyperframe Error]', err);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { main, runPythonNVENCPipeline };
