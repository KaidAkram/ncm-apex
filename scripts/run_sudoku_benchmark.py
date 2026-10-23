"""
NCM-Apex: Run Sudoku Three-Paradigm Benchmark
Executes formal evaluation across Easy, Medium, and Hard tiers,
measuring solve rates, cell accuracies, NFEs, and latencies,
and exporting authentic trajectory trace to assets/trajectories/sudoku_hard_trace.json.
"""

import argparse
import json
from src.benchmarks.sudoku_suite import run_sudoku_three_paradigm_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run Sudoku Three-Paradigm Benchmark")
    parser.add_argument("--tiers", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--boards", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trace-out", default="assets/trajectories/sudoku_hard_trace.json")
    args = parser.parse_args()

    results = run_sudoku_three_paradigm_benchmark(
        tiers=args.tiers,
        boards_per_tier=args.boards,
        device=args.device,
        trace_output_path=args.trace_out,
    )

    print("\n================== BENCHMARK SUMMARY TABLE ==================")
    print(f"{'Difficulty':<10} | {'Paradigm':<8} | {'Solve Rate (%)':<15} | {'Cell Acc (%)':<13} | {'NFE':<6} | {'Latency (ms)':<12}")
    print("-" * 75)
    for tier, data in results.items():
        for p in ["ar", "mdlm", "ncm"]:
            name = p.upper()
            s_rate = data[p]["solve_rate"]
            c_acc = data[p]["cell_accuracy"]
            nfe = data[p]["nfe"]
            lat = data[p]["mean_latency_ms"]
            print(f"{tier.capitalize():<10} | {name:<8} | {s_rate:<15.1f} | {c_acc:<13.1f} | {nfe:<6} | {lat:<12.1f}")


if __name__ == "__main__":
    main()
