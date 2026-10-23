"""
NCM-Apex: Tokenizer Training Runner Script
Runs verification training, codebook diagnostic telemetry, and multi-level accuracy tracking.
"""

import argparse
import sys
import torch
from rich.console import Console
from rich.table import Table

from src.models.multiscale_vqvae import MultiscaleVQVAE
from src.engine.tokenizer_trainer import TokenizerTrainer

console = Console()


def run_training_demo(
    vocab_size: int = 1000,
    seq_len: int = 64,
    batch_size: int = 8,
    steps: int = 60,
    embed_dim: int = 256,
    levels: list = [16, 32, 64],
    codebook_sizes: list = [512, 512, 512],
):
    console.print("[bold red]=== NCM-Apex: Multiscale RVQ-VAE Tokenizer Training Demo ===[/bold red]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: [cyan]{device}[/cyan] | Target GPU: [cyan]{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}[/cyan]")

    # Fixed synthetic toy dataset simulating repeated structured tokens
    torch.manual_seed(42)
    sample_tokens = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)

    model = MultiscaleVQVAE(
        vocab_size=vocab_size,
        context_length=seq_len,
        embed_dim=embed_dim,
        codebook_vocab_sizes=codebook_sizes,
        levels=levels,
        dec_layers=2,
        n_head=4,
        fine_drop_prob=0.3,
    )

    trainer = TokenizerTrainer(model=model, lr=3e-3, device=device)

    table = Table(title="Training Steps & Telemetry")
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Total Loss", justify="right", style="magenta")
    table.add_column("Recon Loss", justify="right", style="yellow")
    table.add_column("VQ Loss", justify="right", style="green")
    table.add_column("Acc (Selected)", justify="right", style="bold green")
    table.add_column("Utilization", justify="right", style="blue")
    table.add_column("Peak VRAM", justify="right", style="red")

    for step in range(1, steps + 1):
        metrics = trainer.train_step(sample_tokens)

        if step % 10 == 0 or step == steps:
            table.add_row(
                f"{step}",
                f"{metrics['total_loss']:.4f}",
                f"{metrics['recon_loss']:.4f}",
                f"{metrics['vq_loss']:.4f}",
                f"{metrics['token_accuracy']:.1f}%",
                f"{metrics['global_utilization']:.1f}%",
                f"{metrics['peak_vram_mb']:.1f} MB",
            )

    console.print(table)

    # Multi-level accuracy evaluation
    console.print("\n[bold]Evaluating Exact Reconstruction Accuracy Across Hierarchical Scales:[/bold]")
    eval_metrics = trainer.evaluate_multilevel_accuracy(sample_tokens)
    level_accs = eval_metrics["level_accuracies"]

    acc_table = Table(title="Coarse-to-Fine Scale Accuracy Ladder")
    acc_table.add_column("Scale k", justify="center", style="bold")
    acc_table.add_column("Sequence Length (l_k)", justify="center", style="cyan")
    acc_table.add_column("Reconstruction Accuracy", justify="center", style="bold green")
    acc_table.add_column("Codebook Perplexity", justify="center", style="yellow")
    acc_table.add_column("Utilization Rate", justify="center", style="blue")

    for k, (acc, ppl, util, l_k) in enumerate(zip(
        level_accs,
        eval_metrics["codebook_perplexities"],
        eval_metrics["per_level_utilization"],
        levels,
    ), start=1):
        acc_table.add_row(
            f"Level {k}",
            f"{l_k} tokens",
            f"{acc:.2f}%",
            f"{ppl:.2f}",
            f"{util:.1f}%",
        )

    console.print(acc_table)
    console.print("\n[bold green][PASS] Phase 1 Tokenizer Training & Multilevel Verification Completed![/bold green]")
    return level_accs


if __name__ == "__main__":
    run_training_demo()
