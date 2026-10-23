"""
NCM-Apex: Official Checkpoint Ingestion & Weight Mapping Adapter
Connects to Hugging Face Hub (hazyresearch/ncm-checkpoints & hazyresearch/ncm-datasets),
downloads official pretrained models, and validates weight compatibility:
1. Sudoku: Easy, Medium, Hard checkpoints
2. TinyStories: Tokenizer & Generator
3. WMT14: De-En Machine Translation
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

# Ensure reference implementation is accessible in path for official model architectures
REF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reference", "ncm"))
if REF_ROOT not in sys.path:
    sys.path.insert(0, REF_ROOT)

from tokenization.vqvae_tokenizer import VQVAETokenizer
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE


CHECKPOINT_REPO = "hazyresearch/ncm-checkpoints"
DATASET_REPO = "hazyresearch/ncm-datasets"


def download_official_checkpoint(
    task: str,
    tier: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Downloads official config, tokenizer, and generator checkpoint files from Hugging Face.

    Args:
        task: 'sudoku', 'tinystories', or 'wmt14'.
        tier: 'easy', 'medium', or 'hard' (for sudoku) or None.
        cache_dir: Optional custom download directory.

    Returns:
        Dict with paths: {'config': ..., 'tokenizer': ..., 'generator': ...}
    """
    if task == "sudoku":
        assert tier in ("easy", "medium", "hard"), f"Sudoku tier must be easy, medium, or hard, got {tier}"
        sub_dir = f"sudoku/{tier}"
    elif task == "tinystories":
        sub_dir = "tinystories"
    elif task == "wmt14":
        sub_dir = "wmt14"
    else:
        raise ValueError(f"Unknown task: {task}")

    cfg_file = f"{sub_dir}/config.yaml"
    tok_file = f"{sub_dir}/tokenizer.pt"
    gen_file = f"{sub_dir}/generator.pt"

    cfg_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=cfg_file, cache_dir=cache_dir)
    tok_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=tok_file, cache_dir=cache_dir)
    gen_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename=gen_file, cache_dir=cache_dir)

    extra_files = {}
    if task == "tinystories":
        vocab_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename="tinystories/bpe_vocab_8192.pickle", cache_dir=cache_dir)
        merges_path = hf_hub_download(repo_id=CHECKPOINT_REPO, filename="tinystories/bpe_merges_8192.pickle", cache_dir=cache_dir)
        extra_files["bpe_vocab"] = vocab_path
        extra_files["bpe_merges"] = merges_path

    return {
        "config": cfg_path,
        "tokenizer": tok_path,
        "generator": gen_path,
        **extra_files,
    }


def download_official_dataset(
    task: str,
    tier: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Downloads official evaluation datasets from Hugging Face.
    """
    if task == "sudoku":
        assert tier in ("easy", "medium", "hard"), f"Sudoku tier must be easy, medium, or hard, got {tier}"
        puzzles_file = f"sudoku/{tier}/puzzles.pickle"
        solutions_file = f"sudoku/{tier}/solutions.pickle"
        p_path = hf_hub_download(repo_id=DATASET_REPO, filename=puzzles_file, repo_type="dataset", cache_dir=cache_dir)
        s_path = hf_hub_download(repo_id=DATASET_REPO, filename=solutions_file, repo_type="dataset", cache_dir=cache_dir)
        return {"puzzles": p_path, "solutions": s_path}
    elif task == "wmt14":
        src_path = hf_hub_download(repo_id=DATASET_REPO, filename="wmt14/sources.pickle", repo_type="dataset", cache_dir=cache_dir)
        ref_path = hf_hub_download(repo_id=DATASET_REPO, filename="wmt14/reference_translations.pickle", repo_type="dataset", cache_dir=cache_dir)
        return {"sources": src_path, "references": ref_path}
    elif task == "tinystories":
        val_path = hf_hub_download(repo_id=DATASET_REPO, filename="tinystories/valid_bpe_8192.pickle", repo_type="dataset", cache_dir=cache_dir)
        return {"validation": val_path}
    else:
        raise ValueError(f"Unknown task: {task}")


def load_sudoku_ncm_model(
    tier: str = "hard",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Any, Any, Any]:
    """
    Loads complete pretrained NCM model for Sudoku (Tokenizer, Generator, Config).

    Returns:
        (tokenizer_wrapper, generator_model, omegaconf_config)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = download_official_checkpoint(task="sudoku", tier=tier)
    cfg = OmegaConf.load(files["config"])
    cfg.tokenizer.vqvae.checkpoint_path = files["tokenizer"]

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()

    gen = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(files["generator"], map_location="cpu", weights_only=False)
    gen.load_state_dict(sd["model"], strict=True)
    gen.eval()

    return tk, gen, cfg


def load_sudoku_dataset(tier: str = "hard") -> Tuple[List[List[int]], List[List[int]]]:
    """Loads official Sudoku puzzles and ground-truth solutions."""
    paths = download_official_dataset(task="sudoku", tier=tier)
    with open(paths["puzzles"], "rb") as f:
        puzzles = pickle.load(f)
    with open(paths["solutions"], "rb") as f:
        solutions = pickle.load(f)
    return puzzles, solutions


def load_tinystories_ncm_model(
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Any, Any, Any]:
    """
    Loads complete pretrained NCM model for TinyStories (Tokenizer, Generator, Config).

    Returns:
        (tokenizer_wrapper, generator_model, omegaconf_config)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = download_official_checkpoint(task="tinystories")
    cfg = OmegaConf.load(files["config"])
    cfg.tokenizer.vqvae.checkpoint_path = files["tokenizer"]
    cfg.tokenizer.pre_tokenizer.vocab_file_path = files["bpe_vocab"]
    cfg.tokenizer.pre_tokenizer.merges_file_path = files["bpe_merges"]

    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.to(device).eval()

    gen = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(device)
    sd = torch.load(files["generator"], map_location="cpu", weights_only=False)
    gen.load_state_dict(sd["model"], strict=True)
    gen.eval()

    return tk, gen, cfg


def load_tinystories_dataset() -> List[List[int]]:
    """Loads official TinyStories validation token sequences."""
    paths = download_official_dataset(task="tinystories")
    with open(paths["validation"], "rb") as f:
        val_data = pickle.load(f)
    return val_data

