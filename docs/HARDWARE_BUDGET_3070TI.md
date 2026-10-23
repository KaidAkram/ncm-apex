# NCM-Apex: Hardware Budget & Execution Strategy (RTX 3070 Ti 8GB)

## 1. Hardware Architecture & Hardware Profile

- **GPU Device:** NVIDIA GeForce RTX 3070 Ti Laptop / Desktop
- **Architecture:** Ampere (GA104)
- **Compute Capability:** 8.6
- **VRAM Total:** 8,192 MiB (8.0 GB GDDR6)
- **Memory Bus Width:** 256-bit
- **Peak Memory Bandwidth:** ~448 GB/s – 608 GB/s
- **Tensor Cores:** 3rd Generation (Native BF16, FP16, INT8, TF32 TensorFloat support)
- **CUDA Runtime / Driver:** NVIDIA 610.60 / CUDA 11.8 & 12.x compatible

---

## 2. VRAM Allocation Breakdown & Parameter Sizing

In an 8GB VRAM envelope, careful partitioning between model parameters, optimizer states, activations, and KV cache/conditioning latents is required.

### 2.1 Model Size Targets
For zero-OOM training and sub-50ms parallel inference, we size our models within two primary operational tiers:

| Component | Compact Config (~25M params) | Mid-Scale Config (~75M params) | Full Paper Baseline (~90M params) |
|---|---|---|---|
| **Tokenizer (Stage 1)** | ~8M params | ~14M params | ~16M params |
| **Generator (Stage 2)** | ~18M params | ~62M params | ~74M params |
| **Total Parameter Count** | **~26M params** | **~76M params** | **~90M params** |
| **FP16 Weight Memory** | 52 MB | 152 MB | 180 MB |
| **AdamW Optimizer States (FP32)** | 208 MB | 608 MB | 720 MB |
| **Gradients (FP16/FP32)** | 52 - 104 MB | 152 - 304 MB | 180 - 360 MB |
| **Peak Training Weights + States** | **~364 MB** | **~1,064 MB** | **~1,260 MB** |

> **VRAM Headroom:** Over **6.7 GB** remains available exclusively for activation memory and batch processing.

---

## 3. Activation Optimization & Compute Acceleration

To maintain maximum throughput and zero Out-Of-Memory (OOM) exceptions during both training and evaluation, the following execution stack is implemented:

### 3.1 PyTorch Scaled Dot-Product Attention (SDPA)
We leverage native PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`):
```python
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=True
):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```
- **Memory Reduction:** $O(L^2)$ attention matrices are never materialized in VRAM. Memory complexity is reduced to $O(L)$, slashing peak activation memory by up to $70\%$.
- **Tensor Core Speedup:** Ampere Tensor Cores execute fused kernel passes with high arithmetic intensity.

### 3.2 Mixed Precision (BF16 / FP16 Automatic Mixed Precision)
- Ampere Compute Capability 8.6 provides hardware acceleration for **Bfloat16 (BF16)**.
- `torch.cuda.amp.autocast(dtype=torch.bfloat16)` provides dynamic dynamic-range stability without requiring manual loss scaling, completely preventing underflow/overflow in gradient computation.

### 3.3 Gradient Checkpointing
- In Stage-2 Transformer training, activation checkpointing (`torch.utils.checkpoint.checkpoint`) is enabled on transformer layers.
- Checkpointing trades a modest $\sim 15\%$ compute overhead during backward passes for a **$60-75\%$ reduction in peak activation footprint**, enabling large context windows ($L=512$ or $1024$) on an 8GB card.

### 3.4 PyTorch Compilation (`torch.compile`)
- In inference and training loops, models are compiled with:
  ```python
  model = torch.compile(model, mode="reduce-overhead", backend="inductor")
  ```
- PyTorch Inductor fuses pointwise operations, LayerNorms, residual adds, and activation functions into single CUDA kernels, minimizing memory read/write cycles.

---

## 4. Batch Sizing, Micro-Batching & Gradient Accumulation

To replicate the effective batch sizes used in the Stanford Hazy Research paper without exceeding 8GB VRAM, we use gradient accumulation:

$$\text{Effective Batch Size} = \text{Micro-Batch Size} \times \text{Gradient Accumulation Steps} \times \text{Num GPUs}$$

| Task / Dataset | Sequence Length $L$ | Micro-Batch Size ($B_{\text{micro}}$) | Accumulation Steps | Effective Batch Size ($B_{\text{eff}}$) | Estimated Peak VRAM |
|---|---|---|---|---|---|
| **Sudoku 9x9** | 81 | 64 | 4 | 256 | ~2.8 GB |
| **WMT14 (De-En)** | 256 | 32 | 8 | 256 | ~4.2 GB |
| **XSum** | 512 | 16 | 16 | 256 | ~5.8 GB |
| **LM1B / TinyStories** | 512 | 16 | 16 | 256 | ~5.6 GB |
| **MBR Inference ($N=16$)** | 512 | 16 (parallel candidates) | - | - | ~3.2 GB |

---

## 5. Thermal & Operating Profile Protocols

During sustained training runs or Manim video rendering on laptop hardware:
1. **CUDA Memory Flushing:** Explicit calls to `torch.cuda.empty_cache()` between evaluation epochs to prevent fragmentation.
2. **Process Monitoring:** Background telemetry logging GPU temperature (keeping $<75^\circ\text{C}$), fan speed, and VRAM utilization via `pynvml` or `nvidia-smi` wrappers.
3. **Deterministic Seeding:** `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)` ensures bitwise reproducibility across all benchmark runs.
