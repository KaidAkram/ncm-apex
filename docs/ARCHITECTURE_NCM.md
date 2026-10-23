# NCM-Apex: Mathematical & Architectural Specification

## 1. Mathematical Formalism & Overview

Next-Concept Modeling (NCM) factorizes the generation of a discrete sequence $x = (x^1, x^2, \dots, x^L)$ of length $L$ over vocabulary $\Sigma$ through a two-stage hierarchical process:
1. **Stage 1 (Representation / Tokenization):** A multiscale residual vector-quantized autoencoder (RVQ-VAE) maps the continuous latent representation $z \in \mathbb{R}^{L \times d}$ into $K$ discrete code sequences $r_1, r_2, \dots, r_K$.
2. **Stage 2 (Generation / Prior):** A Transformer prior models the joint distribution $p_\theta(r_1, \dots, r_K)$ scale-by-scale in coarse-to-fine order, predicting all tokens at scale $k$ in parallel in a single forward pass.

---

## 2. Stage 1: Multiscale Residual Vector-Quantized Autoencoder (RVQ-VAE)

### 2.1 Spatial Downsampling Hierarchy
Let $K$ denote the total number of hierarchical scales, indexed by $k \in \{1, 2, \dots, K\}$ where $k=1$ is the coarsest scale and $k=K$ is the finest scale.
Let $L$ denote the original sequence length. The sequence length at scale $k$ is given by:
$$l_k = \left\lfloor \frac{L}{2^{K-k}} \right\rfloor$$
Thus, $l_1 = \lfloor L / 2^{K-1} \rfloor$ and $l_K = L$, with the sequence length doubling at each successive scale:
$$l_k = 2 \cdot l_{k-1} \quad \text{for } k > 1$$

### 2.2 Continuous Encoding & Downsampling Operators
Given input token indices $x \in \{1, \dots, |\Sigma|\}^L$, the encoder embeds and transforms the sequence into a continuous latent sequence:
$$z = \text{Encoder}_\psi(x) \in \mathbb{R}^{L \times d}$$

For each scale $k \in \{1, \dots, K\}$, a dedicated spatial downsampling operator $\text{Down}_k$ compresses the continuous latents to resolution $l_k$:
$$z_k = \text{Down}_k(z) \in \mathbb{R}^{l_k \times d}$$
where $\text{Down}_K(z) = z$ (identity at scale $K$).

### 2.3 Residual Quantization & Codebook Lookups
Each scale $k$ is equipped with a discrete codebook:
$$\mathcal{V}_k = \{v_k^{(j)}\}_{j=1}^{V_k} \subset \mathbb{R}^d$$
where $V_k = |\mathcal{V}_k|$ is the codebook cardinality (typically $V_k = 4096$).

Quantization operates residually across scales. Let $\hat{z}^{(0)} = 0 \in \mathbb{R}^{L \times d}$ denote the base reconstruction.
At scale $k$:
1. The accumulated reconstruction from scale $k-1$, denoted $\hat{z}^{(k-1)}$, is spatially downsampled to scale $k$:
   $$\hat{z}_k^{(k-1)} = \text{Down}_k(\hat{z}^{(k-1)}) \in \mathbb{R}^{l_k \times d}$$
2. The residual target at scale $k$ is computed as:
   $$\Delta_k = z_k - \hat{z}_k^{(k-1)} \in \mathbb{R}^{l_k \times d}$$
3. Nearest-neighbor vector quantization maps each spatial position $i \in \{1, \dots, l_k\}$ to its closest codebook centroid:
   $$r_k^i = \arg\min_{j \in \{1, \dots, V_k\}} \|\Delta_k^i - v_k^{(j)}\|_2^2$$
4. The quantized residual representation is:
   $$q_k^i = \mathcal{V}_k[r_k^i] \in \mathbb{R}^d$$
5. The accumulated latent representation at scale $k$ is reconstructed at full length $L$ via the scale-specific reconstruction operator $R_k$:
   $$\hat{z}^{(k)} = \hat{z}^{(k-1)} + R_k(q_k) \in \mathbb{R}^{L \times d}$$
   where $R_k: \mathbb{R}^{l_k \times d} \to \mathbb{R}^{L \times d}$ is an upsampling network (transposed 1D convolutions or linear interpolation followed by MLP).

```
Input Tokens x (L) ──> Encoder ──> z (L x d)
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │ (Down_1)                                              │ (Down_K = Id)
         ▼                                                       ▼
      z_1 (l_1 x d)                                           z_K (L x d)
         │                                                       │
      Residual Δ_1 = z_1 - 0                                  Residual Δ_K = z_K - Down_K(z^(K-1))
         │                                                       │
      Quantize in V_1                                         Quantize in V_K
         │                                                       │
         ▼                                                       ▼
      Codes r_1 (l_1)                                         Codes r_K (L)
         │                                                       │
      q_1 = V_1[r_1]                                          q_K = V_K[r_K]
         │                                                       │
      R_1(q_1) ───────> Accumulate z^(1) ───...───> Accumulate z^(K) ───> Decoder ──> x_hat
```

### 2.4 Nested Scale Dropout & Training Objective
To force coarse scales ($k < K$) to be maximally informative and independently decodable, NCM employs **Nested Scale Dropout**:
During training, a random scale index $k^* \sim \mathcal{U}(\{1, \dots, K\})$ is sampled for each batch element. The tokenizer decoder must reconstruct the original text $x$ using *only* the accumulated latent $\hat{z}^{(k^*)}$ up to scale $k^*$:

The overall Stage-1 loss is:
$$\mathcal{L}_{\text{tok}} = \mathbb{E}_{k^*}\left[ -\frac{1}{L}\sum_{i=1}^L \log p_\psi(x^i \mid \hat{z}^{(k^*)}) + \beta \cdot \text{MSE}\left(z, \text{sg}[\hat{z}^{(k^*)}]\right) + \text{MSE}\left(\text{sg}[z], \hat{z}^{(k^*)}\right) \right]$$
where:
- $\text{sg}[\cdot]$ is the stop-gradient operator (implementing the Straight-Through Estimator).
- $\beta \in [0.25, 0.5]$ is the commitment loss weight.
- The first term is token reconstruction cross-entropy over vocabulary $\Sigma$.

---

## 3. Stage 2: Next-Concept Transformer Prior

### 3.1 Probability Factorization
The prior models the distribution of all discrete codes across all scales:
$$p_\theta(r_{1:K}) = \prod_{k=1}^K p_\theta(r_k \mid r_{<k})$$

Within scale $k$, all $l_k$ concept tokens $r_k = (r_k^1, \dots, r_k^{l_k})$ are generated **conditionally independent given the continuous conditioning $\hat{z}^{(k-1)}$**:
$$p_\theta(r_k \mid r_{<k}) = \prod_{i=1}^{l_k} p_\theta(r_k^i \mid \hat{z}^{(k-1)})$$
Crucially, predicting the entire scale $r_k$ requires **only one single forward pass** through the Transformer. Generating the complete text requires exactly $K$ forward passes.

### 3.2 Block-Diagonal Self-Attention Mask Mechanics
During training, all scales $k \in \{1, \dots, K\}$ are packed into a single sequence of total length $M = \sum_{k=1}^K l_k$.
To enforce coarse-to-fine causality while permitting bidirectional peer interaction within each scale, the attention mask $\mathcal{M} \in \{0, 1\}^{M \times M}$ is structured as a **block-lower-triangular / block-diagonal matrix**:

$$\mathcal{M}_{i, j} = \begin{cases} 
1 & \text{if } \text{scale}(j) \le \text{scale}(i) \\ 
0 & \text{if } \text{scale}(j) > \text{scale}(i) 
\end{cases}$$

```
                Scale 1 (l_1)     Scale 2 (l_2)     Scale 3 (l_3)
Scale 1 (l_1) [  1  1  1 ... 1  |  0  0  0 ... 0  |  0  0  0 ... 0  ]
Scale 2 (l_2) [  1  1  1 ... 1  |  1  1  1 ... 1  |  0  0  0 ... 0  ]
Scale 3 (l_3) [  1  1  1 ... 1  |  1  1  1 ... 1  |  1  1  1 ... 1  ]
```
- **Coarse tokens cannot attend to finer tokens** (zeros in upper triangle).
- **Fine tokens attend fully to all coarser tokens** (ones in lower triangle).
- **Tokens within the same scale attend to all peers bidirectionally** (ones in diagonal blocks).

### 3.3 Conditioning Corruption: Combating Exposure Bias
At test time, the model conditions on its own predicted codes $\hat{r}_{<k}$, which inevitably contain sampling errors. If trained only on ground-truth codes $r_{<k}$, the model suffers severe **exposure bias** (compounding distribution shift).

NCM resolves this via **Conditioning Corruption**:
During training, with probability $p_{\text{corrupt}} \in [0.1, 0.3]$, conditioning code tokens $r_k^i$ are replaced with uniformly sampled random codebook tokens:
$$\tilde{r}_k^i \sim \begin{cases} 
\mathcal{U}(\{1, \dots, V_k\}) & \text{with probability } p_{\text{corrupt}} \\ 
r_k^i & \text{with probability } 1 - p_{\text{corrupt}} 
\end{cases}$$
The continuous latents $\hat{z}^{(k)}$ are reconstructed from $\tilde{r}_{\le k}$, making the downstream generator robust to imperfect predictions.

---

## 4. Coarse-to-Fine Parallel Inference & MBR Decoding

### 4.1 $K$-Step Inference Algorithm
```python
def generate_ncm(generator, tokenizer, K, prompt_latents=None):
    # Step 0: Initialize empty conditioning
    z_hat = torch.zeros(batch_size, L, d_latent, device=device)
    
    codes_all_scales = []
    
    # K sequential forward passes
    for k in range(1, K + 1):
        # 1. Project accumulated continuous conditioning to scale k
        z_k_cond = tokenizer.downsample(z_hat, scale=k)
        
        # 2. Parallel single-pass forward evaluation of all l_k tokens
        logits_k = generator.forward_scale(scale=k, conditioning=z_k_cond) # (B, l_k, V_k)
        
        # 3. Sample discrete codes for scale k
        r_k = sample_tokens(logits_k, temperature=0.8, top_p=0.95) # (B, l_k)
        codes_all_scales.append(r_k)
        
        # 4. Lookup embeddings and update accumulated continuous latent
        q_k = tokenizer.codebooks[k](r_k) # (B, l_k, d)
        z_hat = z_hat + tokenizer.reconstruction_upsample(q_k, scale=k) # (B, L, d)
        
    # Final step: Decode continuous latent z_hat into lexical tokens
    tokens = tokenizer.decode(z_hat) # (B, L)
    return tokens
```

### 4.2 Minimum Bayes Risk (MBR) Decoding
In non-autoregressive and parallel generation, modal collapse and conditional independence approximations can create local disfluencies.
NCM utilizes **Minimum Bayes Risk (MBR)** decoding at negligible marginal cost:
1. Because scale $K$ is generated in parallel in a single forward pass, we sample $N \in [8, 32]$ candidate fine-scale sequences $\{r_K^{(1)}, \dots, r_K^{(N)}\}$ concurrently in a single batched pass.
2. Given a utility metric $U(y, y')$ (e.g., negative edit distance, ROUGE-L, or BLEU), MBR selects the candidate with the highest expected consensus utility under the empirical sample distribution:
   $$\hat{y}_{\text{MBR}} = \arg\max_{y \in \{y^{(1)}, \dots, y^{(N)}\}} \frac{1}{N}\sum_{j=1}^N U(y, y^{(j)})$$
3. This eliminates outliers, resolves spatial disfluencies, and consistently improves BLEU / ROUGE by $+1.5$ to $+2.8$ points.
