# NCM-Apex: Research Blog Style Guide & Visual Blueprint

## 1. Aesthetic DNA: Vagabond Dark Theme (Ink & Parchment)

All blog deliverables in NCM-Apex must strictly conform to the established design DNA of the author's Astro research publication platform (`MYBLOG` / `astro-erudite`):

### 1.1 Color Tokens & Design System
```css
:root {
  /* Vagabond Dark Theme — Ink & Parchment */
  --gray-1:  #0a0a0a;  /* Deep ink black — canvas background */
  --gray-2:  #111110;  /* Charcoal wash — card / surface background */
  --gray-3:  #1a1916;  /* Dark woodgrain — code blocks / elevated surfaces */
  --gray-4:  #232119;  /* Aged timber — hover states */
  --gray-5:  #2d2a21;  /* Warm shadow — dropdowns */
  --gray-6:  #3a3528;  /* Border — worn parchment edge */
  --gray-7:  #4a4435;  /* Ink-stained border */
  --gray-8:  #6b6350;  /* Muted brush strokes */
  --gray-9:  #7a7260;  /* Mid-tone ink */
  --gray-10: #8a8070;  /* Faded calligraphy */
  --gray-11: #b8ad9a;  /* Warm parchment muted — secondary text */
  --gray-12: #e8e0d4;  /* Parchment white — primary headings & text */

  /* Accents */
  --red-9:   #c41e3a;  /* Samurai crimson — primary accent / highlights */
  --red-11:  #ff4d4d;  /* Blood red glow — warnings / critical badges */

  --background:         var(--gray-1);
  --foreground:         var(--gray-12);
  --primary:            var(--gray-12);
  --muted:              var(--gray-3);
  --muted-foreground:   var(--gray-11);
  --border:             var(--gray-6);
  --accent:             var(--red-9);
}
```

### 1.2 Typography Hierarchy
- **Primary Body Font:** IBM Plex Sans (weights 300, 400, 500, 600)
- **Monospace Code Font:** IBM Plex Mono
- **Mathematical / Formal Font:** KaTeX / Temml MathML
- **Heading Styles:**
  - `# Title`: 2.5rem, bold, letter-spacing -0.02em, parchment white (`--gray-12`)
  - `## Section`: 1.75rem, semi-bold, border-bottom 1px solid `--gray-6`, with crimson accent bar
  - `### Subsection`: 1.25rem, medium, parchment muted (`--gray-11`)

---

## 2. Established Narrative Voice & Tone

The writing style blends **elite AI research rigor** with **candid engineering authenticity** (inspired by the *Vagabond* warrior-scholar archetype):

### 2.1 Narrative Structure
1. **Epigraph / Philosophical Hook:** An opening quote setting the strategic or philosophical theme (Miyamoto Musashi, Vagabond, Sun Tzu, or classic philosophy).
   ```markdown
   > *"The ultimate aim of martial arts is not having to use them."*  
   > — Miyamoto Musashi, *Vagabond*
   ```
2. **1. Quick Info Card:** High-signal metadata table covering Codename, Core Paper, Benchmark Targets, Hardware (RTX 3070 Ti), and Speedup Multipliers.
3. **2. The Game Plan & The Rabbit Holes (Researcher Mindset):**
   - **The Immediate Thoughts:** Initial hypotheses and architectural questions.
   - **The Aha Moment:** The breakthrough insight that unlocked the solution.
   - **Rabbit Hole #1 & #2:** Honest engineering challenges (e.g., codebook collapse, straight-through estimator gradient explosion, attention mask boundary leaks) and how they were solved.
   - **The Breakthrough:** The exact mechanism that made it work.
   - **Facepalm Moment:** A humorous, relatable debugging detail (e.g., transposed coordinate dimensions, off-by-one scale indices).
4. **3. Mathematical & Architectural Blueprint:** Rigorous formulations, tensor dimensionalities, and LaTeX/Temml equations.
5. **4. Empirical Execution & Benchmark Telemetry:** Concrete numbers, tables, BLEU/ROUGE/accuracy curves, and hardware resource utilization.
6. **5. Side-by-Side Motion Showcase:** Video loops and visual proofs.

---

## 3. Formatting Standards: Math, Code & Callouts

### 3.1 Container Directive Callouts
The blog markdown engine supports native container directives rendering into interactive collapsible `<details>` blocks:

```markdown
:::note[Scale Discretization]
Tokens at scale $k$ are predicted conditionally independent given continuous representations $\hat{z}^{(k-1)}$.
:::

:::tip[Hardware Acceleration]
Using PyTorch SDPA with `mode="reduce-overhead"` cuts activation memory by 68% on the RTX 3070 Ti.
:::

:::warning[Exposure Bias]
Conditioning exclusively on ground-truth codes causes catastrophic error drift at inference time. Uniform codebook corruption is mandatory.
:::

:::important[Attention Causality]
Block-diagonal attention masks must strictly zero out connections from fine to coarse scales.
:::
```

### 3.2 Temml MathML Formulas
- **Inline Math:** Single dollar signs: `$z_k = \text{Down}_k(z)$`
- **Block Display Math:** Double dollar signs:
  ```latex
  $$\mathcal{L}_{\text{tok}} = -\frac{1}{L}\sum_{i=1}^L \log p_\psi(x^i \mid \hat{z}^{(k)}) + \beta \cdot \text{MSE}(z, \text{sg}[\hat{z}^{(k)}]) + \text{MSE}(\text{sg}[z], \hat{z}^{(k)})$$
  ```

### 3.3 Expressive Code Blocks
Code blocks must include language tags, file titles, and optional line highlighting:
````markdown
```python title="src/layers/quant.py" {8-12}
# Straight-through residual vector quantization
quantized = codebook[indices]
quantized_st = latents + (quantized - latents).detach()
```
````

---

## 4. Media & Motion Design Serving Rules

- **Video Loops:** Motion design clips (AR vs. NCM side-by-side, Sudoku step comparison) must be rendered as optimized H.264 / WebM files under 15MB.
- **Embedded Videos:** Served via HTML5 `<video>` tags with `autoplay loop muted playsinline`:
  ```html
  <video autoplay loop muted playsinline class="rounded-lg border border-[#3a3528] shadow-2xl w-full">
    <source src="../videos/sudoku_comparison.mp4" type="video/mp4" />
  </video>
  ```
- **Diagrams:** High-contrast SVG schematics placed in `assets/diagrams/` matching the dark ink-and-parchment palette.
