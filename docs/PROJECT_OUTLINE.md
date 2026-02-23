# Generative Art System: Malevich x Basquiat via CPPN + Interactive Evolution

## Vision

A Picbreeder-style interactive evolution platform that uses CPPN (Compositional Pattern-Producing Networks) evolved via NEAT to generate artworks fusing two polar aesthetic traditions:

- **Malevich's Suprematism** -- geometric purity, floating forms, restricted palette, cosmic weightlessness
- **Basquiat's Neo-Expressionism** -- raw marks, layered text, anatomical symbols, graffiti energy

The system lets users evolve images by selecting favorites from a population, gradually steering CPPNs toward outputs that channel these artistic vocabularies.

---

## Part 1: Why This Combination Works

### The Tension is the Point

Malevich and Basquiat sit at opposite ends of abstraction:

| Dimension | Malevich | Basquiat |
|-----------|----------|----------|
| Form | Pure geometry (squares, circles, rectangles) | Crude figuration (skulls, crowns, bodies) |
| Line | No outlines; filled shapes with hard edges | Aggressive, broken contour lines |
| Color | Flat, unmodulated primaries on white | High-contrast, layered, raw canvas showing |
| Text | None | Central -- crossed-out words, lists, annotations |
| Space | Infinite white void, floating forms | Horror vacui, every surface activated |
| Philosophy | Spiritual purity through reduction | Identity, history, rage through accumulation |

A CPPN can inhabit the *continuum* between these poles. The mathematical properties of CPPNs (symmetry from `gaussian`, repetition from `sin`, sharp boundaries from `step`) can produce both the clean geometry of Suprematism and the chaotic layering of Basquiat -- and everything in between.

### What CPPNs Bring to This

- **Resolution independence**: Works can be rendered at any scale, from thumbnails to gallery prints
- **Smooth interpolation**: Users can evolve gradually from Malevich-like compositions toward Basquiat-like complexity (or discover unexpected hybrids)
- **Mathematical regularity**: CPPNs naturally produce symmetry, repetition, and pattern -- echoing Suprematist geometry
- **Compositional chaos**: Deep CPPNs with many activation types produce organic, layered, almost textural output -- echoing Basquiat's density

---

## Part 2: System Architecture

### 2.1 Core Pipeline

```
[NEAT Genome] --> [CPPN Network] --> [Pixel Evaluation] --> [Image]
      ^                                                        |
      |                                                        v
  [Mutation/                                            [User Selection]
   Crossover]  <---------------------------------------------|
```

### 2.2 CPPN Design

**Inputs (per pixel):**

| Input | Range | Purpose |
|-------|-------|---------|
| `x` | [-1, 1] | Horizontal position |
| `y` | [-1, 1] | Vertical position |
| `d` | [0, ~1.41] | Distance from center (`sqrt(x^2 + y^2)`) |
| `theta` | [-pi, pi] | Angle from center (`atan2(y, x)`) |
| `bias` | 1.0 | Learnable offset |

**Outputs (per pixel):**

Two output modes, selectable per session:

*Mode A: RGB Direct*
- 3 outputs: R, G, B (via `sigmoid`, mapped to [0, 1])

*Mode B: HSV (recommended for richer palettes)*
- 3 outputs: H (via `sin`, cyclic), S (via `sigmoid`), V (via `sigmoid`)
- Convert to RGB for display

**Activation Function Set:**

| Function | What it produces | Malevich affinity | Basquiat affinity |
|----------|-----------------|-------------------|-------------------|
| `sin(x)` | Stripes, waves, repetition | Medium (parallel forms) | Low |
| `cos(x)` | Phase-shifted repetition | Medium | Low |
| `gaussian(x)` | Radial blobs, soft bumps | High (circular forms) | Medium (halos) |
| `sigmoid(x)` | Smooth binary partition | High (figure/ground) | Medium |
| `tanh(x)` | Smooth partition [-1,1] | High | Medium |
| `step(x)` | Hard binary boundary | Very High (hard edges) | Medium (stencil-like) |
| `abs(x)` | Bilateral symmetry | Medium | Low |
| `linear(x)` | Gradients | Low | Medium (washes) |
| `square(x)` | Parabolic curves | Medium | Low |
| `sawtooth(x)` | Repeating ramps | Low | Medium (hatching feel) |
| `noise(x)` | Stochastic texture | Low | High (rawness) |

**Custom "style-biased" activation functions (optional, advanced):**

- `suprematist_step(x)`: Quantizes output to a small set of discrete values (mimics flat color patches)
- `basquiat_jitter(x)`: `x + small_random_perturbation` (mimics hand-drawn imprecision)

### 2.3 NEAT Evolution Parameters

```yaml
population_size: 20          # Grid of 4x5 thumbnails
initial_topology: minimal    # Input -> Output only (no hidden nodes)
weight_init_range: [-2.0, 2.0]

mutation:
  weight_perturbation_rate: 0.8
  weight_perturbation_power: 0.5
  weight_replace_rate: 0.1
  add_node_rate: 0.03
  add_connection_rate: 0.05
  activation_mutation_rate: 0.1
  toggle_enable_rate: 0.01

crossover:
  enabled: true                # When user selects 2+ parents
  weight_inheritance: random   # Matching genes from random parent

speciation:
  compatibility_threshold: 3.0
  excess_coefficient: 1.0
  disjoint_coefficient: 1.0
  weight_diff_coefficient: 0.5

elitism: 1                    # Keep best genome per species
```

### 2.4 Style Modes (Fitness Biasing)

While the primary fitness is human selection (the user picks favorites), the system can offer optional **style nudges** -- soft fitness bonuses that bias evolution toward one artist's aesthetic:

**Malevich Mode:**
- Bonus for images with large flat-colored regions (low local variance)
- Bonus for high white-space ratio (>40% of pixels near white)
- Bonus for limited color count (penalize >5 distinct hue clusters)
- Bonus for hard edges (high gradient magnitude at region boundaries)

**Basquiat Mode:**
- Bonus for high local variance (textural complexity)
- Bonus for mark density (fewer large uniform regions)
- Bonus for high contrast (wide luminance histogram)
- Bonus for asymmetry (low self-correlation under reflection)

**Hybrid Mode (default):**
- No computational fitness bias
- Pure human selection
- The user IS the fitness function

### 2.5 Palette Constraints (Optional Layer)

Post-process CPPN output through a **palette quantization** step:

**Suprematist Palette:**
```
#FFFFFF (white -- background)
#000000 (black)
#CC0000 (red)
#E8C800 (yellow)
#0044AA (blue)
#228B22 (green)
```

**Basquiat Palette:**
```
#F5E6D0 (raw canvas)
#000000 (black)
#CC2200 (red)
#0055CC (blue)
#DDAA00 (yellow)
#FFFFFF (white)
#CC6600 (orange)
#8B4513 (brown)
```

**Quantization method:** For each pixel, find the nearest palette color (in LAB color space for perceptual accuracy). This forces CPPN output into the artist's chromatic vocabulary while preserving the evolved composition.

---

## Part 3: Interface Design (Picbreeder-Style)

### 3.1 Main Evolution View

```
+----------------------------------------------------------------+
|  [Malevich Mode] [Hybrid Mode] [Basquiat Mode]    [Settings]   |
+----------------------------------------------------------------+
|                                                                  |
|   +------+  +------+  +------+  +------+  +------+              |
|   | img1 |  | img2 |  | img3 |  | img4 |  | img5 |              |
|   +------+  +------+  +------+  +------+  +------+              |
|                                                                  |
|   +------+  +------+  +------+  +------+  +------+              |
|   | img6 |  | img7 |  | img8 |  | img9 |  | img10|              |
|   +------+  +------+  +------+  +------+  +------+              |
|                                                                  |
|   +------+  +------+  +------+  +------+  +------+              |
|   | img11|  | img12|  | img13|  | img14|  | img15|              |
|   +------+  +------+  +------+  +------+  +------+              |
|                                                                  |
|   +------+  +------+  +------+  +------+  +------+              |
|   | img16|  | img17|  | img18|  | img19|  | img20|              |
|   +------+  +------+  +------+  +------+  +------+              |
|                                                                  |
|   [Evolve Next Generation]  [Save Selected]  [Undo Generation]  |
+----------------------------------------------------------------+
```

- Click to select/deselect thumbnails (highlighted border)
- "Evolve" breeds next generation from selected parents
- Thumbnails rendered at 128x128 for speed

### 3.2 Detail View

- Click-and-hold or double-click a thumbnail to see full resolution (1024x1024+)
- Side panel shows the CPPN graph topology (nodes and connections visualized)
- Sliders to manually tweak connection weights in real-time
- Export to PNG/SVG at arbitrary resolution

### 3.3 Composition Tools (Post-CPPN)

These operate on the CPPN output to push it further toward artist-specific aesthetics:

**Suprematist Compositor:**
- Threshold CPPN output into discrete shapes
- Extract connected regions and render as clean geometric forms
- Apply palette quantization
- Place on white background with margin

**Basquiat Compositor:**
- Overlay evolved text-like marks (from a symbol CPPN or glyph library)
- Add crown symbols at local maxima of the image
- Apply scratchy line overlay using a secondary CPPN evolved for line patterns
- Layer multiple CPPN outputs with partial transparency

### 3.4 Gallery & Branching

- Save any evolved image + its genome to a local gallery
- Branch from any saved genome to start a new evolution session
- Export genome as JSON for sharing
- Import genomes from others

---

## Part 4: Multi-CPPN Composition (Advanced)

Rather than a single CPPN producing the final image, use **multiple CPPNs** composited together:

### Layer Stack

```
Layer 0: Background CPPN      -> white/canvas base tone
Layer 1: Geometry CPPN         -> large color regions (Malevich shapes)
Layer 2: Texture CPPN          -> surface variation (Basquiat rawness)
Layer 3: Line CPPN             -> contour marks, edges
Layer 4: Symbol CPPN           -> crown, skull, arrow motifs
Layer 5: Text CPPN             -> glyph-like patterns
```

Each layer has:
- Its own independently evolved CPPN
- An **alpha/mask CPPN** controlling where that layer is visible
- Blending mode (normal, multiply, screen, overlay)

Users can evolve layers independently or together, allowing:
- Evolve a Suprematist geometry in Layer 1, then overlay Basquiat-like marks in Layer 3
- Keep a clean background while evolving chaotic foreground textures
- Mix and match layers from different evolution sessions

---

## Part 5: Implementation Plan

### Phase 1: Core Engine (Python)

**Stack:** Python + NumPy + neat-python + Pillow/PIL

```
src/
  cppn/
    network.py        # CPPN forward pass, topological sort
    activations.py    # All activation functions
    renderer.py       # Coordinate grid generation, batch evaluation
  neat/
    genome.py         # NEAT genome encoding
    evolution.py      # Mutation, crossover, speciation
    population.py     # Population management
  art/
    palettes.py       # Suprematist and Basquiat color palettes
    quantizer.py      # Palette quantization (LAB space)
    fitness.py        # Optional style-bias fitness functions
    compositor.py     # Multi-layer composition
```

Deliverables:
- [ ] CPPN evaluation producing RGB images from genomes
- [ ] NEAT evolution loop with all mutation operators
- [ ] Palette quantization to Malevich/Basquiat palettes
- [ ] Render at arbitrary resolution
- [ ] Save/load genomes as JSON

### Phase 2: Interactive Interface (Web)

**Stack:** Python backend (FastAPI) + React/TypeScript frontend OR pure Python with Gradio/Streamlit

Option A -- **Gradio** (fastest to prototype):
```python
import gradio as gr

def evolve_and_render(selected_indices, mode):
    # breed next generation, return grid of images
    ...

interface = gr.Interface(...)
```

Option B -- **Web app** (better UX, closer to Picbreeder):
```
frontend/
  src/
    components/
      EvolutionGrid.tsx    # 4x5 thumbnail grid with selection
      DetailView.tsx       # Full-res view + genome graph
      StyleControls.tsx    # Mode selector, palette, parameters
      Gallery.tsx          # Saved genomes browser
    api/
      evolution.ts         # REST calls to backend
backend/
  api/
    routes.py              # /evolve, /render, /save, /load
    state.py               # Population state management
```

Deliverables:
- [ ] Clickable grid of CPPN-generated thumbnails
- [ ] Select parents -> Evolve -> See next generation
- [ ] Style mode toggle (Malevich / Hybrid / Basquiat)
- [ ] Full-resolution rendering and export
- [ ] Genome save/load/branch

### Phase 3: Advanced Features

- [ ] Multi-CPPN layer composition
- [ ] Real-time weight sliders for manual CPPN tweaking
- [ ] CPPN topology visualization (graph view)
- [ ] Palette editor (custom artist palettes)
- [ ] Animation mode (evolve `t` parameter for animated CPPNs)
- [ ] Collaborative gallery (share genomes via URL)

---

## Part 6: Technical Feasibility Notes

### Performance

- **128x128 thumbnail**: 16,384 pixels. A CPPN with ~20 nodes evaluates in <10ms (NumPy). 20 thumbnails = ~200ms total. Interactive.
- **1024x1024 full res**: ~1M pixels. ~500ms with NumPy, <50ms with GPU (PyTorch/JAX). Acceptable for on-demand rendering.
- **GPU acceleration**: Trivially parallelizable -- each pixel is independent. PyTorch or JAX batch evaluation gives 10-100x speedup.

### Why CPPN Over GAN/Diffusion?

| Aspect | CPPN + NEAT | GAN/Diffusion |
|--------|-------------|---------------|
| Training data needed | None | Thousands of images |
| Interpretability | Full (you can see the network graph) | Black box |
| User control | Direct (selection drives evolution) | Indirect (text prompts) |
| Resolution | Infinite (continuous function) | Fixed (or upscaled) |
| Novelty | Genuinely novel patterns | Interpolation of training data |
| Compute | CPU is sufficient | GPU required |
| Artistic ownership | No training data = no copyright issues | Trained on others' art |

### Limitations & Mitigations

| Limitation | Mitigation |
|------------|------------|
| CPPNs don't naturally produce recognizable objects (faces, crowns, text) | Use multi-layer composition with symbol overlays; accept abstraction as a feature |
| Evolution can be slow to converge on complex patterns | Start from branched genomes rather than random; use larger populations |
| Basquiat's text integration is hard to evolve | Use a glyph/symbol library composited on top of CPPN layers, or train a separate text-pattern CPPN |
| Color palette constraint reduces CPPN expressiveness | Apply quantization as a post-process, not a constraint on the CPPN itself |

---

## Part 7: Artistic Goals

### What Success Looks Like

1. **A Suprematist CPPN** that produces an image with clean geometric regions, limited palette, floating forms on white -- recognizably in the spirit of Malevich without copying any specific work

2. **A Basquiat CPPN** that produces dense, high-contrast, layered textures with crown-like protrusions, skull-like voids, and mark-density that evokes his canvases

3. **A Hybrid** that nobody has seen before -- the geometric clarity of Suprematism disrupted by Basquiat's raw energy, or Basquiat's chaos organized by Suprematist spatial logic

4. **Emergent surprises** -- the whole point of Picbreeder is that users discover images they never would have designed intentionally. The Malevich/Basquiat framing provides aesthetic direction, but the real magic is what the CPPNs invent on their own.

---

## References

- Stanley, K.O. (2007). "Compositional Pattern Producing Networks: A Novel Abstraction of Development." *Genetic Programming and Evolvable Machines*, 8(2):131-162.
- Stanley, K.O. & Miikkulainen, R. (2002). "Evolving Neural Networks through Augmenting Topologies." *Evolutionary Computation*, 10(2):99-127.
- Secretan, J. et al. (2011). "Picbreeder: A Case Study in Collaborative Evolutionary Exploration of Design Space." *Evolutionary Computation*, 19(3):373-403.
- See `references/artwork_references.md` for specific artwork URLs.
