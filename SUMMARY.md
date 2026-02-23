# Malevich x Basquiat: CPPN Generative Art System

## What This Project Is

This is a complete generative art system that uses **CPPN neural networks** (Compositional Pattern-Producing Networks) evolved via **NEAT** (NeuroEvolution of Augmenting Topologies) to create abstract artworks inspired by two artists at opposite ends of the abstraction spectrum:

- **Kazimir Malevich** (1879-1935) -- the father of Suprematism: pure geometric forms, restricted color palettes, floating compositions on white void
- **Jean-Michel Basquiat** (1960-1988) -- Neo-Expressionist: raw marks, dense layering, anatomical symbols, graffiti energy, crossed-out text

The user interacts with the system like **Picbreeder** -- selecting favorite images from a population grid, and the system evolves the next generation using the selected parents. Over many generations, this interactive evolutionary process produces images that can range from clean Suprematist geometry to chaotic Basquiat-like texture, and everything in between.

---

## How It Works (The Science)

### What is a CPPN?

A CPPN is a neural network that takes pixel coordinates as input and outputs a color. For every pixel `(x, y)` in the image, the network computes:

```
f(x, y, distance_from_center, angle, bias) -> (Red, Green, Blue)
```

The key innovation over regular neural networks is that **each node can use a different activation function**. Instead of every node using ReLU or sigmoid, a CPPN mixes functions like:

| Function | What it produces visually |
|----------|--------------------------|
| `sin(x)` | Repeating stripes and waves |
| `gaussian(x)` | Radial blobs, soft circles |
| `step(x)` | Hard edges, flat color regions |
| `sigmoid(x)` | Smooth gradients between two regions |
| `abs(x)` | Bilateral symmetry |
| `sawtooth(x)` | Repeating ramp patterns, hatching feel |

When these functions are **composed** (the output of one feeds into another), they create complex visual patterns: interference patterns, nested shapes, organic textures, and geometric forms. A CPPN with `step` and `gaussian` nodes tends to produce clean Malevich-like geometry; one with `sin`, `sawtooth`, and `square` nodes tends toward Basquiat-like complexity.

Because the CPPN is a continuous mathematical function over coordinates, it is **resolution-independent** -- the same network renders a 64x64 thumbnail and a 4096x4096 print with no loss of quality.

### What is NEAT?

NEAT is the evolutionary algorithm that evolves the CPPN's structure. It evolves both the **weights** (how strong each connection is) and the **topology** (which nodes exist and how they're connected). Evolution starts from a minimal network (just inputs directly connected to outputs) and incrementally adds complexity through mutation:

1. **Weight perturbation** -- nudge connection weights by small random amounts (most common mutation)
2. **Add node** -- split an existing connection by inserting a new hidden node with a random activation function
3. **Add connection** -- connect two previously unconnected nodes
4. **Activation mutation** -- change a hidden node's activation function (e.g., `sin` to `gaussian`)
5. **Toggle enable** -- enable or disable a connection

NEAT also implements **speciation** -- grouping similar genomes into species so that new structural innovations aren't immediately outcompeted. Each species shares fitness among its members, giving novel topologies time to optimize their weights.

### What is Picbreeder-Style Evolution?

Traditional evolution uses a mathematical fitness function. Picbreeder replaces this with **human aesthetic judgment**: the user IS the fitness function. You look at a grid of 20 images, click the ones you find interesting, and the system breeds the next generation from your selections. There is no "correct" answer -- the system explores whatever direction your taste guides it.

Key concepts:
- **Stepping stones**: Complex images are typically NOT found by searching for them directly. You stumble upon intermediate forms that look like something unexpected, then steer from there.
- **Branching**: You can save a genome at any point and start a new evolution session from it, allowing you to explore multiple directions from a single discovery.

---

## What Was Built

### Research Phase

Before writing code, three parallel research efforts gathered the source material:

**1. Malevich Suprematism Analysis**
- Catalogued 15+ iconic Suprematist works (Black Square, White on White, Supremus No. 56, Eight Red Rectangles, Airplane Flying, etc.)
- Documented the restricted shape vocabulary: square, rectangle, circle, cross, triangle, trapezoid, line, ellipse -- all flat-filled, hard-edged, no outlines or gradients
- Mapped the color rules: white background always, black dominant, red most common chromatic, yellow, blue, green -- flat unmodulated, one color per shape, 2-6 distinct colors per composition
- Quantified composition rules: dominant diagonal (~45 degrees), asymmetric balance, 5x-20x scale variation, 20-40% overlap probability, 5-15% edge margin, rotation 15-45 degrees off-axis
- All properties parameterized with numeric ranges for algorithmic reproduction

**2. Basquiat Style Analysis**
- Catalogued 12+ major works (Untitled Skull, Boy and Dog in a Johnnypump, Hollywood Africans, Dustheads, Riding with Death, Horn Players, etc.)
- Documented line quality: aggressive, gestural, bimodal weight (thin scratches vs. heavy oil stick), broken discontinuous contours, intentional drips
- Mapped the color palette: black dominant, red, blue, yellow, white, raw canvas, orange, brown -- high saturation, no blending, high contrast
- Catalogued 12 recurring symbols: three-pointed crown, skull/head, arrow, copyright symbol, halo, crossed-out words, anatomical diagrams, hobo signs, numbers/lists, boxing figures, jazz musicians, teeth
- Documented composition: all-over, grid-like sectioning, figure-ground ambiguity, text as visual form, 3-5 layer palimpsest structure
- Classified 10 mark-making types for generative reproduction

**3. CPPN/NEAT/Picbreeder Technical Research**
- Full CPPN architecture documentation: inputs (x, y, d, theta, bias), heterogeneous activation functions, composition creating complexity, arbitrary DAG topology
- Picbreeder interface mechanics: population grid, human selection, branching/forking, collaborative evolution
- NEAT algorithm details: genome encoding (node genes, connection genes), innovation numbers, five mutation operators, crossover with gene alignment, speciation with compatibility distance, fitness sharing, starting minimal
- Key papers: Stanley 2007 (CPPNs), Stanley & Miikkulainen 2002 (NEAT), Secretan et al. 2011 (Picbreeder)
- Surveyed existing implementations: neat-python, PyTorch-NEAT, neataptic, cppn-js, TensorFlow.js

**4. Reference Artwork Repository**
- Collected museum/gallery URLs for 6 Malevich and 7 Basquiat works
- Sources include MoMA, The Broad, Whitney Museum, WikiArt, Google Arts & Culture

### Implementation Phase (Phase 1: Core Engine)

The full Python engine was built from scratch (no external NEAT libraries), totaling **2,424 lines** across 16 files. Here is every module explained:

---

#### `src/cppn/activations.py` (97 lines)

The visual vocabulary of the system. Defines 13 activation functions, each operating element-wise on numpy arrays for vectorized computation:

- `sigmoid`, `tanh` -- smooth partitioning, good for figure/ground separation
- `sin`, `cos` -- repetition, stripes, interference patterns
- `gaussian` -- radial symmetry, soft blobs (circles, halos)
- `step` -- hard binary boundaries (Suprematist hard edges)
- `abs` -- bilateral symmetry (V-shapes)
- `relu` -- half-space selection
- `linear` -- gradients, passthrough
- `square` -- parabolic curves
- `sawtooth` -- repeating ramps (hatching, texture feel)
- `inv` -- sign negation
- `softplus` -- smooth ReLU variant

Functions are organized into artist-biased subsets:
- **Malevich set** (`MALEVICH_ACTIVATIONS`): sigmoid, step, gaussian, tanh, abs, linear -- favoring hard edges, flat regions, and clean geometry
- **Basquiat set** (`BASQUIAT_ACTIVATIONS`): sin, sawtooth, relu, tanh, gaussian, softplus, square -- favoring texture, repetition, and organic complexity

A registry dictionary `ACTIVATIONS` maps string names to functions, and `get_activation(name)` provides lookup.

---

#### `src/cppn/network.py` (143 lines)

The CPPN evaluation engine. The `CPPNNetwork` class:

1. Takes input/output node keys, per-node activation functions, and weighted connections
2. Builds an evaluation graph using **Kahn's topological sort algorithm** -- necessary because NEAT produces arbitrary DAG topologies, not neat layer structures
3. Pre-groups connections by destination node for fast lookup during forward pass
4. **`forward(inputs)`**: Batch-evaluates the network on an (N, 5) array of pixel coordinates. For each non-input node in topological order, it sums weighted inputs from connected nodes, applies the node's activation function, and stores the result. All operations are vectorized over N pixels simultaneously using numpy.
5. **`from_genome(genome)`**: Class method that extracts the CPPN phenotype from a NEAT genome -- collects enabled connections and their weights, maps node activations, and constructs the network.

The topological sort also performs a **reachability analysis** -- it walks backwards from output nodes to identify which hidden nodes actually contribute to the output, skipping disconnected nodes.

---

#### `src/cppn/renderer.py` (160 lines)

Converts CPPN networks into images. Key functions:

**`make_coordinate_grid(width, height)`**: Creates an (N, 5) array where N = width * height. Each row contains `[x, y, d, theta, bias]`:
- `x, y`: pixel coordinates normalized to [-1, 1]
- `d`: Euclidean distance from center (`sqrt(x^2 + y^2)`) -- enables radial patterns
- `theta`: angle from center (`atan2(y, x)`) -- enables spiral/rotational patterns
- `bias`: constant 1.0 -- acts as a learnable offset

**`render_cppn(network, width, height, color_mode, palette)`**: The main rendering pipeline:
1. Generates coordinate grid
2. Runs CPPN forward pass (all pixels in one batch)
3. Maps outputs to color:
   - **RGB mode**: 3 outputs mapped from [-1, 1] to [0, 1] for R, G, B
   - **HSV mode**: 3 outputs interpreted as Hue (cyclic), Saturation, Value, then converted to RGB via vectorized HSV-to-RGB conversion
4. Optionally quantizes to a palette by Euclidean distance in RGB space

**`render_population_grid(networks, thumb_size, cols, ...)`**: Renders a full population as a labeled grid image with dark padding between thumbnails. This is what the user sees each generation.

**`render_to_image()`**: Convenience wrapper returning a PIL Image object for saving.

---

#### `src/neat/genome.py` (293 lines)

The genetic encoding. Three data structures:

**`NodeGene`**: Represents a node in the network.
- `key`: unique integer ID
- `node_type`: "input", "hidden", or "output"
- `activation`: name of activation function (e.g., "sin", "gaussian")

**`ConnectionGene`**: Represents a connection between two nodes.
- `key`: tuple `(source_node, destination_node)`
- `weight`: float connection weight
- `enabled`: bool (disabled connections are "dormant genes" that can be re-enabled)

**`Genome`**: The complete genetic blueprint.
- `nodes`: dict of all node genes
- `connections`: dict of all connection genes
- `input_keys`, `output_keys`: tuples identifying which nodes are inputs/outputs
- `fitness`: score assigned during evaluation

**Creation**: `Genome.create_minimal(genome_key, num_inputs, num_outputs)` creates the simplest possible network: 5 input nodes (x, y, d, theta, bias), 3 output nodes (R, G, B), and 15 connections (every input to every output) with random weights. No hidden nodes. All complexity emerges through evolution.

**Five mutation operators**:

1. `mutate_weights()`: For each connection, with 80% probability perturb the weight by a Gaussian (sigma=0.5), with 10% probability replace it entirely. Weights clamped to [-4, 4].

2. `mutate_add_node()`: Picks a random enabled connection (say A->B with weight w), disables it, creates a new hidden node C with a random activation function, and adds two new connections: A->C (weight 1.0) and C->B (weight w). This preserves approximate network behavior while adding a new point of variation.

3. `mutate_add_connection()`: Picks two random nodes and adds a connection between them, respecting constraints:
   - Cannot connect TO input nodes
   - Cannot connect FROM output nodes
   - Cannot create cycles (verified via BFS reachability check)
   - If the connection already exists but is disabled, re-enables it

4. `mutate_activation()`: Changes a random hidden node's activation function to a different one from the available set.

5. `mutate_toggle_enable()`: Flips a random connection's enabled/disabled state.

`mutate(config)` applies all operators according to configurable rates (defaults: weight perturbation 80%, add node 3%, add connection 5%, activation change 10%, toggle 1%).

**Crossover**: `crossover(parent1, parent2, child_key)` implements NEAT crossover:
- Aligns parent genomes by connection key (the `(source, dest)` tuple serves as an innovation marker)
- **Matching genes** (same key in both parents): inherited randomly from either parent
- **Disjoint/excess genes** (present in one parent only): inherited from the fitter parent only
- Nodes from both parents are merged, preferring the fitter parent

**Compatibility distance**: `compatibility_distance(g1, g2)` measures genetic difference:
```
delta = c1 * (disjoint + excess) / N + c3 * avg_weight_diff
```
Used by speciation to group similar genomes.

---

#### `src/neat/evolution.py` (166 lines)

Manages speciation and reproduction.

**`Species`** dataclass:
- `representative`: a genome that defines the species' "center"
- `members`: list of genomes assigned to this species
- `adjusted_fitness`: sum of member fitness divided by species size (fitness sharing -- prevents any one species from dominating)

**`Reproducer`** class:

**`speciate(genomes, existing_species)`**: Assigns each genome to a species by comparing its compatibility distance to each species' representative. If no existing species is close enough (distance < threshold, default 3.0), a new species is created. Empty species are pruned.

**`reproduce(species_list, pop_size, ...)`**: Produces the next generation:
1. **Offspring allocation**: Each species gets offspring proportional to its adjusted fitness
2. **Elitism**: The best genome in each species is copied unchanged (default: 1 elite per species)
3. **Offspring creation**: For each remaining slot, either:
   - **Crossover** (50% chance): Two parents selected via tournament selection (k=3), fitter parent first, produce a child via NEAT crossover, then mutate
   - **Cloning** (50% chance): One parent selected, copied, and mutated
4. Population is padded or trimmed to exact `pop_size`

---

#### `src/neat/population.py` (206 lines)

The top-level manager that ties everything together.

**`DEFAULT_CONFIG`**: All hyperparameters in one place:
```python
{
    "num_inputs": 5,          # x, y, d, theta, bias
    "num_outputs": 3,         # R, G, B (or H, S, V)
    "pop_size": 20,           # 4x5 grid
    "output_activation": "tanh",
    "weight_perturb_rate": 0.8,
    "add_node_rate": 0.03,
    "add_connection_rate": 0.05,
    "activation_mutation_rate": 0.1,
    "crossover_rate": 0.5,
    "compatibility_threshold": 3.0,
    "elitism": 1,
    ...
}
```

**`Population`** class:

- **`initialize()`**: Creates `pop_size` minimal-topology genomes (5 inputs, 3 outputs, 15 connections each, random weights)
- **`get_networks()`**: Builds CPPN networks from all current genomes (ready for rendering)
- **`set_fitness(fitness_values)`**: Assigns fitness scores to each genome
- **`evolve()`**: Runs one full NEAT generation: speciate -> reproduce -> replace population
- **`evolve_with_selection(selected_indices)`**: Convenience method for Picbreeder-style selection -- selected genomes get fitness 1.0, others get 0.0, then evolves
- **`save_genome(index, path)`**: Serializes a genome to JSON
- **`load_genome(path)`**: Deserializes a genome from JSON
- **`branch_from(path)`**: Creates a new population where one individual is the loaded genome and the rest are mutated variants -- the "forking" mechanism from Picbreeder

Genome serialization stores all node genes (key, type, activation), connection genes (source, dest, weight, enabled), and structural metadata.

---

#### `src/art/palettes.py` (64 lines)

Artist-specific color palettes defined as named dictionaries of RGB float tuples:

**Malevich Suprematist Palette** (6 colors):
- White `#FFFFFF`, Black `#000000`, Red `#CC0000`, Yellow `#E8C800`, Blue `#0044AA`, Green `#228B22`

**Basquiat Palette** (8 colors):
- Raw Canvas `#F5E6D0`, Black `#000000`, Red `#CC2200`, Blue `#0055CC`, Yellow `#DDAA00`, White `#FFFFFF`, Orange `#CC6600`, Brown `#8B4513`

**Hybrid Palette** (9 colors): union of both

`get_palette_array(name)` returns the palette as an (N, 3) numpy array for use in quantization, or `None` for unrestricted color.

---

#### `src/art/quantizer.py` (85 lines)

Perceptually accurate color quantization using **CIE LAB** color space.

The full conversion pipeline: **sRGB -> Linear RGB -> XYZ (D65 illuminant) -> LAB**

Why LAB? In RGB space, the Euclidean distance between two colors does not correspond to how different they look to human eyes. In LAB space, equal distances correspond to equal perceived differences. This means when we snap a pixel to its nearest palette color, the result looks natural rather than having unexpected color jumps.

`quantize_lab(image, palette_name)` converts both the image pixels and the palette to LAB, finds the nearest palette color for each pixel by squared distance, and returns the quantized image in RGB.

---

#### `src/art/fitness.py` (143 lines)

Optional fitness functions that bias evolution toward artist-specific aesthetics. These are **added** to the human-selection fitness (which is still the primary driver).

**`malevich_fitness(image)`** scores four Suprematist qualities (each 0-1, averaged):
1. **White-space ratio**: Fraction of pixels with luminance > 0.85. Ideal is 30-70% (not all white, not no white). Scored as `1.0 - 2.0 * |ratio - 0.5|`
2. **Flatness**: Low local variance in 4x4 patches. `exp(-variance * 50)`. Flat color regions score high.
3. **Color simplicity**: Fewer distinct colors (quantized to 8 levels per channel). `exp(-unique_colors / 50)`.
4. **Edge sharpness**: Mean gradient magnitude. Some edges are good (between flat regions), scored as `min(1.0, gradient * 10)`.

**`basquiat_fitness(image)`** scores four Basquiat qualities:
1. **Contrast**: Luminance range (max - min). Full range = 1.0.
2. **Textural complexity**: High local variance in 4x4 patches. `1.0 - exp(-variance * 50)`. Opposite of Malevich's flatness.
3. **Asymmetry**: Mean absolute difference between left half and horizontally-flipped right half. `min(1.0, difference * 5)`.
4. **Mark density**: Mean gradient magnitude (lots of edges everywhere). `min(1.0, gradient * 15)`.

`compute_style_fitness(image, mode, weight=0.3)` applies the appropriate function scaled by `weight`. In "hybrid" mode, returns 0 (pure human selection).

---

#### `src/art/compositor.py` (88 lines)

Multi-CPPN layer compositing for advanced artwork creation.

`composite_layers(layers, width, height, ...)` renders multiple CPPN networks as stacked layers with:
- **Per-layer color CPPN**: Each layer has its own CPPN generating RGB/HSV color
- **Per-layer alpha CPPN** (optional): A separate CPPN that generates a per-pixel opacity mask
- **Overall opacity**: A scalar controlling the layer's global transparency
- **Blend modes**: "normal" (standard alpha), "multiply" (darkens), "screen" (lightens), "overlay" (contrast enhancement)
- **Palette quantization**: Each layer can use a different palette

Layers are composited bottom-to-top onto a configurable background (default white). This enables workflows like:
- Layer 1: Suprematist geometry CPPN (clean shapes, Malevich palette)
- Layer 2: Texture CPPN (Basquiat-like marks, multiply blend, 50% opacity)
- Layer 3: Mark overlay CPPN (high-contrast, screen blend)

---

#### `src/main.py` (234 lines)

The interactive CLI entry point.

**Command-line arguments**:
- `--mode malevich|basquiat|hybrid` -- Style bias (controls which activation functions are available and whether style fitness is applied)
- `--palette malevich|basquiat|hybrid|none` -- Palette quantization
- `--color-mode rgb|hsv` -- How CPPN outputs are interpreted as color
- `--pop-size N` -- Population size (default 20)
- `--thumb-size N` -- Thumbnail pixel size (default 128)
- `--hires N` -- High-res export size (default 1024)
- `--branch path.json` -- Start from a saved genome

**Main loop** (each generation):
1. Build CPPN networks from all genomes
2. Render population grid image and save to `output/gen_XXXX_grid.png`
3. Render and save individual thumbnails as `output/gen_XXXX_NN.png`
4. Print a numbered index showing the grid layout
5. Accept user command:
   - **Space-separated indices** (e.g., `2 5 11 17`): Select these as parents, compute fitness (1.0 for selected + optional style bonus), evolve next generation
   - **`s <idx>`**: Save genome to `output/genome_XXXX_NN.json`
   - **`e <idx>`**: Export at high resolution to `output/hires_XXXX_NN.png`
   - **`q`**: Quit

---

### Reference Documents

#### `references/artwork_references.md`
Museum and WikiArt URLs for 6 Malevich works (Black Square, White on White, Supremus No. 56, Eight Red Rectangles, Airplane Flying, Red Square and Black Square) and 7 Basquiat works (Untitled Skull, Boy and Dog in a Johnnypump, Hollywood Africans, Dustheads, In This Case, Irony of a Negro Policeman, Untitled 1982).

#### `references/malevich_style_analysis.md`
Parameterized formal properties of Suprematism: shape vocabulary, color rules, composition rules (dominant diagonal, asymmetry, floating forms, scale hierarchy, overlap, edge margins, rotation), with numeric ranges for each parameter.

#### `references/basquiat_style_analysis.md`
Parameterized formal properties of Basquiat's work: line quality, color palette, 8-entry symbol vocabulary table, composition patterns, 10-category mark-making taxonomy, with numeric ranges for each parameter.

#### `docs/PROJECT_OUTLINE.md`
The full system design document covering: why Malevich + Basquiat works as a combination, complete system architecture (CPPN design, NEAT parameters, style modes, palette constraints), Picbreeder-style interface design with ASCII mockups, multi-CPPN composition architecture, three-phase implementation plan, performance analysis, comparison of CPPN vs GAN/Diffusion approaches, limitations and mitigations, and artistic goals.

---

## Testing

All 12 integration tests pass:

1. **Activations**: All 13 functions produce correct-shape numpy outputs
2. **Genome creation**: Minimal topology creates 8 nodes (5 input + 3 output) and 15 connections
3. **CPPN construction**: Network builds from genome with correct input/output counts
4. **Rendering**: Produces (64, 64, 3) float array in valid range
5. **Palette quantization**: Reduces to palette color count
6. **Mutation**: Adds nodes and connections (9 nodes, 18 connections after add-node + add-connection)
7. **Population**: Creates 12-member population, renders grid image
8. **Evolution**: Runs selection + evolution, advances generation counter
9. **Fitness scoring**: Both Malevich and Basquiat fitness return values in expected range
10. **Genome persistence**: Save to JSON and load back with matching structure
11. **LAB quantizer**: Perceptual color quantization produces correct color count
12. **Compositor**: Multi-layer compositing with blend modes produces valid image

Sample images were generated by running 20 generations of random evolution with elevated mutation rates, producing CPPNs with 4-8 hidden nodes, 35-42 connections, and diverse activation function mixes (sigmoid, step, sin, sawtooth, square, tanh, etc.).

---

## How to Use

```bash
cd /Users/raunit/Downloads/malevich

# Basic: hybrid mode, full color, pure human selection
python3 -m src.main

# Malevich mode: Suprematist activations + palette + style fitness bias
python3 -m src.main --mode malevich --palette malevich

# Basquiat mode: textural activations + his palette + complexity bias
python3 -m src.main --mode basquiat --palette basquiat

# HSV color mode (often produces richer palettes)
python3 -m src.main --color-mode hsv

# Branch from a previously saved genome
python3 -m src.main --branch output/genome_0004_05.json

# Larger population with bigger thumbnails
python3 -m src.main --pop-size 30 --thumb-size 192
```

**During a session**:
- Look at the grid image saved in `output/` (open it in Preview, Finder, or any image viewer)
- Type the indices of images you like: `3 8 14 19`
- The system evolves and saves the next grid
- Use `s 8` to save a favorite genome for later branching
- Use `e 8` to export a 1024x1024 high-res version
- Use `q` to quit

---

## Project Statistics

| Category | Count |
|----------|-------|
| Total source lines | 2,424 |
| Python modules | 11 |
| Reference documents | 4 |
| Activation functions | 13 |
| Artist palettes | 3 (Malevich 6 colors, Basquiat 8 colors, Hybrid 9 colors) |
| NEAT mutation operators | 5 |
| Blend modes | 4 (normal, multiply, screen, overlay) |
| CPPN inputs per pixel | 5 (x, y, d, theta, bias) |
| CPPN outputs per pixel | 3 (R, G, B or H, S, V) |
| External dependencies | 2 (numpy, Pillow) |

---

## What's Next (Phase 2 & 3)

**Phase 2: Web Interface** -- Replace the CLI with a browser-based Picbreeder UI using either Gradio (fast prototype) or React + FastAPI (full app). Clickable thumbnail grid, real-time rendering, genome graph visualization.

**Phase 3: Advanced Features** -- Multi-CPPN layer composition UI, real-time weight sliders, animation mode (time parameter), collaborative gallery with genome sharing via URL, and symbol overlay system for Basquiat motifs (crowns, skulls, text).
