# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Picbreeder-style interactive evolution system for generating abstract art. Users select favorites from a population grid, and the system breeds the next generation from those selections. Two evolution engines:

- **Shapes engine** (default): Evolves explicit geometric primitives (Suprematist style) with hierarchical ShapeGroup/satellite structure
- **CPPN engine**: Evolves neural networks (CPPN + NEAT) that map pixel coordinates to colors
- **Hadid engine**: 3D architecton variant of shapes with oblique cuboid projection

## Commands

```bash
# Install dependencies
pip install -r requirements.txt  # numpy, Pillow, scikit-image, fastapi, uvicorn

# Run web UI (primary interface)
python -m src.server              # starts FastAPI at http://localhost:8000

# Run CLI
python -m src.main                              # shapes engine (default)
python -m src.main --engine cppn --mode malevich --palette malevich
python -m src.main --engine hadid               # 3D architecton mode

# Run server with auto-reload during development
uvicorn src.server:app --reload
```

No test suite exists. The test files in the root directory (test_hadid.py, test_*.png) are one-off visual verification scripts, not automated tests.

## Architecture

### Two Parallel Engines

Both engines share the same Picbreeder interaction model (select favorites → evolve → repeat) but have completely different genome representations:

**CPPN Engine** (`src/cppn/` + `src/neat/`):
- Genome = graph of nodes with heterogeneous activation functions + weighted connections
- NEAT evolves both topology and weights; speciation prevents premature convergence
- Rendering: evaluate CPPN at every pixel coordinate → RGB/HSV output
- Pipeline: `neat/genome.py` → `cppn/network.py` (topological sort + forward pass) → `cppn/renderer.py` (coordinate grid + batch eval)

**Shapes Engine** (`src/shapes/`):
- Genome = hierarchical structure: `ShapeGenome` → `ShapeGroup[]` (each with anchor + members) + `Shape[]` satellites
- Simple generational EA with tournament selection, no speciation needed
- Rendering: PIL-based drawing with 2x supersampling, canvas fiber texture, edge roughness, grain, color gradient post-processing
- 3D mode (Hadid): shapes have `depth`/`elevation` fields, rendered as oblique-projected cuboids with face shading

### Shared Art Layer (`src/art/`)

- `palettes.py`: Five named palettes (Malevich 6-color, Suprematist 16-color, Basquiat 8-color, Hybrid 9-color, Hadid 10-color) as `{name: (r,g,b)}` dicts with float [0,1] values
- `quantizer.py`: sRGB → Linear RGB → XYZ → CIE LAB perceptual color quantization (CPPN engine only)
- `fitness.py`: Optional style-bias fitness functions for malevich (flatness, whitespace, color simplicity, edge sharpness) and basquiat (contrast, texture, asymmetry, mark density)
- `compositor.py`: Multi-CPPN layer compositing with blend modes (normal, multiply, screen, overlay)

### Web Server (`src/server.py`)

FastAPI app with `AppState` singleton managing both engines. Key design:
- Thumbnails rendered server-side as base64 PNG, sent to frontend in JSON
- Generation history stored in `_history` list (up to 200 snapshots) for undo/redo navigation
- Engine selected via `engine` field: `"shapes"`, `"hadid"`, or `"cppn"`
- Frontend is a single HTML file at `src/static/index.html`

### Key Patterns

- All coordinates are normalized to [0,1] canvas space; pixel conversion happens at render time
- Shapes engine uses `_ensure_inside()` to clamp rotated bounding boxes within canvas
- Genome serialization is JSON-based with `to_dict()`/`from_dict()` pattern throughout
- Shapes renderer applies post-processing pipeline: canvas fiber texture → grain → warm-to-cool color gradient
- Module-level `GRAIN_STRENGTH` and `GRADIENT_STRENGTH` in `src/shapes/renderer.py` are mutated by the server for real-time settings

### Output

All generated images and genome JSON files go to `output/` (gitignored). Naming convention: `gen_XXXX_grid.png`, `gen_XXXX_NN.png`, `hires_XXXX_NN.png`, `genome_XXXX_NN.json`.
