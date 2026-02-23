"""Suprematism Evolution -- Web UI Server.

Supports two engines:
  - "cppn"   : CPPN-NEAT (original Picbreeder-style)
  - "shapes" : Shape-genome evolution (explicit geometric primitives)

Launch:
    python -m src.server
    # or: uvicorn src.server:app --reload
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.neat.population import Population
from src.cppn.renderer import render_to_image
from src.art.palettes import get_palette_array, MALEVICH_PALETTE, SUPREMATIST_PALETTE
from src.art.fitness import compute_style_fitness
from src.cppn.activations import MALEVICH_ACTIVATIONS, ALL_ACTIVATIONS

from src.shapes.population import ShapePopulation
from src.shapes.renderer import render_genome

OUTPUT_DIR = Path("output")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Suprematism Evolution")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class _GenerationSnapshot:
    """Lightweight snapshot of a generation for history navigation."""
    __slots__ = ("generation", "genomes", "thumbnails")

    def __init__(self, generation: int, genomes: list, thumbnails: list[str]):
        self.generation = generation
        self.genomes = genomes
        self.thumbnails = thumbnails


class AppState:
    def __init__(self):
        self.engine: str = "shapes"
        self.pop: Population | None = None
        self.shape_pop: ShapePopulation | None = None
        self.mode: str = "malevich"
        self.palette: str = "none"
        self.color_mode: str = "rgb"
        self.pop_size: int = 20
        self.thumb_size: int = 320
        self.hires_size: int = 2048
        self.mutation_strength: float = 1.0
        self._cached_thumbnails: list[str] = []
        self._networks = []

        self._history: list[_GenerationSnapshot] = []
        self._history_idx: int = -1
        self._max_history: int = 200

    # ---- CPPN helpers ----

    def _activation_options(self) -> list[str]:
        if self.mode == "malevich":
            return MALEVICH_ACTIVATIONS
        return ALL_ACTIVATIONS

    def _build_cppn_config(self) -> dict:
        return {
            "pop_size": self.pop_size,
            "num_inputs": 5,
            "num_outputs": 3,
            "output_activation": "tanh",
            "activation_options": self._activation_options(),
            "weight_perturb_rate": 0.8,
            "weight_perturb_power": 0.5,
            "weight_replace_rate": 0.1,
            "add_node_rate": 0.03,
            "add_connection_rate": 0.05,
            "activation_mutation_rate": 0.1,
            "toggle_enable_rate": 0.01,
            "crossover_rate": 0.5,
            "compatibility_threshold": 3.0,
            "elitism": 1,
        }

    # ---- Shape helpers ----

    def _shape_palette(self) -> list[tuple]:
        return list(SUPREMATIST_PALETTE.values())

    def _build_shape_config(self) -> dict:
        return {
            "pop_size": self.pop_size,
            "num_palette_colors": len(SUPREMATIST_PALETTE),
            "elitism": 2,
            "crossover_rate": 0.7,
            "tournament_k": 3,
            "mutation_strength": self.mutation_strength,
        }

    # ---- Initialization ----

    def initialize(self):
        if self.engine == "shapes":
            self.shape_pop = ShapePopulation(self._build_shape_config())
            self.shape_pop.initialize()
            self.pop = None
            self._networks = []
        else:
            self.pop = Population(self._build_cppn_config())
            self.pop.initialize()
            self.shape_pop = None
        self._render_thumbnails()
        self._history = []
        self._history_idx = -1
        self._push_history()

    # ---- History ----

    def _push_history(self):
        genomes = self._current_genomes()
        gen = self._current_generation()
        genome_copies = [g.copy() for g in genomes]
        snap = _GenerationSnapshot(gen, genome_copies, list(self._cached_thumbnails))

        if self._history_idx < len(self._history) - 1:
            self._history = self._history[: self._history_idx + 1]

        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._history_idx = len(self._history) - 1

    def _restore_snapshot(self, idx: int):
        snap = self._history[idx]
        self._history_idx = idx
        self._cached_thumbnails = list(snap.thumbnails)

        if self.engine == "shapes" and self.shape_pop:
            self.shape_pop.genomes = [g.copy() for g in snap.genomes]
            self.shape_pop.generation = snap.generation

    def can_go_prev(self) -> bool:
        return self._history_idx > 0

    def can_go_next(self) -> bool:
        return self._history_idx < len(self._history) - 1

    def go_prev(self):
        if self.can_go_prev():
            self._restore_snapshot(self._history_idx - 1)

    def go_next(self):
        if self.can_go_next():
            self._restore_snapshot(self._history_idx + 1)

    # ---- Thumbnails ----

    def _render_thumbnails(self):
        self._cached_thumbnails = []

        if self.engine == "shapes":
            palette = self._shape_palette()
            for genome in self.shape_pop.genomes:
                img = render_genome(genome, self.thumb_size, self.thumb_size, palette)
                self._cached_thumbnails.append(_image_to_base64(img))
        else:
            palette_array = get_palette_array(self.palette)
            self._networks = self.pop.get_networks()
            for net in self._networks:
                img = render_to_image(net, self.thumb_size, self.thumb_size, self.color_mode, palette_array)
                self._cached_thumbnails.append(_image_to_base64(img))

    # ---- State payload ----

    def _current_genomes(self) -> list:
        if self.engine == "shapes":
            return self.shape_pop.genomes
        return self.pop.genomes

    def _current_generation(self) -> int:
        if self.engine == "shapes":
            return self.shape_pop.generation
        return self.pop.generation

    def get_state_payload(self) -> dict:
        return {
            "generation": self._current_generation(),
            "pop_size": len(self._current_genomes()),
            "engine": self.engine,
            "mode": self.mode,
            "palette": self.palette,
            "color_mode": self.color_mode,
            "thumb_size": self.thumb_size,
            "mutation_strength": self.mutation_strength,
            "thumbnails": self._cached_thumbnails,
            "has_prev": self.can_go_prev(),
            "has_next": self.can_go_next(),
        }

    # ---- Evolve ----

    def evolve(self, selected: list[int]):
        if self.engine == "shapes":
            self.shape_pop.mutation_strength = self.mutation_strength
            self.shape_pop.evolve_with_selection(selected)
        else:
            palette_array = get_palette_array(self.palette)
            networks = self._networks

            fitness = [0.0] * len(self.pop.genomes)
            for idx in selected:
                if 0 <= idx < len(fitness):
                    fitness[idx] = 1.0

            if self.mode != "malevich":
                for i, net in enumerate(networks):
                    img = render_to_image(net, 64, 64, self.color_mode)
                    img_np = np.array(img.convert("RGB")).astype(np.float64) / 255.0
                    style_bonus = compute_style_fitness(img_np, self.mode, weight=0.2)
                    fitness[i] += style_bonus

            self.pop.set_fitness(fitness)
            self.pop.evolve()

        self._render_thumbnails()
        self._push_history()

    # ---- Export ----

    def export_hires(self, index: int) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)
        gen = self._current_generation()

        if self.engine == "shapes":
            palette = self._shape_palette()
            genome = self.shape_pop.genomes[index]
            img = render_genome(genome, self.hires_size, self.hires_size, palette)
        else:
            palette_array = get_palette_array(self.palette)
            net = self._networks[index]
            img = render_to_image(net, self.hires_size, self.hires_size, self.color_mode, palette_array)

        path = OUTPUT_DIR / f"hires_{gen:04d}_{index:02d}.png"
        img.save(path)
        return _image_to_base64(img)

    def export_all_zip(self, resolution: int = 512) -> io.BytesIO:
        """Render all current individuals and return a zip file in memory."""
        buf = io.BytesIO()
        gen = self._current_generation()

        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            genomes = self._current_genomes()
            for i, genome in enumerate(genomes):
                if self.engine == "shapes":
                    palette = self._shape_palette()
                    img = render_genome(genome, resolution, resolution, palette)
                else:
                    palette_array = get_palette_array(self.palette)
                    net = self._networks[i]
                    img = render_to_image(net, resolution, resolution, self.color_mode, palette_array)

                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"gen{gen:04d}_{i:02d}.png", img_buf.getvalue())

        buf.seek(0)
        return buf

    # ---- Save ----

    def save_genome(self, index: int) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)
        gen = self._current_generation()
        path = OUTPUT_DIR / f"genome_{gen:04d}_{index:02d}.json"

        if self.engine == "shapes":
            self.shape_pop.save_genome(index, path)
        else:
            self.pop.save_genome(index, path)

        return str(path)


state = AppState()


def _image_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EvolveRequest(BaseModel):
    selected: list[int]

class IndexRequest(BaseModel):
    index: int

class ResetRequest(BaseModel):
    engine: str = "shapes"
    mode: str = "malevich"
    palette: str = "none"
    color_mode: str = "rgb"
    pop_size: int = 20
    mutation_strength: float = 1.0

class MutationRequest(BaseModel):
    mutation_strength: float = 1.0


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/state")
def api_state():
    if state.engine == "shapes" and state.shape_pop is None:
        state.initialize()
    elif state.engine == "cppn" and state.pop is None:
        state.initialize()
    return JSONResponse(state.get_state_payload())


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest):
    if state.engine == "shapes" and state.shape_pop is None:
        state.initialize()
    elif state.engine == "cppn" and state.pop is None:
        state.initialize()
    state.evolve(req.selected)
    return JSONResponse(state.get_state_payload())


@app.post("/api/export")
def api_export(req: IndexRequest):
    genomes = state._current_genomes()
    if not genomes:
        return JSONResponse({"error": "No population"}, status_code=400)
    if not (0 <= req.index < len(genomes)):
        return JSONResponse({"error": "Invalid index"}, status_code=400)
    b64 = state.export_hires(req.index)
    return JSONResponse({"image": b64, "index": req.index})


@app.get("/api/export_all")
def api_export_all():
    genomes = state._current_genomes()
    if not genomes:
        return JSONResponse({"error": "No population"}, status_code=400)
    gen = state._current_generation()
    buf = state.export_all_zip(resolution=512)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=gen{gen:04d}_all.zip"},
    )


@app.post("/api/save")
def api_save(req: IndexRequest):
    genomes = state._current_genomes()
    if not genomes:
        return JSONResponse({"error": "No population"}, status_code=400)
    if not (0 <= req.index < len(genomes)):
        return JSONResponse({"error": "Invalid index"}, status_code=400)
    path = state.save_genome(req.index)
    return JSONResponse({"path": path, "index": req.index})


@app.post("/api/reset")
def api_reset(req: ResetRequest):
    state.engine = req.engine
    state.mode = req.mode
    state.palette = req.palette
    state.color_mode = req.color_mode
    state.pop_size = req.pop_size
    state.mutation_strength = req.mutation_strength
    state.initialize()
    return JSONResponse(state.get_state_payload())


@app.post("/api/mutation_strength")
def api_mutation_strength(req: MutationRequest):
    state.mutation_strength = max(0.0, min(3.0, req.mutation_strength))
    if state.engine == "shapes" and state.shape_pop:
        state.shape_pop.mutation_strength = state.mutation_strength
    return JSONResponse({"mutation_strength": state.mutation_strength})


@app.post("/api/prev")
def api_prev():
    state.go_prev()
    return JSONResponse(state.get_state_payload())


@app.post("/api/next")
def api_next():
    state.go_next()
    return JSONResponse(state.get_state_payload())


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    state.initialize()
    print("Starting server at http://localhost:8000")
    webbrowser.open("http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
