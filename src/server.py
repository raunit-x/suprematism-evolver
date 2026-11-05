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
import os
import threading
import uuid
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from src.neat.population import Population
from src.cppn.renderer import render_to_image
from src.art.palettes import get_palette_array, MALEVICH_PALETTE, SUPREMATIST_PALETTE, HADID_PALETTE, MAPS_PALETTE
from src.art.fitness import compute_style_fitness
from src.cppn.activations import MALEVICH_ACTIVATIONS, ALL_ACTIVATIONS

from src.shapes.population import ShapePopulation
from src.shapes.renderer import render_genome, render_genome_extended
from src import render_pool

OUTPUT_DIR = Path("output")
STATIC_DIR = Path(__file__).parent / "static"


def _prefetch_home():
    """Warm the worker pool and pre-render the landing-page galleries so the
    first visitor doesn't wait for them."""
    try:
        render_pool.warm_up()
        state.get_home_samples()
        _build_infographic()
    except Exception as exc:  # a failed warm-up must never take the server down
        print(f"[prefetch] skipped: {exc}")


@asynccontextmanager
async def lifespan(application: FastAPI):
    OUTPUT_DIR.mkdir(exist_ok=True)
    state.initialize()
    threading.Thread(target=_prefetch_home, daemon=True).start()
    yield
    render_pool.shutdown()


app = FastAPI(title="Suprematism Evolution", lifespan=lifespan)
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
    """A generation for history navigation.

    ``thumbnails`` holds base64 JPEGs, with ``None`` for any not rendered
    yet -- they are filled in as the stream endpoint completes them.
    ``render_key`` records the render settings the thumbnails were drawn with
    so they are discarded if those settings later change.
    """
    __slots__ = ("rev", "generation", "genomes", "thumbnails", "render_key")

    def __init__(self, rev: int, generation: int, genomes: list,
                 thumbnails: list, render_key: tuple):
        self.rev = rev
        self.generation = generation
        self.genomes = genomes
        self.thumbnails = thumbnails
        self.render_key = render_key


class AppState:
    def __init__(self):
        self.engine: str = "shapes"
        self.pop: Population | None = None
        self.shape_pop: ShapePopulation | None = None
        self.mode: str = "malevich"
        self.palette: str = "none"
        self.color_mode: str = "rgb"
        self.pop_size: int = 24
        self.thumb_size: int = 320
        self.hires_size: int = 4096
        self.batch_size: int = 2048
        self.mutation_strength: float = 1.0
        self.crossover_rate: float = 0.7
        self.grain_strength: float = 8.0
        self.gradient_strength: float = 10.0
        self._networks = []
        self._rev: int = 0
        self.lock = threading.RLock()
        self._home_lock = threading.Lock()

        self._history: list[_GenerationSnapshot] = []
        self._history_idx: int = -1
        self._max_history: int = 200
        self._home_samples_cache: dict[str, list[str]] | None = None
        self._infographic_cache: dict | None = None

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
        if self.engine == "hadid":
            return list(HADID_PALETTE.values())
        if self.engine == "maps":
            return list(MAPS_PALETTE.values())
        return list(SUPREMATIST_PALETTE.values())

    def _build_shape_config(self) -> dict:
        if self.engine == "hadid":
            pal = HADID_PALETTE
        elif self.engine == "maps":
            pal = MAPS_PALETTE
        else:
            pal = SUPREMATIST_PALETTE
        return {
            "pop_size": self.pop_size,
            "num_palette_colors": len(pal),
            "elitism": 2,
            "crossover_rate": self.crossover_rate,
            "tournament_k": 3,
            "mutation_strength": self.mutation_strength,
        }

    # ---- Initialization ----

    def initialize(self):
        if self.engine in ("shapes", "hadid", "maps"):
            self.shape_pop = ShapePopulation(self._build_shape_config())
            if self.engine == "hadid":
                self.shape_pop.initialize_architecton()
            elif self.engine == "maps":
                self.shape_pop.initialize_maps()
            else:
                self.shape_pop.initialize()
            self.pop = None
            self._networks = []
        else:
            self.pop = Population(self._build_cppn_config())
            self.pop.initialize()
            self.shape_pop = None
        if self.engine == "cppn":
            self._networks = self.pop.get_networks()
        self._history = []
        self._history_idx = -1
        self._push_history()

    # ---- History ----

    def _is_shapes(self) -> bool:
        return self.engine in ("shapes", "hadid", "maps")

    def _render_key(self) -> tuple:
        if self._is_shapes():
            return (self.engine, self.thumb_size,
                    self.grain_strength, self.gradient_strength)
        return (self.engine, self.palette, self.color_mode, self.thumb_size)

    @property
    def _cached_thumbnails(self) -> list:
        return self._history[self._history_idx].thumbnails

    def _new_rev(self) -> int:
        self._rev += 1
        return self._rev

    def invalidate_thumbnails(self):
        """Drop the current generation's thumbnails; they re-render on demand."""
        snap = self._history[self._history_idx]
        snap.thumbnails = [None] * len(snap.genomes)
        snap.render_key = self._render_key()
        snap.rev = self._new_rev()

    def _push_history(self):
        genomes = self._current_genomes()
        gen = self._current_generation()
        genome_copies = [g.copy() for g in genomes]
        snap = _GenerationSnapshot(self._new_rev(), gen, genome_copies,
                                   [None] * len(genome_copies),
                                   self._render_key())

        if self._history_idx < len(self._history) - 1:
            self._history = self._history[: self._history_idx + 1]

        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._history_idx = len(self._history) - 1

    def _restore_snapshot(self, idx: int):
        snap = self._history[idx]
        self._history_idx = idx
        if snap.render_key != self._render_key():
            # Render settings changed since this generation was drawn.
            snap.thumbnails = [None] * len(snap.genomes)
            snap.render_key = self._render_key()
            snap.rev = self._new_rev()

        if self.engine in ("shapes", "hadid", "maps") and self.shape_pop:
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

    def stream_thumbs(self, rev: int):
        """Yield ``{"i": index, "img": base64-jpeg}`` for each thumbnail of the
        current generation that isn't rendered yet, as each one completes."""
        with self.lock:
            snap = self._history[self._history_idx]
            if snap.rev != rev:
                yield {"stale": True}
                return
            thumbs = snap.thumbnails          # filled in place as results land
            missing = [i for i, t in enumerate(thumbs) if t is None]
            shapes = self._is_shapes()
            size = self.thumb_size
            if shapes:
                genomes = [snap.genomes[i] for i in missing]
                palette = self._shape_palette()
                grain, gradient = self.grain_strength, self.gradient_strength
            else:
                palette_array = get_palette_array(self.palette)
                networks = list(self._networks)
                cppn_genomes = list(self.pop.genomes)
                color_mode = self.color_mode

        if shapes:
            for k, data in _render_stream(genomes, size, palette, grain, gradient):
                i = missing[k]
                thumbs[i] = _b64(data)
                yield {"i": i, "img": thumbs[i]}
        else:
            for i in missing:
                g = cppn_genomes[i]
                img = render_to_image(
                    networks[i], size, size, color_mode, palette_array,
                    focal_x=g.comp_focal_x, focal_y=g.comp_focal_y,
                    armature_angle=g.comp_armature_angle,
                )
                thumbs[i] = _b64(_encode_jpeg(img))
                yield {"i": i, "img": thumbs[i]}
        yield {"done": True, "rev": rev}

    def _render_home_style_samples(self, style: str, count: int = 6, size: int = 280) -> list[str]:
        """Generate a small gallery for the Home page without mutating live state."""
        if style == "hadid":
            palette = list(HADID_PALETTE.values())
        elif style == "maps":
            palette = list(MAPS_PALETTE.values())
        else:
            style = "shapes"
            palette = list(SUPREMATIST_PALETTE.values())

        config = {
            "pop_size": count,
            "num_palette_colors": len(palette),
            "elitism": 2,
            "crossover_rate": self.crossover_rate,
            "tournament_k": 3,
            "mutation_strength": self.mutation_strength,
        }
        demo_pop = ShapePopulation(config)
        if style == "hadid":
            demo_pop.initialize_architecton()
        elif style == "maps":
            demo_pop.initialize_maps()
        else:
            demo_pop.initialize()

        return _render_all(demo_pop.genomes[:count], size, palette,
                           self.grain_strength, self.gradient_strength)

    def get_home_samples(self, refresh: bool = False) -> dict[str, list[str]]:
        with self._home_lock:
            if self._home_samples_cache is None or refresh:
                self._home_samples_cache = {
                    "malevich": self._render_home_style_samples("shapes"),
                    "hadid": self._render_home_style_samples("hadid"),
                    "maps": self._render_home_style_samples("maps"),
                }
            return self._home_samples_cache

    # ---- State payload ----

    def _current_genomes(self) -> list:
        if self.engine in ("shapes", "hadid", "maps"):
            return self.shape_pop.genomes
        return self.pop.genomes

    def _current_generation(self) -> int:
        if self.engine in ("shapes", "hadid", "maps"):
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
            "crossover_rate": self.crossover_rate,
            "grain_strength": self.grain_strength,
            "gradient_strength": self.gradient_strength,
            "thumbnails": list(self._cached_thumbnails),
            "rev": self._history[self._history_idx].rev,
            "has_prev": self.can_go_prev(),
            "has_next": self.can_go_next(),
        }

    # ---- Evolve ----

    def evolve(self, selected: list[int]):
        if self.engine in ("shapes", "hadid", "maps"):
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
                    g = self.pop.genomes[i]
                    img = render_to_image(
                        net, 64, 64, self.color_mode,
                        focal_x=g.comp_focal_x, focal_y=g.comp_focal_y,
                        armature_angle=g.comp_armature_angle,
                    )
                    img_np = np.array(img.convert("RGB")).astype(np.float64) / 255.0
                    style_bonus = compute_style_fitness(img_np, self.mode, weight=0.2)
                    fitness[i] += style_bonus

            self.pop.set_fitness(fitness)
            self.pop.evolve()

        if self.engine == "cppn":
            self._networks = self.pop.get_networks()
        self._push_history()

    # ---- Export ----

    def export_hires(self, index: int, orientation: str = "square") -> str:
        """Render one individual at full resolution; returns the saved filename."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        gen = self._current_generation()
        suffix = f"_{orientation}" if orientation != "square" else ""
        path = OUTPUT_DIR / f"hires_{gen:04d}_{index:02d}{suffix}.png"

        if self._is_shapes():
            fut = render_pool.submit_hires(
                self.shape_pop.genomes[index].copy(), self.hires_size,
                orientation, self._shape_palette(),
                self.grain_strength, self.gradient_strength)
            try:
                data = fut.result()
            except Exception:  # pool unavailable -> render inline
                data = render_pool.render_hires_task(
                    self.shape_pop.genomes[index].copy(), self.hires_size,
                    orientation, self._shape_palette(),
                    self.grain_strength, self.gradient_strength)
            path.write_bytes(data)
        else:
            palette_array = get_palette_array(self.palette)
            net = self._networks[index]
            g = self.pop.genomes[index]
            img = render_to_image(
                net, self.hires_size, self.hires_size,
                self.color_mode, palette_array,
                focal_x=g.comp_focal_x, focal_y=g.comp_focal_y,
                armature_angle=g.comp_armature_angle,
            )
            img.save(path, compress_level=1)
        return path.name

    def start_export_all(self) -> str:
        """Snapshot the population and render it all in a background job."""
        gen = self._current_generation()
        genomes = [g.copy() for g in self._current_genomes()]
        shapes = self.engine in ("shapes", "hadid", "maps")
        palette = self._shape_palette() if shapes else None
        networks = None if shapes else list(self._networks)
        cppn_genomes = None if shapes else list(self.pop.genomes)
        palette_array = None if shapes else get_palette_array(self.palette)
        color_mode = self.color_mode
        res = self.batch_size

        job_id = uuid.uuid4().hex[:12]
        job = {"done": 0, "total": len(genomes), "status": "running",
               "gen": gen, "path": None, "error": None}
        _export_jobs[job_id] = job

        grain, gradient = self.grain_strength, self.gradient_strength
        # Hadid's 3D post-processing needs ~2 GB per in-flight render.
        inflight = 3 if self.engine == "hadid" else None

        def render_one(i):
            g = cppn_genomes[i]
            img = render_to_image(
                networks[i], res, res, color_mode, palette_array,
                focal_x=g.comp_focal_x, focal_y=g.comp_focal_y,
                armature_angle=g.comp_armature_angle)
            b = io.BytesIO()
            img.save(b, format="PNG", compress_level=1)
            job["done"] += 1
            return i, b.getvalue()

        def run():
            try:
                OUTPUT_DIR.mkdir(exist_ok=True)
                path = OUTPUT_DIR / f"gen{gen:04d}_all_{job_id}.zip"
                with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
                    if shapes:
                        # Fan the whole population out over the process pool and
                        # write each PNG into the zip as soon as it finishes.
                        for i, data in _render_stream(
                                genomes, res, palette, grain, gradient, "png",
                                max_inflight=inflight):
                            zf.writestr(f"gen{gen:04d}_{i:02d}.png", data)
                            job["done"] += 1
                    else:
                        workers = min(4, os.cpu_count() or 2)
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            for i, data in ex.map(render_one, range(len(genomes))):
                                zf.writestr(f"gen{gen:04d}_{i:02d}.png", data)
                job["path"] = path
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001
                job["error"] = str(e)
                job["status"] = "error"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    # ---- Save ----

    def save_genome(self, index: int) -> str:
        OUTPUT_DIR.mkdir(exist_ok=True)
        gen = self._current_generation()
        path = OUTPUT_DIR / f"genome_{gen:04d}_{index:02d}.json"

        if self.engine in ("shapes", "hadid", "maps"):
            self.shape_pop.save_genome(index, path)
        else:
            self.pop.save_genome(index, path)

        return str(path)


state = AppState()
_export_jobs: dict[str, dict] = {}


def _image_to_base64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _encode_jpeg(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _render_stream(genomes, size, palette, grain, gradient, fmt="jpeg",
                   max_inflight=None):
    """Render genomes on the process pool, yielding ``(index, bytes)`` in
    completion order.

    ``max_inflight`` bounds how many renders are queued at once (large
    exports use a lot of memory per worker).  Pending work is cancelled if
    the consumer goes away -- e.g. the browser aborts the request -- and a
    broken pool falls back to rendering in-process.
    """
    limit = max_inflight or len(genomes) or 1
    todo = iter(enumerate(genomes))
    pending = {}
    try:
        while True:
            while len(pending) < limit:
                nxt = next(todo, None)
                if nxt is None:
                    break
                i, g = nxt
                try:
                    fut = render_pool.submit_render(g, size, palette, grain,
                                                    gradient, fmt)
                except Exception:   # pool unusable -> render this one inline
                    yield i, render_pool.render_task(g, size, palette, grain,
                                                     gradient, fmt)
                    continue
                pending[fut] = i
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                i = pending.pop(f)
                try:
                    data = f.result()
                except Exception:
                    data = render_pool.render_task(genomes[i], size, palette,
                                                   grain, gradient, fmt)
                yield i, data
    finally:
        for f in pending:
            f.cancel()


def _render_all(genomes, size, palette, grain, gradient) -> list[str]:
    """Render genomes in parallel; returns base64 JPEGs in input order."""
    out = [""] * len(genomes)
    for i, data in _render_stream(genomes, size, palette, grain, gradient):
        out[i] = _b64(data)
    return out


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class EvolveRequest(BaseModel):
    selected: list[int]

class IndexRequest(BaseModel):
    index: int

class ExportRequest(BaseModel):
    index: int
    orientation: str = "square"

class ResetRequest(BaseModel):
    engine: str = "shapes"
    mode: str = "malevich"
    palette: str = "none"
    color_mode: str = "rgb"
    pop_size: int = 24
    mutation_strength: float = 1.0
    crossover_rate: float = 0.7

class SettingsRequest(BaseModel):
    mutation_strength: float | None = None
    crossover_rate: float | None = None
    grain_strength: float | None = None
    gradient_strength: float | None = None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/state")
def api_state():
    if state.engine in ("shapes", "hadid", "maps") and state.shape_pop is None:
        state.initialize()
    elif state.engine == "cppn" and state.pop is None:
        state.initialize()
    return JSONResponse(state.get_state_payload())


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest):
    if state.engine in ("shapes", "hadid", "maps") and state.shape_pop is None:
        state.initialize()
    elif state.engine == "cppn" and state.pop is None:
        state.initialize()
    state.evolve(req.selected)
    return JSONResponse(state.get_state_payload())


@app.post("/api/export")
def api_export(req: ExportRequest):
    genomes = state._current_genomes()
    if not genomes:
        return JSONResponse({"error": "No population"}, status_code=400)
    if not (0 <= req.index < len(genomes)):
        return JSONResponse({"error": "Invalid index"}, status_code=400)
    orientation = req.orientation if req.orientation in ("square", "landscape", "portrait") else "square"
    name = state.export_hires(req.index, orientation=orientation)
    return JSONResponse({"url": f"/api/export_file/{name}", "filename": name,
                         "index": req.index, "orientation": orientation})


@app.get("/api/export_file/{name}")
def api_export_file(name: str):
    path = OUTPUT_DIR / Path(name).name
    if not path.is_file() or path.suffix != ".png":
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.post("/api/export_all/start")
def api_export_all_start():
    if not state._current_genomes():
        return JSONResponse({"error": "No population"}, status_code=400)
    return JSONResponse({"job": state.start_export_all()})


@app.get("/api/export_all/status/{job_id}")
def api_export_all_status(job_id: str):
    job = _export_jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "Unknown job"}, status_code=404)
    return JSONResponse({k: job[k] for k in ("done", "total", "status", "error")})


@app.get("/api/export_all/download/{job_id}")
def api_export_all_download(job_id: str):
    job = _export_jobs.get(job_id)
    if job is None or job["status"] != "done":
        return JSONResponse({"error": "Not ready"}, status_code=404)
    return FileResponse(job["path"], media_type="application/zip",
                        filename=f"gen{job['gen']:04d}_all.zip")


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
    state.crossover_rate = req.crossover_rate
    state._infographic_cache = None
    state.initialize()
    return JSONResponse(state.get_state_payload())


@app.post("/api/settings")
def api_settings(req: SettingsRequest):
    """Update any combination of tunable parameters."""
    rerender = False
    if req.mutation_strength is not None:
        state.mutation_strength = max(0.0, min(3.0, req.mutation_strength))
        if state.engine in ("shapes", "hadid", "maps") and state.shape_pop:
            state.shape_pop.mutation_strength = state.mutation_strength
    if req.crossover_rate is not None:
        state.crossover_rate = max(0.0, min(1.0, req.crossover_rate))
        if state.engine in ("shapes", "hadid", "maps") and state.shape_pop:
            state.shape_pop.crossover_rate = state.crossover_rate
    if req.grain_strength is not None:
        state.grain_strength = max(0.0, min(30.0, req.grain_strength))
        rerender = True
    if req.gradient_strength is not None:
        state.gradient_strength = max(0.0, min(30.0, req.gradient_strength))
        rerender = True
    if rerender and state.shape_pop and state._history:
        state.invalidate_thumbnails()   # re-rendered lazily via /api/thumbs/stream
    return JSONResponse(state.get_state_payload())


@app.get("/api/home_samples")
def api_home_samples():
    return JSONResponse(state.get_home_samples())


_infographic_lock = threading.Lock()


def _build_infographic(refresh: bool = False) -> dict:
    """Demo images showing the evolution process for the home-page infographic."""
    with _infographic_lock:
        if not refresh and state._infographic_cache:
            return state._infographic_cache

        palette = list(SUPREMATIST_PALETTE.values())
        size = 280
        grain, gradient = state.grain_strength, state.gradient_strength

        config = {
            "pop_size": 6,
            "num_palette_colors": len(palette),
            "elitism": 1,
            "crossover_rate": 0.7,
            "tournament_k": 2,
            "mutation_strength": 1.0,
        }

        demo = ShapePopulation(config)
        demo.initialize()
        parents = _render_all(demo.genomes, size, palette, grain, gradient)

        sel = [1, 4]
        demo.evolve_with_selection(sel)
        children = _render_all(demo.genomes, size, palette, grain, gradient)

        result = {
            "parents": parents,
            "selected": sel,
            "parent_a": parents[sel[0]],
            "parent_b": parents[sel[1]],
            "child_demo": children[0],
            "children": children,
        }
        state._infographic_cache = result
        return result


@app.get("/api/infographic")
def api_infographic(refresh: int = 0):
    return JSONResponse(_build_infographic(bool(refresh)))


@app.get("/api/thumbs/stream")
def api_thumbs_stream(rev: int):
    """NDJSON stream: one line per thumbnail as it finishes rendering."""
    def lines():
        for item in state.stream_thumbs(rev):
            yield json.dumps(item, separators=(",", ":")) + "\n"
    return StreamingResponse(
        lines(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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


@app.get("/evolve", response_class=HTMLResponse)
def evolve_page():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    if not os.environ.get("PORT"):
        import webbrowser
        print(f"Starting server at http://localhost:{port}")
        webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
