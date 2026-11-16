"""Suprematism Evolution -- Web UI Server.

Engines:
  - "cppn"        : CPPN-NEAT (original Picbreeder-style)
  - "shapes"      : Shape-genome evolution (explicit geometric primitives)
  - "hadid"/"maps": 3D variants of the shapes engine
  - "renaissance" : sacred-scene genomes + CPPN fields, glaze renderer
  - "flow"        : expressionist landscapes painted along evolvable flow fields

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
from collections import OrderedDict
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from src.gallery import Gallery
from src.neat.population import Population
from src.cppn.renderer import render_to_image
from src.art.palettes import get_palette_array, MALEVICH_PALETTE, SUPREMATIST_PALETTE, HADID_PALETTE, MAPS_PALETTE
from src.art.fitness import compute_style_fitness
from src.cppn.activations import MALEVICH_ACTIVATIONS, ALL_ACTIVATIONS

from src.shapes.population import ShapePopulation
from src.shapes.renderer import render_genome, render_genome_extended
from src.renaissance.population import RenaissancePopulation
from src.flow.population import FlowPopulation
from src import render_pool

OUTPUT_DIR = Path("output")
# Engines whose population lives in ``AppState.shape_pop`` and renders on
# the process pool.
SHAPE_ENGINES = ("shapes", "hadid", "maps", "renaissance", "flow")
STATIC_DIR = Path(__file__).parent / "static"


def _prefetch_home():
    """Warm the worker pool and pre-render the landing-page galleries so the
    first visitor doesn't wait for them."""
    try:
        render_pool.warm_up()
        state.get_home_samples()
        api_demo()
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
        self.refine: float | None = None      # None = auto (cools with generation)
        self.detail: float | None = None      # None = auto (follows refine)
        self._view_cache: OrderedDict = OrderedDict()
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
            "refine": self.refine,
            "detail": self.detail,
        }

    # ---- Initialization ----

    def initialize(self):
        if self.engine in ("renaissance", "flow"):
            cls = FlowPopulation if self.engine == "flow" else RenaissancePopulation
            pop = cls(self._build_shape_config())
            pop.initialize()
            self.shape_pop = pop
            self.pop = None
            self._networks = []
        elif self.engine in SHAPE_ENGINES:
            # Build fully, then swap in: a request arriving mid-reset must
            # never see an empty population.
            pop = ShapePopulation(self._build_shape_config())
            if self.engine == "hadid":
                pop.initialize_architecton()
            elif self.engine == "maps":
                pop.initialize_maps()
            else:
                pop.initialize()
            self.shape_pop = pop
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

    def adopt_shapes(self, genomes: list, engine: str = "shapes") -> dict:
        """Start a shapes-family session from externally bred genomes (the
        home demo).  Missing slots are filled with offspring of the adopted
        genomes."""
        import random
        from src.shapes import citymap
        with self.lock:
            self.engine = engine
            self.pop = None
            self._networks = []
            self.shape_pop = ShapePopulation(self._build_shape_config())
            pool = list(genomes)
            while len(pool) < self.pop_size:
                if genomes[0].city is not None:
                    child = random.choice(genomes).copy()
                    citymap.mutate_city(child.city, self.mutation_strength, 1.0)
                else:
                    child = self.shape_pop._breed_child(
                        random.sample(genomes, min(2, len(genomes))))
                pool.append(child)
            self.shape_pop.genomes = pool[: self.pop_size]
            self._history = []
            self._history_idx = -1
            self._push_history()
            return self.get_state_payload()

    # ---- History ----

    def _is_shapes(self) -> bool:
        return self.engine in SHAPE_ENGINES

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

    def discard_renders(self):
        """Forget every cached render so the next request draws it afresh:
        the current generation (re-streamed via /api/thumbs/stream), older
        history snapshots (re-rendered when navigated to), full-size views
        and the home-page galleries."""
        with self.lock:
            for snap in self._history:
                snap.render_key = None
            if self._history:
                self.invalidate_thumbnails()
            self._view_cache.clear()
        with self._home_lock:
            self._home_samples_cache = None
        self._infographic_cache = None

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

        if self.engine in SHAPE_ENGINES and self.shape_pop:
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
        if self.engine in SHAPE_ENGINES:
            return self.shape_pop.genomes
        return self.pop.genomes

    def _current_generation(self) -> int:
        if self.engine in SHAPE_ENGINES:
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
            "refine": -1.0 if self.refine is None else self.refine,
            "detail": -1.0 if self.detail is None else self.detail,
            "temperature": self.shape_pop.temperature() if self.shape_pop else 1.0,
            "grain_strength": self.grain_strength,
            "gradient_strength": self.gradient_strength,
            "thumbnails": list(self._cached_thumbnails),
            "rev": self._history[self._history_idx].rev,
            "has_prev": self.can_go_prev(),
            "has_next": self.can_go_next(),
        }

    # ---- Evolve ----

    def evolve(self, selected: list[int]):
        if self.engine in SHAPE_ENGINES:
            self.shape_pop.mutation_strength = self.mutation_strength
            self.shape_pop.refine = self.refine
            self.shape_pop.detail = self.detail
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

    def view_image(self, index: int, size: int) -> bytes:
        """Render one individual of the current generation for on-screen
        viewing (PNG bytes, nothing written to disk).  A few recent renders
        are cached so stepping back and forth in the viewer is instant."""
        rev = self._history[self._history_idx].rev
        key = (rev, index, size, self.grain_strength, self.gradient_strength)
        cached = self._view_cache.get(key)
        if cached is not None:
            self._view_cache.move_to_end(key)
            return cached
        if self._is_shapes():
            genome = self.shape_pop.genomes[index].copy()
            args = (genome, size, "square", self._shape_palette(),
                    self.grain_strength, self.gradient_strength)
            try:
                data = render_pool.submit_hires(*args).result()
            except Exception:  # pool unavailable -> render inline
                data = render_pool.render_hires_task(*args)
        else:
            palette_array = get_palette_array(self.palette)
            g = self.pop.genomes[index]
            img = render_to_image(
                self._networks[index], size, size, self.color_mode, palette_array,
                focal_x=g.comp_focal_x, focal_y=g.comp_focal_y,
                armature_angle=g.comp_armature_angle,
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG", compress_level=1)
            data = buf.getvalue()
        self._view_cache[key] = data
        while len(self._view_cache) > 16:
            self._view_cache.popitem(last=False)
        return data

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
        shapes = self.engine in SHAPE_ENGINES
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

        if self.engine in SHAPE_ENGINES:
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
    refine: float | None = None          # 0 explore .. 1 refine; negative = auto
    detail: float | None = None          # 0 none .. 1 lots of new structure; negative = auto
    grain_strength: float | None = None
    gradient_strength: float | None = None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/state")
def api_state():
    if state.engine in SHAPE_ENGINES and state.shape_pop is None:
        state.initialize()
    elif state.engine == "cppn" and state.pop is None:
        state.initialize()
    return JSONResponse(state.get_state_payload())


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest):
    if state.engine in SHAPE_ENGINES and state.shape_pop is None:
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


@app.get("/api/view/{index}")
def api_view(index: int, size: int = 2048):
    """Full-resolution PNG of one individual for the in-page viewer."""
    genomes = state._current_genomes()
    if not genomes or not (0 <= index < len(genomes)):
        return JSONResponse({"error": "Invalid index"}, status_code=400)
    size = max(512, min(3072, int(size)))
    return Response(state.view_image(index, size), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


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
        if state.engine in SHAPE_ENGINES and state.shape_pop:
            state.shape_pop.mutation_strength = state.mutation_strength
    if req.crossover_rate is not None:
        state.crossover_rate = max(0.0, min(1.0, req.crossover_rate))
        if state.engine in SHAPE_ENGINES and state.shape_pop:
            state.shape_pop.crossover_rate = state.crossover_rate
    if req.refine is not None:
        state.refine = None if req.refine < 0 else max(0.0, min(1.0, req.refine))
        if state.shape_pop:
            state.shape_pop.refine = state.refine
    if req.detail is not None:
        state.detail = None if req.detail < 0 else max(0.0, min(1.0, req.detail))
        if state.shape_pop:
            state.shape_pop.detail = state.detail
    if req.grain_strength is not None:
        state.grain_strength = max(0.0, min(30.0, req.grain_strength))
        rerender = True
    if req.gradient_strength is not None:
        state.gradient_strength = max(0.0, min(30.0, req.gradient_strength))
        rerender = True
    if rerender and state.shape_pop and state._history:
        state.invalidate_thumbnails()   # re-rendered lazily via /api/thumbs/stream
    return JSONResponse(state.get_state_payload())


@app.post("/api/rerender")
def api_rerender():
    """Called on every page load: drop all cached renders so a reload always
    shows freshly drawn images of the same genomes."""
    state.discard_renders()
    with _demo_lock:
        _demo["images"] = None
    return JSONResponse({"ok": True})


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


# ---------------------------------------------------------------------------
# Demo lab -- a tiny, separate population the home page breeds interactively
# ---------------------------------------------------------------------------

DEMO_SIZE = 8
DEMO_PX = 240
DEMO_ENGINE = "maps"
DEMO_PALETTE = list(MAPS_PALETTE.values())
_demo_lock = threading.Lock()
_demo: dict = {"pop": None, "lineage": []}


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb)


def _city_summary(c, palette) -> dict:
    """A city map's structural genes in the same shape as ``_genome_summary``:
    streets, avenues, rails and waterways become lines/curves (``pts``),
    plazas, the radial hub and building complexes become solid forms."""
    import math

    def col(i):
        return _hex(palette[i % len(palette)])

    def curve(kind, cv, color, width):
        pts = cv.get("pts") or []
        if len(pts) < 2:
            return None
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        return {"k": kind, "x": round(mx, 4), "y": round(my, 4),
                "w": round(math.hypot(x1 - x0, y1 - y0), 4), "h": round(width, 4),
                "r": round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 1), "c": color,
                "pts": [[round(x, 4), round(y, 4)] for x, y in pts]}

    ink = col(1 if c.ground != 1 else 7)
    water = col(c.water_color if c.water_color >= 0 else 3)
    lead = col(c.lead)
    geom = []
    for w in c.waters:
        if w.get("kind") == "lake":
            d = 2 * w.get("r", 0.1)
            geom.append({"k": "lake", "x": round(w["x"], 4), "y": round(w["y"], 4),
                         "w": round(d, 4), "h": round(d, 4), "r": 0.0, "c": water})
        else:
            g = curve(w.get("kind", "river"), w, water, max(0.012, w.get("width", 0.0)))
            if g:
                geom.append(g)
    if c.hub:
        d = 2 * c.hub.get("ring", 0.1)
        geom.append({"k": "hub", "x": round(c.hub["x"], 4), "y": round(c.hub["y"], 4),
                     "w": round(d, 4), "h": round(d, 4), "r": round(c.hub.get("rot", 0.0), 1),
                     "c": ink})
    for cx in c.complexes:
        geom.append({"k": cx.get("kind", "complex"), "x": round(cx["x"], 4),
                     "y": round(cx["y"], 4), "w": round(cx["w"], 4), "h": round(cx["h"], 4),
                     "r": round(cx.get("angle", 0.0), 1), "c": lead})
    for pl in c.plazas:
        d = 2 * pl.get("r", 0.05)
        geom.append({"k": "plaza" if pl.get("shape") == "circle" else "square",
                     "x": round(pl["x"], 4), "y": round(pl["y"], 4), "w": round(d, 4),
                     "h": round(d, 4), "r": 0.0,
                     "c": lead if pl.get("fill") == "lead" else ink})
    for av in c.avenues:
        g = curve("avenue", av, ink, max(0.008, av.get("width", 0.01)))
        if g:
            geom.append(g)
    for rl in c.rails:
        g = curve("rail", rl, ink, 0.006)
        if g:
            geom.append(g)
    for cut in c.cuts:
        geom.append({"k": "street", "x": round(cut["x"], 4), "y": round(cut["y"], 4),
                     "w": round(min(cut["length"], 1.5), 4), "h": round(cut["width"], 4),
                     "r": round(cut["angle"], 1), "c": ink})
    weights = sorted(range(len(c.weights)), key=lambda i: -c.weights[i])
    used = [lead] + [col(i) for i in weights if c.weights[i] > 0 and i != c.ground]
    colors = list(dict.fromkeys(used))[:6]
    kinds: dict[str, int] = {}
    for g in geom:
        kinds[g["k"]] = kinds.get(g["k"], 0) + 1
    return {"groups": 0, "satellites": 0, "shapes": len(geom), "kinds": kinds,
            "colors": colors, "armature": round(c.base_angle, 1), "plan": c.plan,
            "geom": geom[:40]}


def _genome_summary(g, palette) -> dict:
    if getattr(g, "city", None) is not None:
        return _city_summary(g.city, palette)
    shapes = g.flatten()
    kinds: dict[str, int] = {}
    for s in shapes:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    used = []
    for s in shapes:
        c = _hex(palette[s.color_idx % len(palette)])
        if c not in used:
            used.append(c)
    return {
        "groups": len(g.groups),
        "satellites": len(g.satellites),
        "shapes": len(shapes),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "colors": used[:6],
        "armature": round(g.armature_angle, 1),
        # Geometry for the home page's gene overlay (normalised canvas units,
        # centre-anchored, rotation in degrees clockwise).
        "geom": [{"k": sh.kind, "x": round(sh.x, 4), "y": round(sh.y, 4),
                  "w": round(sh.w, 4), "h": round(sh.h, 4),
                  "r": round(sh.rotation, 1),
                  "c": _hex(palette[sh.color_idx % len(palette)])}
                 for sh in shapes[:40]],
    }


def _demo_payload(images: list[str]) -> dict:
    pop = _demo["pop"]
    palette = DEMO_PALETTE
    return {
        "generation": pop.generation,
        "images": images,
        "genomes": [_genome_summary(g, palette) for g in pop.genomes],
        "lineage": _demo["lineage"],
        "mutation": pop.mutation_strength,
        "engine": DEMO_ENGINE,
    }


class DemoBreedRequest(BaseModel):
    selected: list[int]
    mutation: float = 1.0


@app.get("/api/demo")
def api_demo():
    with _demo_lock:
        pop = _demo["pop"]
        if pop is not None:
            if not _demo.get("images"):
                # Renders were discarded (page reload): redraw the same genomes.
                _demo["images"] = _render_all(
                    pop.genomes, DEMO_PX, DEMO_PALETTE,
                    state.grain_strength, state.gradient_strength)
            return JSONResponse(_demo_payload(_demo["images"]))
    return api_demo_seed()


@app.post("/api/demo/seed")
def api_demo_seed():
    palette = DEMO_PALETTE
    with _demo_lock:
        pop = ShapePopulation({
            "pop_size": DEMO_SIZE, "num_palette_colors": len(palette),
            "elitism": 2, "crossover_rate": 0.7, "tournament_k": 2,
            "mutation_strength": 1.0,
        })
        pop.initialize_maps()
        _demo["pop"] = pop
        _demo["lineage"] = []
        imgs = _render_all(pop.genomes, DEMO_PX, palette,
                           state.grain_strength, state.gradient_strength)
        _demo["images"] = imgs
        return JSONResponse(_demo_payload(imgs))


@app.post("/api/demo/breed")
def api_demo_breed(req: DemoBreedRequest):
    palette = DEMO_PALETTE
    with _demo_lock:
        pop = _demo["pop"]
        if pop is None:
            return JSONResponse({"error": "seed first"}, status_code=409)
        sel = [i for i in req.selected if 0 <= i < len(pop.genomes)][:4]
        if not sel:
            return JSONResponse({"error": "select at least one"}, status_code=400)
        prev = _demo.get("images") or []
        _demo["lineage"].append({
            "generation": pop.generation,
            "parents": [prev[i] for i in sel if i < len(prev)],
        })
        _demo["lineage"] = _demo["lineage"][-6:]
        pop.mutation_strength = max(0.0, min(3.0, req.mutation))
        pop.evolve_with_selection(sel)
        imgs = _render_all(pop.genomes, DEMO_PX, palette,
                           state.grain_strength, state.gradient_strength)
        _demo["images"] = imgs
        return JSONResponse(_demo_payload(imgs))


@app.post("/api/demo/adopt")
def api_demo_adopt():
    """Hand the demo population over to the main Evolve workspace."""
    with _demo_lock:
        pop = _demo["pop"]
        if pop is None:
            return JSONResponse({"error": "seed first"}, status_code=409)
        genomes = [g.copy() for g in pop.genomes]
    data = state.adopt_shapes(genomes, DEMO_ENGINE)
    return JSONResponse(data)


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


@app.get("/gallery", response_class=HTMLResponse)
def gallery_page():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


# ---------------------------------------------------------------------------
# Gallery: finished work in output/ (export-all zips + hi-res renders)
# ---------------------------------------------------------------------------

_gallery = Gallery(OUTPUT_DIR)


@app.get("/api/gallery")
def api_gallery():
    return JSONResponse(_gallery.listing())


@app.get("/api/gallery/img/{iid}")
def api_gallery_img(iid: str, size: str = "thumb"):
    if size == "original":
        got = _gallery.original(iid)
        if got is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        data, name = got
        return Response(data, media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})
    path = _gallery.preview(iid, size)
    if path is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


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
