"""Process pool for parallel genome rendering.

Rendering is CPU-bound PIL/numpy work, so threads don't help much.  A small
pool of worker processes lets a whole population render in roughly the time
of one image, and lets the server hand results to the browser as each one
finishes.

Workers receive everything they need as arguments (genome, palette, grain /
gradient) -- they never read the server's mutable module-level settings.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
from concurrent.futures import Future, ProcessPoolExecutor

_pool: ProcessPoolExecutor | None = None


def worker_count() -> int:
    env = os.environ.get("RENDER_WORKERS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(os.cpu_count() or 1, 8))


def _worker_init() -> None:
    # Pay import + fiber-tile generation once per worker, not per request.
    from src.shapes import renderer
    renderer._get_fiber_tile()
    renderer._tile_fiber(1024, 1024)


def get_pool() -> ProcessPoolExecutor:
    global _pool
    if _pool is None:
        # "spawn" is safe with a threaded server and behaves the same on
        # macOS and Linux.
        _pool = ProcessPoolExecutor(
            max_workers=worker_count(),
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
        )
    return _pool


def warm_up() -> None:
    """Start every worker now so the first real request doesn't pay for it."""
    pool = get_pool()
    for f in [pool.submit(_noop) for _ in range(worker_count())]:
        f.result()


def shutdown() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _noop() -> int:
    return 0


# ---------------------------------------------------------------------------
# Worker tasks (module-level so they can be pickled)
# ---------------------------------------------------------------------------

def _encode(img, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _is_renaissance(genome) -> bool:
    return getattr(genome, "engine_tag", None) == "renaissance"


def _is_flow(genome) -> bool:
    return getattr(genome, "engine_tag", None) == "flow"


def render_task(genome, size: int, palette, grain: float, gradient: float,
                fmt: str = "jpeg") -> bytes:
    if _is_flow(genome):
        # The grain slider scales the impasto relief.
        from src.flow.renderer import render_flow
        return _encode(render_flow(genome, size, size, impasto_scale=grain / 8.0), fmt)
    if _is_renaissance(genome):
        # Palette/gradient don't apply; the grain slider scales brushwork.
        from src.renaissance.renderer import render_scene
        img = render_scene(genome, size, size, brush_scale=grain / 8.0)
        return _encode(img, fmt)
    from src.shapes.renderer import render_genome
    img = render_genome(genome, size, size, palette, grain, gradient)
    return _encode(img, fmt)


def render_hires_task(genome, size: int, orientation: str, palette,
                      grain: float, gradient: float) -> bytes:
    if _is_flow(genome):
        from src.flow.renderer import render_flow_extended
        img = render_flow_extended(genome, size, orientation, impasto_scale=grain / 8.0)
        return _encode(img, "png")
    if _is_renaissance(genome):
        from src.renaissance.renderer import render_scene_extended
        img = render_scene_extended(genome, size, orientation,
                                    brush_scale=grain / 8.0)
        return _encode(img, "png")
    from src.shapes.renderer import render_genome_extended
    img = render_genome_extended(genome, size, orientation, palette,
                                 grain, gradient)
    return _encode(img, "png")


def submit_render(genome, size: int, palette, grain: float, gradient: float,
                  fmt: str = "jpeg") -> Future:
    return get_pool().submit(render_task, genome, size, palette, grain,
                             gradient, fmt)


def submit_hires(genome, size: int, orientation: str, palette, grain: float,
                 gradient: float) -> Future:
    return get_pool().submit(render_hires_task, genome, size, orientation,
                             palette, grain, gradient)
