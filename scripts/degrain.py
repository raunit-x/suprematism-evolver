"""Reduce grain in an exported folder of renders.

The renderer adds per-pixel Gaussian grain (same value on R, G and B) on top
of the paper fibres and paint tooth.  This script removes most of that grain
with edge-preserving non-local-means denoising, so flat colour fields and
hard edges stay crisp, then can blend some of the original back in so the
surface does not go plastic.

Usage:
    python scripts/degrain.py ~/Downloads/gen0019_all
    python scripts/degrain.py ~/Downloads/gen0019_all --strength 12 --keep 0.2
    python scripts/degrain.py IN_DIR -o OUT_DIR --workers 4

Writes to ``<IN_DIR>_degrained`` by default; originals are never modified.
Needs OpenCV (``pip install opencv-python-headless``); falls back to
scikit-image total-variation denoising if OpenCV is missing (slower, softer).
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _denoise_cv2(rgb: np.ndarray, strength: float, color: float,
                 window: int) -> np.ndarray:
    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # h = luminance filter strength (the grain is luminance noise),
    # hColor = chroma strength (kept low so colours don't bleed).
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h=strength, hColor=color,
                                          templateWindowSize=7,
                                          searchWindowSize=window)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _denoise_skimage(rgb: np.ndarray, strength: float) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle
    weight = strength / 255.0 * 0.8
    out = denoise_tv_chambolle(rgb.astype(np.float32) / 255.0, weight=weight,
                               channel_axis=-1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def degrain(rgb: np.ndarray, strength: float, color: float, keep: float,
            window: int) -> np.ndarray:
    try:
        out = _denoise_cv2(rgb, strength, color, window)
    except ImportError:
        out = _denoise_skimage(rgb, strength)
    if keep > 0:
        mix = rgb.astype(np.float32) * keep + out.astype(np.float32) * (1 - keep)
        out = np.clip(mix + 0.5, 0, 255).astype(np.uint8)
    return out


def process(src: Path, dst: Path, strength: float, color: float, keep: float,
            window: int) -> str:
    with Image.open(src) as im:
        info = {k: v for k, v in im.info.items() if k in ("dpi", "icc_profile")}
        alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
        rgb = np.asarray(im.convert("RGB"))
    out = Image.fromarray(degrain(rgb, strength, color, keep, window))
    if alpha is not None:
        out.putalpha(alpha)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, **info)
    return dst.name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", type=Path, help="folder of exported images (or one image)")
    p.add_argument("-o", "--output", type=Path,
                   help="output folder (default: <input>_degrained)")
    p.add_argument("--strength", type=float, default=10.0,
                   help="luminance denoise strength; ~grain setting + 2 (default 10)")
    p.add_argument("--color", type=float, default=4.0,
                   help="chroma denoise strength (default 4)")
    p.add_argument("--keep", type=float, default=0.15,
                   help="fraction of original grain blended back, 0-1 (default 0.15)")
    p.add_argument("--window", type=int, default=21,
                   help="NLM search window, odd; larger = smoother, slower (default 21)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--overwrite", action="store_true",
                   help="replace files already in the output folder")
    a = p.parse_args(argv)

    src = a.input.expanduser().resolve()
    if src.is_file():
        files, base = [src], src.parent
    elif src.is_dir():
        files = sorted(f for f in src.iterdir()
                       if f.suffix.lower() in EXTS and not f.name.startswith("."))
        base = src
    else:
        print(f"not found: {src}", file=sys.stderr)
        return 1
    if not files:
        print(f"no images in {src}", file=sys.stderr)
        return 1

    out_dir = (a.output.expanduser().resolve() if a.output
               else base.with_name(base.name + "_degrained"))
    if out_dir == base:
        print("output folder must differ from input", file=sys.stderr)
        return 1

    jobs = [(f, out_dir / f.name) for f in files]
    if not a.overwrite:
        jobs = [(s, d) for s, d in jobs if not d.exists()]
    print(f"{len(jobs)} image(s) -> {out_dir}")

    args = (a.strength, a.color, min(max(a.keep, 0.0), 1.0), a.window | 1)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, s, d, *args) for s, d in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  [{i}/{len(jobs)}] {f.result()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
