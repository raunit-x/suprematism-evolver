"""Gallery of finished work: everything exported into ``output/``.

Each ``genNNNN_all_*.zip`` (an "Export all" of one generation) is one
gallery; loose ``hires_*.png`` renders form another.  Images are listed with
their pixel size (read from the PNG header only) and served as cached JPEG
previews, so multi-megabyte PNGs only travel when explicitly downloaded.
"""

from __future__ import annotations

import hashlib
import io
import re
import threading
import zipfile
from pathlib import Path

from PIL import Image

SIZES = {"thumb": 720, "large": 2400}
_ZIP_RE = re.compile(r"^gen(\d{4})_all(?:_([0-9a-f]+))?\.zip$")
_HIRES_RE = re.compile(r"^hires_(\d{4})_(\d{2})(?:_(landscape|portrait))?\.png$")
_MEMBER_RE = re.compile(r"gen(\d{4})_(\d{2})")

Image.MAX_IMAGE_PIXELS = None          # hi-res renders are legitimately huge


class Gallery:
    def __init__(self, root: Path):
        self.root = root
        self.cache_dir = root / ".gallery_cache"
        self._lock = threading.Lock()
        self._sig: tuple | None = None
        self._listing: dict | None = None
        self._items: dict[str, dict] = {}
        self._dims: dict[tuple, tuple[int, int]] = {}

    # ---- listing -------------------------------------------------------

    def _signature(self) -> tuple:
        return tuple(sorted((p.name, p.stat().st_mtime_ns)
                            for p in self.root.iterdir()
                            if p.suffix in (".zip", ".png")))

    def _size_of(self, key: tuple, opener) -> tuple[int, int]:
        """Pixel size from the PNG IHDR chunk (first 24 bytes)."""
        if key not in self._dims:
            with opener() as fh:
                head = fh.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                self._dims[key] = (int.from_bytes(head[16:20], "big"),
                                   int.from_bytes(head[20:24], "big"))
            else:
                self._dims[key] = (1, 1)
        return self._dims[key]

    def listing(self) -> dict:
        if not self.root.is_dir():
            return {"galleries": [], "items": []}
        with self._lock:
            sig = self._signature()
            if sig == self._sig and self._listing is not None:
                return self._listing
            galleries, items = [], []
            zips = sorted((p for p in self.root.glob("gen*_all*.zip") if _ZIP_RE.match(p.name)),
                          key=lambda p: (-int(_ZIP_RE.match(p.name).group(1)), -p.stat().st_mtime))
            seen_gen: dict[int, int] = {}
            for zp in zips:
                gen = int(_ZIP_RE.match(zp.name).group(1))
                try:
                    zf = zipfile.ZipFile(zp)
                except zipfile.BadZipFile:
                    continue
                names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
                if not names:
                    zf.close()
                    continue
                seen_gen[gen] = seen_gen.get(gen, 0) + 1
                gid = zp.stem
                title = f"Generation {gen}"
                if seen_gen[gen] > 1:
                    title += f" · {'ABCDEFGH'[min(seen_gen[gen] - 1, 7)]}"
                galleries.append({"id": gid, "title": title, "gen": gen, "count": len(names),
                                  "date": int(zp.stat().st_mtime)})
                for n in names:
                    m = _MEMBER_RE.search(n)
                    idx = int(m.group(2)) if m else len(items)
                    key = (zp.name, zp.stat().st_mtime_ns, n)

                    w, h = self._size_of(key, lambda zf=zf, n=n: zf.open(n))
                    items.append(self._item(gid, f"{gid}/{n}", w, h,
                                            f"Gen {gen} · No. {idx + 1:02d}",
                                            {"zip": zp, "member": n}))
                zf.close()
            hires = sorted((p for p in self.root.glob("hires_*.png") if _HIRES_RE.match(p.name)),
                           key=lambda p: -p.stat().st_mtime)
            if hires:
                galleries.insert(0, {"id": "hires", "title": "Hi-res", "gen": None,
                                     "count": len(hires),
                                     "date": int(max(p.stat().st_mtime for p in hires))})
                for p in hires:
                    m = _HIRES_RE.match(p.name)
                    key = (p.name, p.stat().st_mtime_ns, "")
                    w, h = self._size_of(key, lambda p=p: open(p, "rb"))
                    items.append(self._item("hires", p.name, w, h,
                                            f"Gen {int(m.group(1))} · No. {int(m.group(2)) + 1:02d}"
                                            + (f" · {m.group(3)}" if m.group(3) else ""),
                                            {"path": p}))
            self._items = {it["id"]: it for it in items}
            self._listing = {"galleries": galleries,
                             "items": [{k: v for k, v in it.items() if k != "_src"} for it in items]}
            self._sig = sig
            return self._listing

    @staticmethod
    def _item(gallery: str, ref: str, w: int, h: int, title: str, src: dict) -> dict:
        iid = hashlib.sha1(ref.encode()).hexdigest()[:16]
        return {"id": iid, "gallery": gallery, "w": w, "h": h, "title": title,
                "name": ref.rsplit("/", 1)[-1], "_src": src}

    # ---- images --------------------------------------------------------

    def _open_original(self, item: dict):
        src = item["_src"]
        if "path" in src:
            return open(src["path"], "rb")
        return _ZipMember(src["zip"], src["member"])

    def original(self, iid: str) -> tuple[bytes, str] | None:
        self.listing()
        item = self._items.get(iid)
        if item is None:
            return None
        with self._open_original(item) as fh:
            return fh.read(), item["name"]

    def preview(self, iid: str, size: str) -> Path | None:
        self.listing()
        item = self._items.get(iid)
        if item is None or size not in SIZES:
            return None
        out = self.cache_dir / f"{iid}_{size}.jpg"
        if out.exists():
            return out
        self.cache_dir.mkdir(exist_ok=True)
        with self._open_original(item) as fh:
            im = Image.open(fh)
            im.draft("RGB", (SIZES[size], SIZES[size]))
            im = im.convert("RGB")
            im.thumbnail((SIZES[size], SIZES[size]), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88 if size == "thumb" else 92,
                    optimize=True, progressive=True)
        tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.replace(out)
        return out


class _ZipMember:
    """A zip member opened as a seekable file (PIL needs ``seek``); each use
    opens its own ZipFile so concurrent requests never share a handle."""

    def __init__(self, zp: Path, name: str):
        with zipfile.ZipFile(zp) as zf:
            self._buf = io.BytesIO(zf.read(name))

    def __enter__(self):
        return self._buf

    def __exit__(self, *exc):
        self._buf.close()
