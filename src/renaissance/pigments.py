"""Historical pigments and Kubelka-Munk mixing.

Paint does not mix like light.  Blue + yellow pigment gives green, a
pigment darkened with black loses chroma unevenly, and a thin transparent
glaze over a light underlayer goes deep and saturated.  Single-constant
Kubelka-Munk theory reproduces this: each pigment is described by its
absorption/scattering ratio K/S per channel, mixtures average K/S, and the
reflectance of the mixture follows from it.

Mixing happens on linear-light reflectance; colours are stored as sRGB.
"""

from __future__ import annotations

import numpy as np


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


# Approximate masstone reflectances (sRGB) of a period panel painter's box.
PIGMENTS_SRGB: dict[str, np.ndarray] = {
    "lead_white":      _hex("#EEE7D7"),
    "bone_black":      _hex("#1D1A17"),
    "raw_umber":       _hex("#5B4733"),
    "burnt_umber":     _hex("#4A2E1F"),
    "burnt_sienna":    _hex("#8C4B2B"),
    "yellow_ochre":    _hex("#C2923F"),
    "red_ochre":       _hex("#8F3C27"),
    "vermilion":       _hex("#C93A1F"),
    "madder_lake":     _hex("#8C1F30"),
    "ultramarine":     _hex("#1E3A8E"),
    "azurite":         _hex("#3C6B9F"),
    "green_earth":     _hex("#6F7E5A"),
    "verdigris":       _hex("#2F6C53"),
    "malachite":       _hex("#4F8B5F"),
    "lead_tin_yellow": _hex("#E3C35B"),
    "gold_leaf":       _hex("#C99E42"),
}

PIGMENTS: dict[str, np.ndarray] = {k: srgb_to_linear(v) for k, v in PIGMENTS_SRGB.items()}

# Pigments a figure's robe or mantle may be painted in.  Blue mantles and red
# robes dominate sacred painting; the rest appear on saints and attendants.
DRAPERY_PIGMENTS = [
    "vermilion", "madder_lake", "ultramarine", "azurite", "verdigris",
    "malachite", "yellow_ochre", "red_ochre", "lead_white", "green_earth",
    "burnt_sienna", "lead_tin_yellow",
]
HAIR_PIGMENTS = ["burnt_umber", "raw_umber", "bone_black", "burnt_sienna", "yellow_ochre"]
GROUND_PIGMENTS = ["yellow_ochre", "red_ochre", "raw_umber", "green_earth"]


def pigment(name: str) -> np.ndarray:
    return PIGMENTS[name]


_EPS = 1e-3


def ks(r: np.ndarray) -> np.ndarray:
    """Reflectance -> K/S."""
    r = np.clip(r, _EPS, 1.0 - _EPS)
    return (1.0 - r) ** 2 / (2.0 * r)


def from_ks(k: np.ndarray) -> np.ndarray:
    """K/S -> reflectance."""
    return 1.0 + k - np.sqrt(k * k + 2.0 * k)


def mix(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Kubelka-Munk mixture of reflectances ``a`` and ``b``; ``t`` is the
    share of ``b`` (scalar or array broadcastable against ``a``)."""
    t = np.asarray(t, dtype=np.float32)
    if t.ndim and t.shape[-1] not in (1, 3):
        t = t[..., None]
    return from_ks((1.0 - t) * ks(a) + t * ks(b))


def mix_many(names_weights: list[tuple[str, float]]) -> np.ndarray:
    total = sum(w for _, w in names_weights)
    k = sum(ks(PIGMENTS[n]) * (w / total) for n, w in names_weights)
    return from_ks(k)


def glaze(under: np.ndarray, color: np.ndarray, thickness) -> np.ndarray:
    """A transparent glaze layer over ``under``: light passes through the
    glaze, reflects off the layer below and passes through again, so the
    glaze's own colour acts as a transmittance applied twice (Beer-Lambert).
    """
    thickness = np.asarray(thickness, dtype=np.float32)
    if thickness.ndim and thickness.shape[-1] not in (1, 3):
        thickness = thickness[..., None]
    trans = np.power(np.clip(color, _EPS, 1.0), 0.5 * thickness)
    return under * trans * trans


def flesh(tone: float) -> np.ndarray:
    """Carnation mixed from lead white, vermilion and ochre; ``tone`` 0 is
    pale, 1 is warm olive."""
    return mix_many([
        ("lead_white", 1.0 - 0.45 * tone),
        ("vermilion", 0.05 + 0.03 * tone),
        ("yellow_ochre", 0.10 + 0.25 * tone),
        ("raw_umber", 0.02 + 0.10 * tone),
    ])
