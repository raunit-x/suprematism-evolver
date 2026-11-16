"""A late-19th / early-20th-century oil palette.

Mixing reuses the Kubelka-Munk helpers from the Renaissance engine, so
yellow + blue strokes go green and a colour darkened with black loses
chroma the way paint does.
"""

from __future__ import annotations

import numpy as np

from src.renaissance.pigments import (  # noqa: F401  (re-exported)
    from_ks, ks, linear_to_srgb, mix, srgb_to_linear,
)


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


PIGMENTS_SRGB: dict[str, np.ndarray] = {
    "zinc_white":     _hex("#F1EDE2"),
    "lemon_yellow":   _hex("#EFD93E"),
    "chrome_yellow":  _hex("#E9AE22"),
    "chrome_orange":  _hex("#E07322"),
    "vermilion":      _hex("#D43F27"),
    "cadmium_red":    _hex("#B8241E"),
    "crimson_lake":   _hex("#851B2E"),
    "rose":           _hex("#E2909C"),
    "cobalt_violet":  _hex("#82579A"),
    "ultramarine":    _hex("#23338C"),
    "cobalt_blue":    _hex("#2E56A6"),
    "prussian_blue":  _hex("#18293F"),
    "cerulean":       _hex("#4D90BF"),
    "viridian":       _hex("#1E6A56"),
    "emerald_green":  _hex("#3E9F62"),
    "olive_green":    _hex("#6A7630"),
    "yellow_ochre":   _hex("#C6963F"),
    "raw_sienna":     _hex("#B4733A"),
    "burnt_sienna":   _hex("#8A4628"),
    "ivory_black":    _hex("#1D1B1C"),
}
PIGMENTS: dict[str, np.ndarray] = {k: srgb_to_linear(v) for k, v in PIGMENTS_SRGB.items()}
NAMES = list(PIGMENTS)

# What each kind of primitive is plausibly painted in.
CHOICES: dict[str, list[str]] = {
    "sky":     ["ultramarine", "cobalt_blue", "prussian_blue", "cerulean",
                "cobalt_violet", "chrome_orange", "vermilion", "lemon_yellow", "rose"],
    "sky_low": ["zinc_white", "lemon_yellow", "chrome_yellow", "chrome_orange",
                "cerulean", "rose", "vermilion"],
    "hill":    ["viridian", "emerald_green", "olive_green", "cobalt_blue",
                "cobalt_violet", "ultramarine", "burnt_sienna", "rose", "prussian_blue"],
    "field":   ["chrome_yellow", "yellow_ochre", "lemon_yellow", "olive_green",
                "emerald_green", "vermilion", "raw_sienna", "chrome_orange"],
    "water":   ["prussian_blue", "ultramarine", "cobalt_blue", "cerulean", "viridian",
                "cobalt_violet"],
    "orb":     ["lemon_yellow", "chrome_yellow", "chrome_orange", "zinc_white", "vermilion"],
    "cypress": ["viridian", "prussian_blue", "olive_green", "ultramarine"],
    "tree":    ["emerald_green", "olive_green", "viridian", "rose", "chrome_orange",
                "cobalt_violet"],
    "road":    ["yellow_ochre", "rose", "raw_sienna", "zinc_white", "vermilion",
                "chrome_orange"],
    "ground":  ["raw_sienna", "yellow_ochre", "zinc_white", "rose", "vermilion",
                "burnt_sienna"],
}

# Broken-colour partners: the colour a painter flecks into a passage.
ACCENTS: dict[str, list[str]] = {
    "zinc_white":    ["lemon_yellow", "cerulean", "rose"],
    "lemon_yellow":  ["zinc_white", "chrome_orange", "emerald_green"],
    "chrome_yellow": ["chrome_orange", "lemon_yellow", "cobalt_violet"],
    "chrome_orange": ["chrome_yellow", "cobalt_blue", "vermilion"],
    "vermilion":     ["chrome_orange", "emerald_green", "crimson_lake"],
    "cadmium_red":   ["vermilion", "viridian", "crimson_lake"],
    "crimson_lake":  ["cadmium_red", "cobalt_violet", "viridian"],
    "rose":          ["zinc_white", "cobalt_violet", "emerald_green"],
    "cobalt_violet": ["rose", "ultramarine", "chrome_yellow"],
    "ultramarine":   ["cobalt_blue", "cerulean", "chrome_orange"],
    "cobalt_blue":   ["cerulean", "zinc_white", "chrome_orange"],
    "prussian_blue": ["ultramarine", "viridian", "cobalt_blue"],
    "cerulean":      ["zinc_white", "cobalt_blue", "lemon_yellow"],
    "viridian":      ["emerald_green", "prussian_blue", "vermilion"],
    "emerald_green": ["lemon_yellow", "viridian", "rose"],
    "olive_green":   ["yellow_ochre", "emerald_green", "cobalt_violet"],
    "yellow_ochre":  ["chrome_yellow", "raw_sienna", "cobalt_blue"],
    "raw_sienna":    ["yellow_ochre", "burnt_sienna", "cerulean"],
    "burnt_sienna":  ["raw_sienna", "chrome_orange", "cobalt_blue"],
    "ivory_black":   ["prussian_blue", "viridian", "crimson_lake"],
}


def pigment(name: str) -> np.ndarray:
    return PIGMENTS[name]


def darker(name: str, t: float = 0.55) -> np.ndarray:
    """A deeper version of a pigment's own hue (mixed toward Prussian blue
    and black, as a painter would, rather than multiplied toward grey)."""
    return mix(mix(PIGMENTS[name], PIGMENTS["prussian_blue"], t * 0.7),
               PIGMENTS["ivory_black"], t * 0.35)
