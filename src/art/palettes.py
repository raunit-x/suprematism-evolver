"""Artist-specific color palettes.

Colors stored as (R, G, B) tuples in [0, 1] float range.
"""

from __future__ import annotations

import numpy as np


def hex_to_rgb(h: str) -> tuple:
    """Convert '#RRGGBB' to (r, g, b) floats in [0, 1]."""
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# --- Malevich Suprematist Palette (original 6) ---
MALEVICH_PALETTE = {
    "white": hex_to_rgb("#FFFFFF"),
    "black": hex_to_rgb("#000000"),
    "red": hex_to_rgb("#CC0000"),
    "yellow": hex_to_rgb("#E8C800"),
    "blue": hex_to_rgb("#0044AA"),
    "green": hex_to_rgb("#228B22"),
}

# --- Expanded Suprematist Palette ---
# Drawn from the full range of colours in Malevich's Suprematist works
# (1915-1920): the core primaries plus the ochres, greys, and muted
# secondaries that appear across his paintings.
SUPREMATIST_PALETTE = {
    "white": hex_to_rgb("#FAFAFA"),
    "cream": hex_to_rgb("#F0E6D2"),
    "black": hex_to_rgb("#0A0A0A"),
    "charcoal": hex_to_rgb("#3C3C3C"),
    "grey": hex_to_rgb("#9E9E9E"),
    "light_grey": hex_to_rgb("#C8C8C8"),
    "red": hex_to_rgb("#C62828"),
    "burgundy": hex_to_rgb("#7B1A1A"),
    "rose": hex_to_rgb("#C47070"),
    "orange": hex_to_rgb("#D4762C"),
    "ochre": hex_to_rgb("#C49B2A"),
    "yellow": hex_to_rgb("#E8C800"),
    "olive": hex_to_rgb("#5E6B2E"),
    "green": hex_to_rgb("#2E7D32"),
    "blue": hex_to_rgb("#1A4FA0"),
    "navy": hex_to_rgb("#0D1F4B"),
}

# --- Basquiat Palette ---
BASQUIAT_PALETTE = {
    "raw_canvas": hex_to_rgb("#F5E6D0"),
    "black": hex_to_rgb("#000000"),
    "red": hex_to_rgb("#CC2200"),
    "blue": hex_to_rgb("#0055CC"),
    "yellow": hex_to_rgb("#DDAA00"),
    "white": hex_to_rgb("#FFFFFF"),
    "orange": hex_to_rgb("#CC6600"),
    "brown": hex_to_rgb("#8B4513"),
}

# --- Hybrid Palette (union of both) ---
HYBRID_PALETTE = {
    "white": hex_to_rgb("#FFFFFF"),
    "black": hex_to_rgb("#000000"),
    "red": hex_to_rgb("#CC1100"),
    "yellow": hex_to_rgb("#E0B400"),
    "blue": hex_to_rgb("#004DBB"),
    "green": hex_to_rgb("#228B22"),
    "orange": hex_to_rgb("#CC6600"),
    "raw_canvas": hex_to_rgb("#F5E6D0"),
    "brown": hex_to_rgb("#8B4513"),
}

# --- Hadid Palette ---
# Inspired by Zaha Hadid's suprematist-influenced paintings (1976-1992)
# "The Peak", "Malevich's Tektonik", "The World (89 Degrees)"
HADID_PALETTE = {
    "white":         hex_to_rgb("#F5F5F7"),
    "warm_white":    hex_to_rgb("#EDE8DE"),
    "light_grey":    hex_to_rgb("#C7C8CD"),
    "mid_grey":      hex_to_rgb("#8C8A92"),
    "graphite":      hex_to_rgb("#4A4950"),
    "black":         hex_to_rgb("#0B0B10"),
    "deep_blue":     hex_to_rgb("#0B2465"),
    "cobalt_blue":   hex_to_rgb("#2059BA"),
    "signal_red":    hex_to_rgb("#CF2E2E"),
    "ochre_yellow":  hex_to_rgb("#D5B143"),
    "pale_blue":     hex_to_rgb("#AFC0DC"),
    # Saturated primaries in several tones (Hadid's "Malevich's Tektonik").
    # Appended so indices of existing saved genomes are unchanged.
    "ultramarine":   hex_to_rgb("#1D35C4"),
    "vermilion":     hex_to_rgb("#E0371F"),
    "carmine":       hex_to_rgb("#A6182C"),
    "chrome_yellow": hex_to_rgb("#EDB31B"),
    "lavender_grey": hex_to_rgb("#A6ADCB"),
    "prussian":      hex_to_rgb("#10295A"),
}

# --- Maps Palette ---
# Constructivist city maps: lamp-black, vermilion, ultramarine and chrome
# yellow with warm greys, painted on aged paper.  Order is fixed: indices
# are named in src/shapes/citymap.py.
MAPS_PALETTE = {
    "paper":         hex_to_rgb("#E6DAC0"),    # 0  ground / streets
    "lamp_black":    hex_to_rgb("#1D1B1A"),    # 1
    "vermilion":     hex_to_rgb("#D8401F"),    # 2
    "ultramarine":   hex_to_rgb("#1F3B9A"),    # 3
    "chrome_yellow": hex_to_rgb("#E9B021"),    # 4
    "warm_grey":     hex_to_rgb("#8C8379"),    # 5
    "dark_grey":     hex_to_rgb("#48443F"),    # 6
    "ivory":         hex_to_rgb("#F3EDE0"),    # 7
    "carmine":       hex_to_rgb("#A3222B"),    # 8
    "cobalt":        hex_to_rgb("#3E6FB8"),    # 9
    "ochre":         hex_to_rgb("#BF8A36"),    # 10
    # Wider Constructivist range (Popova, Stepanova, Stenberg, Exter).
    "viridian":      hex_to_rgb("#2F6E52"),    # 11
    "teal":          hex_to_rgb("#1E7078"),    # 12
    "salmon":        hex_to_rgb("#E2876C"),    # 13
    "plum":          hex_to_rgb("#5A3557"),    # 14
    "sienna":        hex_to_rgb("#94452A"),    # 15
    "navy":          hex_to_rgb("#1B2541"),    # 16
    "sky":           hex_to_rgb("#93B2CF"),    # 17
    "kraft":         hex_to_rgb("#C7A774"),    # 18
    "pale_grey":     hex_to_rgb("#CEC9BE"),    # 19
    "olive":         hex_to_rgb("#6F6A35"),    # 20
    "rose":          hex_to_rgb("#E59A9C"),    # 21  Rodchenko pink
    "orange":        hex_to_rgb("#E8722A"),    # 22
}

PALETTES = {
    "malevich": MALEVICH_PALETTE,
    "suprematist": SUPREMATIST_PALETTE,
    "basquiat": BASQUIAT_PALETTE,
    "hybrid": HYBRID_PALETTE,
    "hadid": HADID_PALETTE,
    "maps": MAPS_PALETTE,
    "none": None,
}


def get_palette_array(name: str) -> np.ndarray | None:
    """Return palette as (N, 3) numpy array, or None for no quantization."""
    palette = PALETTES.get(name)
    if palette is None:
        return None
    return np.array(list(palette.values()), dtype=np.float64)
