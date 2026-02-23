"""Artist-specific color palettes.

Colors stored as (R, G, B) tuples in [0, 1] float range.
"""

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
    "white":       hex_to_rgb("#F0F0F0"),
    "light_grey":  hex_to_rgb("#C8C8C8"),
    "mid_grey":    hex_to_rgb("#888888"),
    "dark_grey":   hex_to_rgb("#404040"),
    "black":       hex_to_rgb("#0A0A0A"),
    "deep_blue":   hex_to_rgb("#0B2366"),
    "royal_blue":  hex_to_rgb("#1A4FA0"),
    "red":         hex_to_rgb("#C62828"),
    "amber":       hex_to_rgb("#C49B2A"),
    "cream":       hex_to_rgb("#E8DCC8"),
}

PALETTES = {
    "malevich": MALEVICH_PALETTE,
    "suprematist": SUPREMATIST_PALETTE,
    "basquiat": BASQUIAT_PALETTE,
    "hybrid": HYBRID_PALETTE,
    "hadid": HADID_PALETTE,
    "none": None,
}


def get_palette_array(name: str) -> np.ndarray | None:
    """Return palette as (N, 3) numpy array, or None for no quantization."""
    palette = PALETTES.get(name)
    if palette is None:
        return None
    return np.array(list(palette.values()), dtype=np.float64)
