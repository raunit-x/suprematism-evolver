"""Palette quantization utilities.

Provides LAB-space quantization for perceptually accurate color mapping,
as well as simpler RGB-space quantization.
"""

from __future__ import annotations

import numpy as np

from src.art.palettes import get_palette_array


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB [0,1]."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) sRGB [0,1] to CIE LAB.

    Uses D65 illuminant.
    """
    # sRGB -> linear RGB -> XYZ
    linear = _srgb_to_linear(rgb)

    # Linear RGB to XYZ (D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ m.T

    # Normalize by D65 white point
    xyz[:, 0] /= 0.95047
    xyz[:, 1] /= 1.00000
    xyz[:, 2] /= 1.08883

    # XYZ -> LAB
    epsilon = 0.008856
    kappa = 903.3
    f = np.where(xyz > epsilon, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    return np.column_stack([L, a, b])


def quantize_lab(image: np.ndarray, palette_name: str) -> np.ndarray:
    """Quantize an (H, W, 3) RGB image to a named palette using LAB distance.

    Args:
        image: (H, W, 3) float array in [0, 1].
        palette_name: One of "malevich", "basquiat", "hybrid".

    Returns:
        (H, W, 3) quantized image.
    """
    palette_rgb = get_palette_array(palette_name)
    if palette_rgb is None:
        return image

    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)

    # Convert both to LAB
    pixels_lab = rgb_to_lab(pixels)
    palette_lab = rgb_to_lab(palette_rgb)

    # Find nearest palette color in LAB space
    diff = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)

    quantized = palette_rgb[nearest]
    return quantized.reshape(h, w, 3)
