"""Render CPPN networks to images.

Generates coordinate grids, evaluates the CPPN in batch, and maps outputs to
RGB pixels. Supports HSV mode and optional palette quantization.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.cppn.network import CPPNNetwork


def make_coordinate_grid(width: int, height: int) -> np.ndarray:
    """Create input coordinate grid: (N, 5) with columns [x, y, d, theta, bias].

    x, y in [-1, 1]. d = distance from center. theta = angle from center.
    N = width * height.
    """
    xs = np.linspace(-1.0, 1.0, width)
    ys = np.linspace(-1.0, 1.0, height)
    x_grid, y_grid = np.meshgrid(xs, ys)

    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    d_flat = np.sqrt(x_flat ** 2 + y_flat ** 2)
    theta_flat = np.arctan2(y_flat, x_flat)
    bias_flat = np.ones_like(x_flat)

    return np.column_stack([x_flat, y_flat, d_flat, theta_flat, bias_flat])


def render_cppn(
    network: CPPNNetwork,
    width: int = 256,
    height: int = 256,
    color_mode: str = "rgb",
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Render a CPPN to an (H, W, 3) float array in [0, 1].

    Args:
        network: The CPPN to evaluate.
        width: Image width in pixels.
        height: Image height in pixels.
        color_mode: "rgb" or "hsv". Determines how outputs are interpreted.
        palette: Optional (N, 3) array of palette colors for quantization.

    Returns:
        (height, width, 3) float64 array with RGB values in [0, 1].
    """
    coords = make_coordinate_grid(width, height)
    outputs = network.forward(coords)  # (N, num_outputs)

    num_out = outputs.shape[1]

    if color_mode == "hsv":
        # Outputs: H (cyclic via sin, map to [0,1]), S, V (sigmoid-ish, already [0,1]-ish)
        h = (outputs[:, 0 % num_out] + 1.0) / 2.0  # map [-1,1] to [0,1]
        s = np.clip((outputs[:, 1 % num_out] + 1.0) / 2.0, 0, 1)
        v = np.clip((outputs[:, 2 % num_out] + 1.0) / 2.0, 0, 1)
        rgb = _hsv_to_rgb_batch(h, s, v)
    else:
        # RGB mode: map outputs to [0, 1]
        if num_out >= 3:
            r = (outputs[:, 0] + 1.0) / 2.0
            g = (outputs[:, 1] + 1.0) / 2.0
            b = (outputs[:, 2] + 1.0) / 2.0
        elif num_out == 1:
            # Grayscale
            r = g = b = (outputs[:, 0] + 1.0) / 2.0
        else:
            r = (outputs[:, 0] + 1.0) / 2.0
            g = (outputs[:, 1 % num_out] + 1.0) / 2.0
            b = np.full_like(r, 0.5)

        rgb = np.column_stack([r, g, b])

    rgb = np.clip(rgb, 0.0, 1.0)

    # Palette quantization
    if palette is not None:
        rgb = _quantize_to_palette(rgb, palette)

    return rgb.reshape(height, width, 3)


def render_to_image(
    network: CPPNNetwork,
    width: int = 256,
    height: int = 256,
    color_mode: str = "rgb",
    palette: np.ndarray | None = None,
) -> Image.Image:
    """Render a CPPN directly to a PIL Image."""
    arr = render_cppn(network, width, height, color_mode, palette)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def render_population_grid(
    networks: list[CPPNNetwork],
    thumb_size: int = 128,
    cols: int = 5,
    color_mode: str = "rgb",
    palette: np.ndarray | None = None,
    padding: int = 4,
) -> Image.Image:
    """Render a grid of CPPN thumbnails for population display."""
    n = len(networks)
    rows = (n + cols - 1) // cols

    cell = thumb_size + padding
    grid_w = cols * cell + padding
    grid_h = rows * cell + padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(40, 40, 40))

    for i, net in enumerate(networks):
        row, col = divmod(i, cols)
        img = render_to_image(net, thumb_size, thumb_size, color_mode, palette)
        x = padding + col * cell
        y = padding + row * cell
        grid.paste(img, (x, y))

    return grid


def _quantize_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Snap each pixel to the nearest palette color (Euclidean in RGB).

    Args:
        rgb: (N, 3) pixel colors.
        palette: (P, 3) palette colors.

    Returns:
        (N, 3) quantized colors.
    """
    # (N, 1, 3) - (1, P, 3) -> (N, P, 3) -> sum -> (N, P)
    diff = rgb[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)
    return palette[nearest]


def _hsv_to_rgb_batch(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV to RGB conversion. All inputs/outputs in [0, 1]."""
    h = h % 1.0
    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))))
    g = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))))
    b = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))))

    return np.column_stack([r, g, b])
