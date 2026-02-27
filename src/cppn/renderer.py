"""Render CPPN networks to images.

Generates coordinate grids, evaluates the CPPN in batch, and maps outputs to
RGB pixels. Supports HSV mode and optional palette quantization.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.cppn.network import CPPNNetwork


def make_coordinate_grid(
    width: int, height: int,
    focal_x: float = 0.0, focal_y: float = 0.0,
    armature_angle: float = 0.0,
) -> np.ndarray:
    """Create input coordinate grid: (N, 7) array.

    Columns: [x, y, d, theta, bias, armature_d, armature_t].

    x, y in [-1, 1]. d = distance from center. theta = angle from center.
    armature_d = signed perpendicular distance from the armature axis line.
    armature_t = projection along the armature axis (parametric position).
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

    cos_a = np.cos(armature_angle)
    sin_a = np.sin(armature_angle)
    dx = x_flat - focal_x
    dy = y_flat - focal_y
    armature_d = dx * sin_a - dy * cos_a
    armature_t = dx * cos_a + dy * sin_a

    return np.column_stack([x_flat, y_flat, d_flat, theta_flat, bias_flat,
                            armature_d, armature_t])


def render_cppn(
    network: CPPNNetwork,
    width: int = 256,
    height: int = 256,
    color_mode: str = "rgb",
    palette: np.ndarray | None = None,
    focal_x: float = 0.0,
    focal_y: float = 0.0,
    armature_angle: float = 0.0,
) -> np.ndarray:
    """Render a CPPN to an (H, W, 3) float array in [0, 1].

    Args:
        network: The CPPN to evaluate.
        width: Image width in pixels.
        height: Image height in pixels.
        color_mode: "rgb" or "hsv". Determines how outputs are interpreted.
        palette: Optional (N, 3) array of palette colors for quantization.
        focal_x: Compositional focal point x in [-1, 1].
        focal_y: Compositional focal point y in [-1, 1].
        armature_angle: Dominant compositional axis in radians.

    Returns:
        (height, width, 3) float64 array with RGB values in [0, 1].
    """
    coords = make_coordinate_grid(width, height, focal_x, focal_y,
                                   armature_angle)
    outputs = network.forward(coords)  # (N, num_outputs)

    num_out = outputs.shape[1]

    if color_mode == "hsv":
        h = (outputs[:, 0 % num_out] + 1.0) / 2.0
        s = np.clip((outputs[:, 1 % num_out] + 1.0) / 2.0, 0, 1)
        v = np.clip((outputs[:, 2 % num_out] + 1.0) / 2.0, 0, 1)
        rgb = _hsv_to_rgb_batch(h, s, v)
    else:
        if num_out >= 3:
            r = (outputs[:, 0] + 1.0) / 2.0
            g = (outputs[:, 1] + 1.0) / 2.0
            b = (outputs[:, 2] + 1.0) / 2.0
        elif num_out == 1:
            r = g = b = (outputs[:, 0] + 1.0) / 2.0
        else:
            r = (outputs[:, 0] + 1.0) / 2.0
            g = (outputs[:, 1 % num_out] + 1.0) / 2.0
            b = np.full_like(r, 0.5)

        rgb = np.column_stack([r, g, b])

    rgb = np.clip(rgb, 0.0, 1.0)

    if palette is not None:
        rgb = _quantize_to_palette(rgb, palette)

    rgb = rgb.reshape(height, width, 3)

    # Compositional post-processing: focal contrast + vignette
    rgb = _apply_focal_modulation(rgb, width, height, focal_x, focal_y)

    return rgb


def render_to_image(
    network: CPPNNetwork,
    width: int = 256,
    height: int = 256,
    color_mode: str = "rgb",
    palette: np.ndarray | None = None,
    focal_x: float = 0.0,
    focal_y: float = 0.0,
    armature_angle: float = 0.0,
) -> Image.Image:
    """Render a CPPN directly to a PIL Image."""
    arr = render_cppn(network, width, height, color_mode, palette,
                       focal_x, focal_y, armature_angle)
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


def _apply_focal_modulation(
    rgb: np.ndarray, width: int, height: int,
    focal_x: float, focal_y: float,
) -> np.ndarray:
    """Apply subtle compositional post-processing keyed to the focal point.

    - Contrast boost near focal point (gravitational weighting)
    - Gentle vignette radiating from focal point (entry/exit)
    - Slight saturation boost near focal (draws the eye)
    """
    xs = np.linspace(-1.0, 1.0, width)
    ys = np.linspace(-1.0, 1.0, height)
    x_grid, y_grid = np.meshgrid(xs, ys)

    focal_dist = np.sqrt((x_grid - focal_x) ** 2 + (y_grid - focal_y) ** 2)
    max_dist = np.sqrt(2.0)
    t = np.clip(focal_dist / max_dist, 0, 1)

    # Contrast scaling: boost near focal, soften far away
    mean = rgb.mean(axis=2, keepdims=True)
    contrast_w = 0.85 + 0.30 * (1.0 - t[:, :, np.newaxis])
    rgb = (rgb - mean) * contrast_w + mean

    # Gentle focal vignette
    vignette = 1.0 - 0.10 * (t ** 1.5)
    rgb = rgb * vignette[:, :, np.newaxis]

    return np.clip(rgb, 0.0, 1.0)


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
