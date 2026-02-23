"""Multi-CPPN layer compositing.

Allows stacking multiple CPPN outputs with alpha masks and blend modes,
enabling Suprematist geometry on one layer with Basquiat-like texture on another.
"""

from __future__ import annotations

import numpy as np
from src.cppn.network import CPPNNetwork
from src.cppn.renderer import render_cppn


def composite_layers(
    layers: list[dict],
    width: int = 512,
    height: int = 512,
    color_mode: str = "rgb",
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Composite multiple CPPN layers into a single image.

    Each layer dict:
        {
            "network": CPPNNetwork,         # required
            "alpha_network": CPPNNetwork,   # optional, generates per-pixel alpha
            "opacity": float,               # overall opacity [0, 1], default 1.0
            "blend": str,                   # "normal", "multiply", "screen", "overlay"
            "palette": np.ndarray | None,   # optional palette quantization
        }

    Args:
        layers: List of layer dicts, bottom to top.
        width, height: Output resolution.
        color_mode: "rgb" or "hsv".
        background: Background color as (R, G, B) in [0, 1].

    Returns:
        (H, W, 3) float array.
    """
    canvas = np.full((height, width, 3), background, dtype=np.float64)

    for layer in layers:
        net = layer["network"]
        opacity = layer.get("opacity", 1.0)
        blend = layer.get("blend", "normal")
        palette = layer.get("palette")

        # Render this layer's color
        color = render_cppn(net, width, height, color_mode, palette)

        # Compute alpha mask
        if "alpha_network" in layer and layer["alpha_network"] is not None:
            alpha_img = render_cppn(layer["alpha_network"], width, height, "rgb")
            alpha = np.mean(alpha_img, axis=2, keepdims=True)  # grayscale
        else:
            alpha = np.ones((height, width, 1))

        alpha = alpha * opacity

        # Apply blend mode
        blended = _blend(canvas, color, blend)

        # Composite with alpha
        canvas = canvas * (1.0 - alpha) + blended * alpha
        canvas = np.clip(canvas, 0.0, 1.0)

    return canvas


def _blend(base: np.ndarray, top: np.ndarray, mode: str) -> np.ndarray:
    """Apply a blend mode between base and top layers."""
    if mode == "normal":
        return top
    elif mode == "multiply":
        return base * top
    elif mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - top)
    elif mode == "overlay":
        # Overlay: multiply where base < 0.5, screen where base >= 0.5
        return np.where(
            base < 0.5,
            2.0 * base * top,
            1.0 - 2.0 * (1.0 - base) * (1.0 - top),
        )
    else:
        return top
