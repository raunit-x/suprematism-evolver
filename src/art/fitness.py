"""Optional style-bias fitness functions.

These provide soft fitness bonuses that nudge evolution toward
Malevich or Basquiat aesthetics. They are computed on rendered images
and added to human-selection fitness.
"""

from __future__ import annotations

import numpy as np


def malevich_fitness(image: np.ndarray) -> float:
    """Score an image for Suprematist qualities.

    Rewards:
    - Large flat-colored regions (low local variance)
    - High white-space ratio
    - Limited distinct colors
    - Sharp edges between regions

    Args:
        image: (H, W, 3) float array in [0, 1].

    Returns:
        Fitness score (higher = more Malevich-like). Range ~[0, 1].
    """
    h, w, _ = image.shape
    scores = []

    # 1. White-space ratio: fraction of pixels near white
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    white_ratio = np.mean(luminance > 0.85)
    # Ideal: 30-70% white space
    white_score = 1.0 - 2.0 * abs(white_ratio - 0.5)
    scores.append(max(0, white_score))

    # 2. Local variance (should be low = flat regions)
    # Compute variance in 4x4 patches
    patch = 4
    patches_h = h // patch
    patches_w = w // patch
    cropped = image[: patches_h * patch, : patches_w * patch]
    reshaped = cropped.reshape(patches_h, patch, patches_w, patch, 3)
    patch_var = reshaped.var(axis=(1, 3)).mean()
    flatness_score = np.exp(-patch_var * 50)  # lower variance -> higher score
    scores.append(flatness_score)

    # 3. Color count penalty (fewer distinct hues = better)
    # Quantize to 8 levels per channel and count unique colors
    quantized = (image * 7).astype(int)
    unique_colors = len(
        set(map(tuple, quantized.reshape(-1, 3).tolist()[:2000]))
    )
    color_score = np.exp(-unique_colors / 50)
    scores.append(color_score)

    # 4. Edge sharpness (high gradient at region boundaries)
    gray = luminance
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    # We want bimodal: mostly low gradient (flat) with some high (edges)
    edge_mean = (gx.mean() + gy.mean()) / 2
    edge_score = min(1.0, edge_mean * 10)  # some edges are good
    scores.append(edge_score)

    return float(np.mean(scores))


def basquiat_fitness(image: np.ndarray) -> float:
    """Score an image for Basquiat-like qualities.

    Rewards:
    - High local variance (textural complexity)
    - High contrast
    - Asymmetry
    - Dense mark coverage

    Args:
        image: (H, W, 3) float array in [0, 1].

    Returns:
        Fitness score (higher = more Basquiat-like). Range ~[0, 1].
    """
    h, w, _ = image.shape
    scores = []

    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

    # 1. Contrast: wide luminance range
    lum_range = luminance.max() - luminance.min()
    scores.append(lum_range)

    # 2. Textural complexity (high local variance)
    patch = 4
    patches_h = h // patch
    patches_w = w // patch
    cropped = luminance[: patches_h * patch, : patches_w * patch]
    reshaped = cropped.reshape(patches_h, patch, patches_w, patch)
    patch_var = reshaped.var(axis=(1, 3)).mean()
    complexity_score = 1.0 - np.exp(-patch_var * 50)
    scores.append(complexity_score)

    # 3. Asymmetry (difference between left and right halves)
    left = image[:, : w // 2, :]
    right = image[:, w // 2 : w // 2 + left.shape[1], :]
    right_flipped = right[:, ::-1, :]
    asym = np.mean(np.abs(left - right_flipped))
    asymmetry_score = min(1.0, asym * 5)
    scores.append(asymmetry_score)

    # 4. Mark density (fewer empty/uniform large patches)
    gx = np.abs(np.diff(luminance, axis=1))
    gy = np.abs(np.diff(luminance, axis=0))
    edge_density = (gx.mean() + gy.mean()) / 2
    density_score = min(1.0, edge_density * 15)
    scores.append(density_score)

    return float(np.mean(scores))


def compute_style_fitness(
    image: np.ndarray,
    mode: str = "hybrid",
    weight: float = 0.3,
) -> float:
    """Compute style fitness bonus for a given mode.

    Args:
        image: (H, W, 3) float array.
        mode: "malevich", "basquiat", or "hybrid" (no bias).
        weight: How strongly the style bias affects total fitness [0, 1].

    Returns:
        Style fitness bonus to add to selection fitness.
    """
    if mode == "malevich":
        return weight * malevich_fitness(image)
    elif mode == "basquiat":
        return weight * basquiat_fitness(image)
    else:
        return 0.0
