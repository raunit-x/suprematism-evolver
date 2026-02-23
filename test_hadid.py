#!/usr/bin/env python3
"""Generate sample Hadid-style architecton compositions.

Produces a set of test images showing 3D oblique-projected cuboid
stacks inspired by Malevich's Arkhitektons and Zaha Hadid's paintings.

Usage:
    python test_hadid.py
"""

import random
from pathlib import Path

from src.shapes.genome import create_architecton
from src.shapes.renderer import render_genome
from src.art.palettes import HADID_PALETTE

OUTPUT = Path("output")
PALETTE = list(HADID_PALETTE.values())
NUM_COLORS = len(PALETTE)


def main():
    OUTPUT.mkdir(exist_ok=True)

    seeds = [17, 42, 55, 73, 99, 128, 200, 314]

    for i, seed in enumerate(seeds):
        random.seed(seed)
        genome = create_architecton(num_palette_colors=NUM_COLORS)
        img = render_genome(genome, 1024, 1024, PALETTE)
        path = OUTPUT / f"hadid_test_{i}_seed{seed}.png"
        img.save(path)
        print(f"[{i+1}/{len(seeds)}] Saved {path}  "
              f"(proj={genome.projection_angle:.0f}deg, "
              f"str={genome.projection_strength:.2f}, "
              f"shapes={genome.total_shapes()})")

    print(f"\nDone — {len(seeds)} images in {OUTPUT}/")


if __name__ == "__main__":
    main()
