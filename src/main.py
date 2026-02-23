#!/usr/bin/env python3
"""Malevich x Basquiat Evolution -- CLI Interface.

Interactive evolution loop:
1. Renders a grid of thumbnails (CPPN or Shape genomes)
2. User selects favorites by index
3. Breeds the next generation from selected parents
4. Repeat

Usage:
    python -m src.main [--engine shapes|cppn] [--mode malevich|basquiat|hybrid] [--palette ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.art.palettes import get_palette_array, MALEVICH_PALETTE, SUPREMATIST_PALETTE
from src.art.fitness import compute_style_fitness
from src.cppn.activations import MALEVICH_ACTIVATIONS, BASQUIAT_ACTIVATIONS, ALL_ACTIVATIONS

OUTPUT_DIR = Path("output")
GRID_COLS = 5


def parse_args():
    p = argparse.ArgumentParser(description="Evolutionary Art: Malevich x Basquiat")
    p.add_argument(
        "--engine",
        choices=["shapes", "cppn"],
        default="shapes",
        help="Evolution engine (default: shapes)",
    )
    p.add_argument(
        "--mode",
        choices=["malevich", "basquiat", "hybrid"],
        default="hybrid",
        help="Style bias mode — CPPN only (default: hybrid = no bias)",
    )
    p.add_argument(
        "--palette",
        choices=["malevich", "basquiat", "hybrid", "none"],
        default="none",
        help="Palette quantization — CPPN only (default: none = full color)",
    )
    p.add_argument(
        "--color-mode",
        choices=["rgb", "hsv"],
        default="rgb",
        help="Color output mode — CPPN only (default: rgb)",
    )
    p.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    p.add_argument("--thumb-size", type=int, default=128, help="Thumbnail size in pixels (default: 128)")
    p.add_argument("--hires", type=int, default=1024, help="High-res export size (default: 1024)")
    p.add_argument("--branch", type=str, default=None, help="Path to a saved genome JSON to branch from")
    return p.parse_args()


def get_activation_options(mode: str) -> list[str] | None:
    if mode == "malevich":
        return MALEVICH_ACTIVATIONS
    elif mode == "basquiat":
        return BASQUIAT_ACTIVATIONS
    return ALL_ACTIVATIONS


def _run_shapes(args):
    from src.shapes.population import ShapePopulation
    from src.shapes.renderer import render_genome, render_population_grid

    palette = list(SUPREMATIST_PALETTE.values())

    config = {
        "pop_size": args.pop_size,
        "num_palette_colors": len(SUPREMATIST_PALETTE),
        "elitism": 2,
        "crossover_rate": 0.7,
        "tournament_k": 3,
    }

    pop = ShapePopulation(config)

    if args.branch:
        print(f"Branching from genome: {args.branch}")
        pop.branch_from(args.branch)
    else:
        pop.initialize()

    print(f"=== Suprematist Shape Evolution ===")
    print(f"Engine: shapes | Population: {args.pop_size}")
    print()

    while True:
        grid_img = render_population_grid(
            pop.genomes, thumb_size=args.thumb_size, cols=GRID_COLS, palette=palette,
        )
        grid_path = OUTPUT_DIR / f"gen_{pop.generation:04d}_grid.png"
        grid_img.save(grid_path)
        print(f"Generation {pop.generation} -- Grid saved to: {grid_path}")

        n = len(pop.genomes)
        rows = (n + GRID_COLS - 1) // GRID_COLS
        print(f"  Images 0-{n-1} ({rows} rows x {GRID_COLS} cols):")
        for r in range(rows):
            start = r * GRID_COLS
            end = min(start + GRID_COLS, n)
            indices = "  ".join(f"[{i:2d}]" for i in range(start, end))
            print(f"    {indices}")
        print()

        print("Commands:")
        print("  <indices>  -- Select favorites (e.g., '2 5 11 17')")
        print("  s <idx>    -- Save genome to JSON")
        print("  e <idx>    -- Export high-res image")
        print("  q          -- Quit")
        print()

        try:
            user_input = input("Select> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            print("Exiting.")
            break

        if user_input.lower().startswith("s "):
            try:
                idx = int(user_input.split()[1])
                save_path = OUTPUT_DIR / f"genome_{pop.generation:04d}_{idx:02d}.json"
                pop.save_genome(idx, save_path)
                print(f"  Saved genome to: {save_path}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
            continue

        if user_input.lower().startswith("e "):
            try:
                idx = int(user_input.split()[1])
                genome = pop.genomes[idx]
                hires_path = OUTPUT_DIR / f"hires_{pop.generation:04d}_{idx:02d}.png"
                img = render_genome(genome, args.hires, args.hires, palette)
                img.save(hires_path)
                print(f"  High-res ({args.hires}x{args.hires}) saved to: {hires_path}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
            continue

        try:
            selected = [int(x) for x in user_input.split()]
            valid = [i for i in selected if 0 <= i < n]
            if not valid:
                print("  No valid indices. Try again.")
                continue
            print(f"  Selected: {valid}")
        except ValueError:
            print("  Invalid input. Enter space-separated indices.")
            continue

        pop.evolve_with_selection(valid)
        print(f"  -> Evolved to generation {pop.generation}\n")


def _run_cppn(args):
    from src.neat.population import Population
    from src.cppn.renderer import render_to_image, render_population_grid

    palette_array = get_palette_array(args.palette)
    activation_opts = get_activation_options(args.mode)

    config = {
        "pop_size": args.pop_size,
        "num_inputs": 5,
        "num_outputs": 3,
        "output_activation": "tanh",
        "activation_options": activation_opts,
        "weight_perturb_rate": 0.8,
        "weight_perturb_power": 0.5,
        "weight_replace_rate": 0.1,
        "add_node_rate": 0.03,
        "add_connection_rate": 0.05,
        "activation_mutation_rate": 0.1,
        "toggle_enable_rate": 0.01,
        "crossover_rate": 0.5,
        "compatibility_threshold": 3.0,
        "elitism": 1,
    }

    pop = Population(config)

    if args.branch:
        print(f"Branching from genome: {args.branch}")
        pop.branch_from(args.branch)
    else:
        pop.initialize()

    print(f"=== Malevich x Basquiat CPPN Evolution ===")
    print(f"Mode: {args.mode} | Palette: {args.palette} | Color: {args.color_mode}")
    print(f"Population: {args.pop_size} | Thumbnails: {args.thumb_size}px")
    print()

    while True:
        networks = pop.get_networks()

        grid = render_population_grid(
            networks,
            thumb_size=args.thumb_size,
            cols=GRID_COLS,
            color_mode=args.color_mode,
            palette=palette_array,
        )

        grid_path = OUTPUT_DIR / f"gen_{pop.generation:04d}_grid.png"
        grid.save(grid_path)
        print(f"Generation {pop.generation} -- Grid saved to: {grid_path}")

        for i, net in enumerate(networks):
            thumb_path = OUTPUT_DIR / f"gen_{pop.generation:04d}_{i:02d}.png"
            img = render_to_image(net, args.thumb_size, args.thumb_size, args.color_mode, palette_array)
            img.save(thumb_path)

        rows = (len(networks) + GRID_COLS - 1) // GRID_COLS
        print(f"  Images 0-{len(networks)-1} ({rows} rows x {GRID_COLS} cols):")
        for r in range(rows):
            start = r * GRID_COLS
            end = min(start + GRID_COLS, len(networks))
            indices = "  ".join(f"[{i:2d}]" for i in range(start, end))
            print(f"    {indices}")
        print()

        print("Commands:")
        print("  <indices>  -- Select favorites (e.g., '2 5 11 17')")
        print("  s <idx>    -- Save genome to JSON (e.g., 's 5')")
        print("  e <idx>    -- Export high-res image (e.g., 'e 5')")
        print("  q          -- Quit")
        print()

        try:
            user_input = input("Select> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "q":
            print("Exiting.")
            break

        if user_input.lower().startswith("s "):
            try:
                idx = int(user_input.split()[1])
                save_path = OUTPUT_DIR / f"genome_{pop.generation:04d}_{idx:02d}.json"
                pop.save_genome(idx, save_path)
                print(f"  Saved genome to: {save_path}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
            continue

        if user_input.lower().startswith("e "):
            try:
                idx = int(user_input.split()[1])
                net = networks[idx]
                hires_path = OUTPUT_DIR / f"hires_{pop.generation:04d}_{idx:02d}.png"
                img = render_to_image(net, args.hires, args.hires, args.color_mode, palette_array)
                img.save(hires_path)
                print(f"  High-res ({args.hires}x{args.hires}) saved to: {hires_path}")
            except (ValueError, IndexError) as e:
                print(f"  Error: {e}")
            continue

        try:
            selected = [int(x) for x in user_input.split()]
            valid = [i for i in selected if 0 <= i < len(networks)]
            if not valid:
                print("  No valid indices. Try again.")
                continue
            print(f"  Selected: {valid}")
        except ValueError:
            print("  Invalid input. Enter space-separated indices.")
            continue

        fitness = [0.0] * len(networks)
        for idx in valid:
            fitness[idx] = 1.0

        if args.mode != "hybrid":
            for i, net in enumerate(networks):
                img_arr = render_to_image(net, 64, 64, args.color_mode).convert("RGB")
                img_np = np.array(img_arr).astype(np.float64) / 255.0
                style_bonus = compute_style_fitness(img_np, args.mode, weight=0.2)
                fitness[i] += style_bonus

        pop.set_fitness(fitness)
        pop.evolve()
        print(f"  -> Evolved to generation {pop.generation}\n")


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.engine == "shapes":
        _run_shapes(args)
    else:
        _run_cppn(args)


if __name__ == "__main__":
    main()
