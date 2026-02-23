# Malevich Suprematism: Style Analysis for Generative Art

## Shape Vocabulary (Restricted)

- **Square** -- foundational form, "zero of form", various scales/rotations
- **Rectangle** -- most frequent element (~50-60%), aspect ratios 1:2 to 1:10+
- **Circle** -- primary form, always full (no arcs)
- **Cross** -- perpendicular rectangle intersection, equal-width arms
- **Triangle** -- sparingly used, acute or right
- **Trapezoid/Parallelogram** -- skewed rectangles implying motion
- **Line (thin rectangle)** -- directional vectors, axes
- **Ellipse/Oval** -- later works only

All shapes: flat-filled, hard-edged, no outlines, no gradients, no textures.

## Color Rules

- Background: ALWAYS white
- Shape colors: Black (dominant), Red (most common chromatic), Yellow, Blue, Green, Orange
- Flat, unmodulated -- one solid color per shape
- High saturation, not fluorescent
- 2-6 distinct colors per composition
- Black probability in mixed works: 0.3-0.5

## Composition Rules

- **Dominant diagonal**: lower-left to upper-right, ~45 degrees
- **Asymmetry**: balance through visual weight, never mirror symmetry
- **White = infinite space**: forms float without gravity or horizon
- **Scale hierarchy**: 5x-20x variation between largest and smallest elements
- **Overlap**: 20-40% of pairs overlap, clear z-ordering
- **Edge avoidance**: 5-15% margin from canvas edge
- **Rotation**: 15-45 degrees off-axis typical, avoid pure horizontal/vertical
- **Element count**: 1 (iconic) to ~30 (dense), typical 5-15

## Parameterization Reference

| Parameter | Typical Range |
|-----------|--------------|
| `num_elements` | 1-30 (typical: 5-15) |
| `dominant_axis_angle` | 20-70 degrees |
| `rotation_per_shape` | 15-50 deg off dominant axis |
| `scale_ratio (largest:smallest)` | 3:1 to 10:1 |
| `overlap_probability` | 0.3-0.6 |
| `canvas_margin` | 0.05-0.20 |
| `clustering_factor` | 0.4-0.7 |
| `color_count` | 1-6 |
