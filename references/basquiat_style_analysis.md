# Basquiat: Style Analysis for Generative Art

## Line Quality

- Aggressive, gestural, varied -- thin scratches to heavy oil-stick strokes
- Deliberately "untrained" appearance (strategic, not naive)
- Broken, discontinuous contours -- outlines never fully close
- Drips and runs left intentional
- Bimodal weight distribution: very thin (scratched) or very thick (oil stick)

## Color Palette

- Core: Black, Red, Blue, Yellow, White, Raw canvas (warm beige)
- Accents: Orange, Brown, Metallic (gold, silver spray)
- Colors rarely blended -- flat opaque patches or raw gestural strokes
- Extremely high contrast
- Saturation: 0.7-1.0, no pastels
- Background often raw/unprimed canvas or single flat color

## Recurring Symbol Vocabulary

| Symbol | Visual Description | Frequency |
|--------|-------------------|-----------|
| Three-pointed crown | Three inverted V's on horizontal bar | Very high |
| Skull/head | Frontal, circular, hollow eyes, grid teeth | Very high |
| Arrow | Simple directional | High |
| Copyright (C) | Circle-C near words | Moderate |
| Halo/radiance | Circle or radiating lines around head | Moderate |
| Cross/plus | Simple line intersection | Moderate |
| Anatomical diagrams | Rib cages, labeled organs, spine | High |
| Crossed-out words | Text with single strikethrough line | Very high |

## Composition

- **All-over composition**: entire canvas activated, no single focal point
- **Grid-like sectioning**: ~40% of works use implicit/explicit grid divisions
- **Figure-ground ambiguity**: figures dissolve into background
- **Horror vacui**: densely packed (early/mid work)
- **Text as form**: uppercase block letters, not captions
- **Layering**: 3-5 conceptual layers (base wash -> color fields -> figures -> text -> overpainting)
- **Margins**: marks extend to canvas edge, no traditional framing

## Mark-Making Taxonomy (for generative reproduction)

1. Contour strokes (heavy, defining)
2. Scratched lines (thin, incised, revealing underlayer)
3. Fill patches (flat single-color areas)
4. Spray clouds (soft-edged atmospheric)
5. Drips (vertical paint runs)
6. Text marks (uppercase block letters)
7. Symbol marks (crowns, arrows, copyright)
8. Diagram marks (labeled anatomical elements)
9. Erasure/overpainting (partial obscuring, opacity 0.6-0.85)
10. Hatching/scribble (dense energetic fill)

## Parameterization Reference

| Parameter | Typical Range |
|-----------|--------------|
| `line_weight_min` | 1px equivalent |
| `line_weight_max` | 15-20px equivalent |
| `contour_completeness` | 0.5-0.8 (never fully closed) |
| `text_density` | 0.0-1.0 |
| `strikethrough_probability` | 0.3-0.5 |
| `background_mark_density` | High |
| `layer_count` | 3-5 |
| `overpainting_coverage` | 0.1-0.4 of previous layer |
| `symbol_frequency` | crown: very high, skull: very high |
| `saturation` | 0.7-1.0 |
