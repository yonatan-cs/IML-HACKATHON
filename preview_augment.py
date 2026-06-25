"""
preview_augment.py  —  OWNER: Person C (Augmentation / robustness)

A *visual* sanity tool for Part B. It runs every augment.py pipeline on a sample
image and writes ONE grid PNG you can open and eyeball. No dataset needed — point it
at any photo on your laptop, or let it generate a synthetic test image.

Why this matters: the augmentations are the score-moving lever, but a probability in
code tells you nothing about whether the result is still recognizable. Seeing them
lets you catch label-breaking transforms (e.g. color-invert turning a yellow lemon
blue) and pick good strengths. Great material for the README + interview too.

Usage:
    python preview_augment.py                 # synthetic test image
    python preview_augment.py path/to/img.jpg # your own image
    python preview_augment.py img.jpg out.png  # custom output path

Output: preview_augment.png (a labeled grid: original | each OOD probe | random
train-transform samples).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from augment import (
    IMAGENET_MEAN, IMAGENET_STD,
    build_train_transform, build_eval_transform, OOD_TRANSFORMS,
)

CELL = 200          # px per image cell in the grid
PAD = 8             # px padding around each cell
LABEL_H = 22        # px reserved under each cell for its text label
COLS = 4            # grid columns


def make_synthetic_image(size: int = 256) -> Image.Image:
    """A recognizable test image: a colored disc on a contrasting background with a
    stripe, so rotations / color shifts / inversions are obviously visible."""
    img = Image.new("RGB", (size, size), (70, 140, 90))      # green-ish background
    d = ImageDraw.Draw(img)
    d.ellipse([size * 0.25, size * 0.25, size * 0.75, size * 0.75],
              fill=(220, 180, 40))                            # yellow disc ("lemon")
    d.rectangle([0, size * 0.45, size, size * 0.55], fill=(180, 40, 40))  # red stripe
    return img


def denormalize(t: torch.Tensor) -> Image.Image:
    """Undo ImageNet normalization so a normalized tensor is viewable as a PIL image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (t.detach().cpu() * std + mean).clamp(0, 1)
    arr = (x * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def build_panels(src: Image.Image) -> list[tuple[str, Image.Image]]:
    """Return [(label, PIL image)] for the grid: original, eval, each OOD, train samples."""
    panels: list[tuple[str, Image.Image]] = [("original", src.copy())]
    # eval (deterministic, what the grader sees) + every single-axis OOD probe
    panels.append(("eval (grader)", denormalize(build_eval_transform()(src))))
    for name, tf in OOD_TRANSFORMS.items():
        panels.append((f"OOD: {name}", denormalize(tf(src))))
    # several random draws of the heavy TRAIN transform — shows the variety per epoch
    train_tf = build_train_transform()
    for i in range(4):
        panels.append((f"train #{i + 1}", denormalize(train_tf(src))))
    return panels


def compose_grid(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    rows = (len(panels) + COLS - 1) // COLS
    cell_w = CELL + 2 * PAD
    cell_h = CELL + 2 * PAD + LABEL_H
    canvas = Image.new("RGB", (COLS * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    for idx, (label, img) in enumerate(panels):
        r, c = divmod(idx, COLS)
        x0, y0 = c * cell_w + PAD, r * cell_h + PAD
        canvas.paste(img.resize((CELL, CELL)), (x0, y0))
        draw.text((x0, y0 + CELL + 4), label, fill=(20, 20, 20))
    return canvas


def main():
    args = sys.argv[1:]
    if args and Path(args[0]).exists():
        src = Image.open(args[0]).convert("RGB")
        print(f"loaded image: {args[0]}")
    else:
        src = make_synthetic_image()
        print("no image given (or path missing) — using synthetic test image")
    out = Path(args[1]) if len(args) > 1 else Path("preview_augment.png")

    grid = compose_grid(build_panels(src))
    grid.save(out)
    print(f"wrote {out}  ({grid.width}x{grid.height})  — open it to eyeball the augmentations")


if __name__ == "__main__":
    main()
