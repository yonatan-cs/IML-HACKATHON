"""
make_augmented.py  —  OWNER: Person C (Augmentation / robustness)

Builds an OFFLINE augmented DUPLICATE of the training images, as the team requested:
for every image in dataset/train/<class>/, create `copies` new image(s) where a RANDOM
subset (one or more) of these filters is applied:

    grayscale · shift · rotation (45/90/180/270, empty corners filled with the image's
    mean color) · salt-&-pepper noise · zoom · color inversion · color jitter

Output mirrors the train tree:
    dataset/train_aug/<class>/<originalstem>__aug0.jpg   (and __aug1, ... if copies>1)

WHY a mirror with the same filename: it lets data.py find each augmented twin from its
ORIGINAL's path, so the twin inherits the original's split partition. That means an
augmented copy of a P1 (test) image is only ever used when P1 is in the TRAIN set —
never leaking into a test evaluation.

Deterministic: seeded per-image, so all four teammates regenerate byte-for-byte identical
augmented data. It's git-ignored (regenerate locally with `python run.py materialize`).

These are PIL-native filters (Pillow only) so we can save normal viewable JPGs.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageStat

from labels import HF_INDEX_TO_NAME, TARGET_HF_INDICES

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
AUG_DIR = PROJECT_ROOT / "dataset" / "train_aug"


# ── individual filters (PIL -> PIL, RGB in/out) ───────────────────────────────

def _mean_color(img: Image.Image) -> tuple[int, int, int]:
    m = ImageStat.Stat(img).mean
    return tuple(int(round(c)) for c in m[:3])


def f_grayscale(img, rng):
    return ImageOps.grayscale(img).convert("RGB")

def f_shift(img, rng):
    w, h = img.size
    dx, dy = rng.randint(-w // 6, w // 6), rng.randint(-h // 6, h // 6)
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                         fillcolor=_mean_color(img))

def f_rotate(img, rng):
    angle = rng.choice([45, 90, 180, 270])
    return img.rotate(angle, fillcolor=_mean_color(img), expand=False)

def f_salt_pepper(img, rng, amount=0.02):
    arr = np.array(img)
    r = np.random.default_rng(rng.randint(0, 2**31 - 1)).random(arr.shape[:2])
    arr[r < amount / 2] = 0          # pepper
    arr[r > 1 - amount / 2] = 255    # salt
    return Image.fromarray(arr)

def f_zoom(img, rng):
    w, h = img.size
    s = rng.uniform(0.6, 0.9)
    cw, ch = int(w * s), int(h * s)
    l, t = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((l, t, l + cw, t + ch)).resize((w, h))

def f_invert(img, rng):
    return ImageOps.invert(img)

def f_color_jitter(img, rng):
    for Enh in (ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast):
        img = Enh(img).enhance(rng.uniform(0.6, 1.4))
    return img

# Geometric filters apply first, then photometric/noise — a sensible fixed order.
FILTERS = [
    ("rotate", f_rotate), ("shift", f_shift), ("zoom", f_zoom),       # geometric
    ("grayscale", f_grayscale), ("invert", f_invert),                 # color
    ("color_jitter", f_color_jitter), ("salt_pepper", f_salt_pepper), # photometric/noise
]
# Person C TODO: tune which filters / how many per image / probabilities. The provided
# OOD sets (color_jitter, random_rotation) confirm those two are real test manipulations.
MAX_FILTERS_PER_IMAGE = 3


def apply_random_filters(img: Image.Image, rng: random.Random) -> Image.Image:
    """Pick 1..MAX_FILTERS_PER_IMAGE filters at random (in canonical order) and apply them."""
    k = rng.randint(1, MAX_FILTERS_PER_IMAGE)
    chosen = set(rng.sample(range(len(FILTERS)), k))
    out = img
    for i, (_, fn) in enumerate(FILTERS):
        if i in chosen:
            out = fn(out, rng)
    return out.convert("RGB")


# ── driver ────────────────────────────────────────────────────────────────────

def build_augmented(copies: int = 1, seed: int = 42, quality: int = 90) -> int:
    """Create `copies` augmented twin(s) per train image. Returns total images written."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"{TRAIN_DIR} not found — download train_set first.")

    class_names = [HF_INDEX_TO_NAME[hf] for hf in sorted(TARGET_HF_INDICES)]
    written = 0
    for class_name in class_names:
        src_dir = TRAIN_DIR / class_name
        dst_dir = AUG_DIR / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(src_dir.glob("*.jpg")):
            img = Image.open(img_path).convert("RGB")
            for i in range(copies):
                # seed per (file, copy) so every machine produces identical output
                rng = random.Random(f"{seed}:{img_path.name}:{i}")
                aug = apply_random_filters(img, rng)
                aug.save(dst_dir / f"{img_path.stem}__aug{i}.jpg", quality=quality)
                written += 1
        print(f"  {class_name:<18} done ({written} so far)")
    print(f"Wrote {written} augmented images -> {AUG_DIR}")
    return written


def main():
    build_augmented(copies=1)


if __name__ == "__main__":
    main()
