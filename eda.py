"""
eda.py  —  OWNER: Person D (Exploratory Data Analysis)

Methodology step 3: LOOK at the data before modeling. Three quick checks:
  1. class balance      — are all 20 classes ~equally represented?
  2. label/folder sanity — do folder names match labels.py exactly? (a typo = silent mislabel)
  3. sample grid         — a montage so the team can eyeball backgrounds / lighting / colors
                           (this directly informs which augmentations Person C should add)

Outputs a printed report + outputs/eda_grid.jpg.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
OUT = PROJECT_ROOT / "outputs"


def expected_class_names() -> list[str]:
    return [HF_INDEX_TO_NAME[hf] for hf in sorted(TARGET_HF_INDICES)]


def class_balance() -> dict[str, int]:
    counts = {}
    for name in expected_class_names():
        d = TRAIN_DIR / name
        counts[name] = len(list(d.glob("*.jpg"))) if d.exists() else -1
    return counts


def check_label_sanity() -> list[str]:
    """Return a list of problems (empty list = all good)."""
    problems = []
    present = {p.name for p in TRAIN_DIR.iterdir() if p.is_dir()} if TRAIN_DIR.exists() else set()
    for name in expected_class_names():
        if name not in present:
            problems.append(f"missing class folder: {name}")
    extra = present - set(expected_class_names())
    if extra:
        problems.append(f"unexpected folders (won't be loaded): {sorted(extra)}")
    return problems


def sample_grid(per_class: int = 5, thumb: int = 96) -> Path:
    """Montage: one row per class, `per_class` thumbnails each. Saved to outputs/eda_grid.jpg."""
    names = expected_class_names()
    OUT.mkdir(parents=True, exist_ok=True)
    grid = Image.new("RGB", (per_class * thumb, len(names) * thumb), (20, 20, 20))
    for r, name in enumerate(names):
        imgs = sorted((TRAIN_DIR / name).glob("*.jpg"))[:per_class]
        for c, p in enumerate(imgs):
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
            grid.paste(im, (c * thumb, r * thumb))
    path = OUT / "eda_grid.jpg"
    grid.save(path, quality=85)
    return path


def main():
    print("=== Class balance (images per class) ===")
    counts = class_balance()
    for name, c in counts.items():
        print(f"  {name:<18} {c}")
    vals = [c for c in counts.values() if c >= 0]
    if vals:
        print(f"  total={sum(vals)}  min={min(vals)}  max={max(vals)}  "
              f"{'BALANCED' if max(vals) == min(vals) else 'IMBALANCED — consider class weights'}")

    print("\n=== Label / folder sanity ===")
    problems = check_label_sanity()
    print("  OK — all 20 folders match labels.py" if not problems else "\n".join("  " + p for p in problems))

    print("\n=== Sample grid ===")
    if TRAIN_DIR.exists():
        print("  saved:", sample_grid())
        print("  -> open it; note backgrounds/lighting/color to guide augmentation (Person C).")
    else:
        print("  (no dataset/train — download data first)")


if __name__ == "__main__":
    main()
