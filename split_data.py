"""
split_data.py  —  OWNER: Person D (Data + methodology)

Builds the seeded, STRATIFIED, 5-partition PROGRESSIVE split described in the plan:

    P0 = 40%,  P1..P4 = 15% each   (per class, so every partition mirrors the whole)

Progressive (incremental-data) training then goes:
    train P0          -> test P1
    train P0+P1       -> test P2
    train P0+P1+P2    -> test P3
    train P0..P3      -> test P4
    final model: train P0..P4 (100%)

Each 15% block is tested exactly once WHILE UNSEEN, then folded into train → no leakage.

----------------------------------------------------------------------------------
HOW THIS MAPS TO THE METHODOLOGY'S Train / Dev / Test ("Test touched once")
----------------------------------------------------------------------------------
The plan's iron rule is a 3-way split where the local Test is touched EXACTLY ONCE, at
the end, and never tuned against. We do NOT carve out one fixed reserved Test partition;
instead the SAME one-shot-test discipline is realised PROGRESSIVELY. At every stage the
roles are:

    stage k:  Train = P0..P(k-1)  (all blocks already seen / folded in)
              Test  = Pk          (the next 15% block, still UNSEEN)

Pk plays the role of the methodology's local Test *for that one stage*: we score on it
exactly ONCE while it is unseen (an honest, un-overfit generalization estimate), THEN it
is folded into Train for the next stage and is never used as a clean test again. Because
each block is scored only on its first (and only) unseen appearance, no block is ever
tuned against as a test set — the "touch the test once" guarantee holds per block.

  - Dev / tuning happens on the CURRENT Train via engine.py's early-stopping (the
    validation slice the trainer holds out from Train) and on already-seen blocks; all
    architecture iteration + error_analysis runs there.
  - The block being scored at a stage (Pk) is the one-shot Test for that stage — look at
    its accuracy, do NOT error-analyze or tune against it before it's folded in.
  - The HIDDEN GRADER test (50% clean + 50% OOD) is the ULTIMATE one-shot test: we never
    see it, so it can never be overfit. Our progressive Pk scores + robust_eval's SCORE
    are our internal proxies for it.

This is a deliberate design choice (more data exercised as test over the run, fresh init
each stage) — it does NOT weaken the "test touched once" rule; it applies it block-by-block.
Do NOT silently change the split logic to chase a different mapping without updating this.

Output: `dataset/splits.json` — the single source of truth all teammates load from
(deterministic given the seed, so it's git-ignored and regenerated locally).
`materialize_validation()` copies one partition into `dataset/validation/` so the
provided `evaluate.py` (which reads dataset/validation/<class>/*.jpg) keeps working.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
SPLITS_JSON = PROJECT_ROOT / "dataset" / "splits.json"

# Partition fractions. Person D TODO: change here if you want different ratios.
PARTITION_FRACS = {"P0": 0.40, "P1": 0.15, "P2": 0.15, "P3": 0.15, "P4": 0.15}


def _class_name_to_local_idx() -> dict[str, int]:
    """folder name (e.g. 'tiger') -> local index 0..19, via labels.py (the source of truth)."""
    return {HF_INDEX_TO_NAME[hf]: HF_INDEX_TO_IDX[hf] for hf in sorted(TARGET_HF_INDICES)}


def build_splits(seed: int = 42) -> dict:
    """
    Create the stratified partitions and write dataset/splits.json. Returns the dict.

    Stratified = we split EACH class's 1000 images by the same fractions, so P0 has 40%
    of every class (not 40% of classes). Seeded shuffle = reproducible across machines.
    """
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"{TRAIN_DIR} not found — download train_set first.")

    name_to_idx = _class_name_to_local_idx()
    rng = random.Random(seed)
    partitions: dict[str, list] = {p: [] for p in PARTITION_FRACS}

    for class_name, local_idx in sorted(name_to_idx.items()):
        class_dir = TRAIN_DIR / class_name
        imgs = sorted(p for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png")
                      for p in class_dir.glob(ext))
        if not imgs:
            raise RuntimeError(f"No images in {class_dir}")
        rng.shuffle(imgs)

        # cut points from the fractions, in fixed P0..P4 order
        n = len(imgs)
        names = list(PARTITION_FRACS)
        counts = [int(round(PARTITION_FRACS[p] * n)) for p in names]
        counts[-1] = n - sum(counts[:-1])           # last takes the remainder (exact total)

        start = 0
        for pname, c in zip(names, counts):
            for img in imgs[start:start + c]:
                rel = img.relative_to(PROJECT_ROOT).as_posix()   # portable relative path
                partitions[pname].append([rel, local_idx])
            start += c

    out = {
        "seed": seed,
        "fractions": PARTITION_FRACS,
        "counts": {p: len(v) for p, v in partitions.items()},
        "partitions": partitions,
    }
    SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(json.dumps(out))
    print(f"Wrote {SPLITS_JSON}")
    print("  per-partition totals:", out["counts"])
    return out


def load_splits() -> dict:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(f"{SPLITS_JSON} missing — run `python run.py split` first.")
    return json.loads(SPLITS_JSON.read_text())


def progressive_stages() -> list[tuple[list[str], str]]:
    """The 4 incremental iterations: (train_partitions, test_partition)."""
    return [
        (["P0"], "P1"),
        (["P0", "P1"], "P2"),
        (["P0", "P1", "P2"], "P3"),
        (["P0", "P1", "P2", "P3"], "P4"),
    ]


def materialize_validation(partition: str = "P1", out_dir: Path | None = None) -> Path:
    """
    Copy one partition into dataset/validation/<class>/*.jpg so the PROVIDED evaluate.py
    can score it. (evaluate.py can't read our manifest, so we give it a real folder.)
    """
    out_dir = out_dir or (PROJECT_ROOT / "dataset" / "validation")
    splits = load_splits()
    idx_to_name = {v: k for k, v in _class_name_to_local_idx().items()}

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for rel, local_idx in splits["partitions"][partition]:
        src = PROJECT_ROOT / rel
        dst_dir = out_dir / idx_to_name[local_idx]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
    print(f"Materialized partition {partition} -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    build_splits()
