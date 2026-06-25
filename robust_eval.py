"""
robust_eval.py  —  OWNER: Person D (Robustness measurement)

Measures the thing we're actually graded on: does accuracy SURVIVE manipulations?

Reports, for a trained weights.joblib:
  - clean accuracy on a Dev partition (materialized to dataset/validation/)
  - accuracy on each PROVIDED OOD set (dataset/augmentations/color_jitter, .../random_rotation)
    → these are the course's sample of the hidden test manipulations (honest signal; we
      never train on them).

A big clean→OOD drop = the model leans on color/orientation shortcuts. Person C's
augmentation work should shrink that gap while keeping clean accuracy.

Reuses base_model.ImageNetSubset (provided) for folder-based loading.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

from base_model import ImageNetSubset
from augment import build_eval_transform
from engine import get_device, evaluate

PROJECT_ROOT = Path(__file__).resolve().parent
TEAM_DIR = PROJECT_ROOT / "submissions" / "my_team"
AUG_ROOT = PROJECT_ROOT / "dataset" / "augmentations"


def load_trained_model(team_dir: Path = TEAM_DIR):
    """Import ModelArchitecture from the team's model.py and load weights.joblib into it."""
    spec = importlib.util.spec_from_file_location("team_model", team_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ModelArchitecture()
    state = joblib.load(team_dir / "weights.joblib")
    model.load_state_dict(state)
    model.eval()
    return model


def _acc_on_folder(model, root: Path, split: str, device) -> float:
    ds = ImageNetSubset(root, split=split, transform=build_eval_transform())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    acc, *_ = evaluate(model, loader, device)
    return acc


def main():
    device = get_device()
    model = load_trained_model().to(device)

    print("=== Robustness report ===")
    val_dir = PROJECT_ROOT / "dataset" / "validation"
    if val_dir.exists():
        clean = _acc_on_folder(model, PROJECT_ROOT / "dataset", "validation", device)
        print(f"  clean (dataset/validation)      acc = {clean:.4f}")
    else:
        clean = None
        print("  clean: (run `python run.py split` to materialize dataset/validation)")

    if AUG_ROOT.exists():
        for aug in sorted(p.name for p in AUG_ROOT.iterdir() if p.is_dir()):
            try:
                acc = _acc_on_folder(model, AUG_ROOT, aug, device)
                gap = "" if clean is None else f"  (drop {clean - acc:+.4f})"
                print(f"  OOD {aug:<22} acc = {acc:.4f}{gap}")
            except Exception as e:
                print(f"  OOD {aug:<22} FAILED — {e}")
    else:
        print("  no dataset/augmentations/ found.")


if __name__ == "__main__":
    main()
