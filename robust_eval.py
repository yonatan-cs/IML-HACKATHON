"""
robust_eval.py  —  OWNER: Person D (Robustness measurement)

Measures the thing we're actually graded on: does accuracy SURVIVE manipulations?

Reports, for a trained weights.joblib:
  - clean accuracy on a Dev partition (materialized to dataset/validation/)
  - accuracy on each PROVIDED OOD set (dataset/augmentations/color_jitter, .../random_rotation)
    → these are the course's sample of the hidden test manipulations (honest signal; we
      never train on them).
  - a single combined SCORE that MIRRORS THE GRADER: 50% clean + 50% OOD, where the OOD
    half is the MEAN over the provided OOD sets. This is the one number the team should
    optimize — it weights robustness exactly as the hidden test does (50/50), so chasing
    clean accuracy alone (which the per-axis breakdown can tempt you into) is penalised.
    The per-axis breakdown is kept so you can see WHICH manipulation costs you.

      SCORE = 0.5 * clean_acc + 0.5 * mean(OOD_acc over provided OOD sets)

    (The real grader's OOD half is a hidden mix; averaging the provided sets is our best
    unbiased estimate of it. If only some axes are present, SCORE uses whatever is there
    and says so — treat a partial SCORE as a lower-confidence proxy.)

A big clean→OOD drop = the model leans on color/orientation shortcuts. Person C's
augmentation work should shrink that gap while keeping clean accuracy; watch SCORE, not
clean acc, to confirm the trade is net-positive.

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

    # ── per-OOD-axis breakdown ──────────────────────────────────────────────────
    # One row per provided OOD axis: its accuracy AND its clean->OOD drop (how much
    # accuracy the manipulation costs). A large drop on an axis = the model leans on
    # that cue (e.g. color for color_jitter, orientation for random_rotation) as a
    # shortcut. Person C targets the WORST-drop axis with augmentation; we track each
    # axis separately (not just the mean) so the team knows WHICH cue to attack.
    ood: list[tuple[str, float]] = []     # (axis_name, acc) kept in display order
    if AUG_ROOT.exists():
        for aug in sorted(p.name for p in AUG_ROOT.iterdir() if p.is_dir()):
            try:
                acc = _acc_on_folder(model, AUG_ROOT, aug, device)
                ood.append((aug, acc))
                gap = "" if clean is None else f"  (drop {clean - acc:+.4f})"
                print(f"  OOD {aug:<22} acc = {acc:.4f}{gap}")
            except Exception as e:
                print(f"  OOD {aug:<22} FAILED — {e}")
    else:
        print("  no dataset/augmentations/ found.")

    ood_accs = [a for _, a in ood]

    # ── per-axis drop summary: worst axis + mean drop ───────────────────────────
    # Surfaces the single manipulation that hurts most (the highest-leverage target
    # for the next augmentation round) and the average robustness gap across axes.
    if clean is not None and ood:
        drops = [(name, clean - a) for name, a in ood]            # +ve = OOD worse
        worst_name, worst_drop = max(drops, key=lambda t: t[1])
        mean_drop = sum(d for _, d in drops) / len(drops)
        print("  " + "-" * 44)
        print(f"  mean clean->OOD drop ({len(drops)} axes)     = {mean_drop:+.4f}")
        print(f"  worst axis: {worst_name:<22} drop = {worst_drop:+.4f}  <-- target this with aug")

    # ── combined SCORE: mirrors the grader's 50% clean + 50% OOD weighting ──────
    # OOD half = unweighted mean over the provided OOD sets (our estimate of the hidden
    # OOD mix). Optimize THIS, not clean acc alone — it's the real graded objective.
    print("  " + "-" * 44)
    if clean is not None and ood_accs:
        ood_mean = sum(ood_accs) / len(ood_accs)
        score = 0.5 * clean + 0.5 * ood_mean
        print(f"  clean half                          acc = {clean:.4f}")
        print(f"  OOD half (mean of {len(ood_accs)} axis/es)        acc = {ood_mean:.4f}")
        print(f"  >> SCORE (0.5*clean + 0.5*OOD-mean)  = {score:.4f}  <-- optimize this")
    elif clean is not None:
        print(f"  >> SCORE: clean only = {clean:.4f} (no OOD sets — partial, low-confidence proxy)")
    elif ood_accs:
        ood_mean = sum(ood_accs) / len(ood_accs)
        print(f"  >> SCORE: OOD-mean only = {ood_mean:.4f} (no clean set materialized — partial proxy)")
    else:
        print("  >> SCORE: unavailable (need at least a clean or an OOD set).")


if __name__ == "__main__":
    main()
