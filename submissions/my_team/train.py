"""
train.py  —  SHARED (Claude-built glue; rarely edited)

Progressive training pipeline → produces weights.joblib.

It trains ONE model whose training set grows one partition at a time (warm-start), testing
each next 15% block while it's still unseen (the learning curve), then trains the final
model on 100% of the data and saves it.

During development this imports helpers from the repo root (engine/data/...). The grader
NEVER runs this file — it only loads weights.joblib via predict.py. Before final
submission we collapse the needed helpers into this file so it's self-contained
(see plan Stage 6).

Run it either way:
    cd submissions/my_team && python train.py          # CWD = team dir
    python run.py train                                # from repo root
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

TEAM_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEAM_DIR.parent.parent
# make both the repo-root helpers AND the local model.py importable
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEAM_DIR))

from model import ModelArchitecture                       # local, self-contained
from engine import get_device, set_seed, train, evaluate  # Person B
from data import build_loaders, ManifestDataset           # Person D
from augment import build_eval_transform                  # Person C
from split_data import progressive_stages                 # Person D
from error_analysis import log_errors                     # Person D

OUTPUT = TEAM_DIR / "weights.joblib"

# Knobs (Person B may tune). Small per-stage epochs keep the 4-stage curve cheap on laptops.
EPOCHS_PER_STAGE = 5
FINAL_EPOCHS = 10
IMG_SIZE = 224          # full res by default; drop to 128 for a speed escape hatch
BATCH_SIZE = 64
NUM_WORKERS = 2         # set 0 on Windows if you hit DataLoader worker errors


def main(epochs_per_stage=EPOCHS_PER_STAGE, final_epochs=FINAL_EPOCHS,
         img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    set_seed(42)
    device = get_device()
    print(f"device={device}  img_size={img_size}  batch={batch_size}")

    model = ModelArchitecture()
    curve = []

    # ── progressive learning curve: train on accumulated blocks, test the next unseen one ──
    for stage, (train_parts, test_part) in enumerate(progressive_stages(), start=1):
        print(f"\n[stage {stage}] train={'+'.join(train_parts)} -> test={test_part}")
        train_loader, eval_loader = build_loaders(
            train_parts, test_part, img_size=img_size,
            batch_size=batch_size, num_workers=num_workers,
        )
        model, _ = train(model, train_loader, eval_loader,
                          epochs=epochs_per_stage, device=device,
                          log_csv=PROJECT_ROOT / "outputs" / f"train_stage{stage}.csv")
        acc, preds, labels, confs = evaluate(model, eval_loader, device)
        log_errors(eval_loader.dataset.samples, preds, labels, confs, tag=f"stage{stage}_{test_part}")
        curve.append((len(train_loader.dataset), acc))

    print("\n=== learning curve (train_size -> held-out acc) ===")
    for n, a in curve:
        print(f"  {n:6d}  ->  {a:.4f}")

    # ── final model: train on ALL data, then save ──
    print("\n[final] training on 100% (P0..P4)")
    all_parts = ["P0", "P1", "P2", "P3", "P4"]
    final_loader, _ = build_loaders(all_parts, None, img_size=img_size,
                                    batch_size=batch_size, num_workers=num_workers)
    model, _ = train(model, final_loader, None, epochs=final_epochs, device=device)

    joblib.dump(model.cpu().state_dict(), OUTPUT)   # CPU state dict via joblib (grader contract)
    print(f"\nSaved trained weights -> {OUTPUT}")


if __name__ == "__main__":
    main()
