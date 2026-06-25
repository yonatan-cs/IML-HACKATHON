"""
train.py  —  SHARED (Claude-built glue; rarely edited)

GATED progressive trainer → produces weights.joblib.

The team's curriculum: train on ONE data block, judge it on the NEXT unseen 15% block,
manually tune the network, and ONLY advance to more data when the result is good. So this
file trains exactly ONE stage per run and STOPS — it does NOT auto-advance and does NOT
train on 100%. Advancing to the next stage is a human decision (`--stage N+1`).

Each run starts from a FRESH random init ("forget all previous models") — correct when the
architecture changes between iterations, and avoids stale-weight surprises.

    stage 1: train P0 (40%)            -> test P1   <- default, start here
    stage 2: train P0+P1 (55%)         -> test P2
    stage 3: train P0+P1+P2 (70%)      -> test P3
    stage 4: train P0+P1+P2+P3 (85%)   -> test P4

The training set of each stage includes the offline augmented twins of its parts; the test
block is ALWAYS clean originals (data.py guarantees no twin leakage into eval).

"Good enough" to advance (your call): clean-Dev on the test block ~0.55-0.65, comfortably
above the naive floor 0.2230 (measured), and error analysis shows no systematic shortcut.

During development this imports helpers from the repo root (engine/data/...). The grader
NEVER runs this file — it only loads weights.joblib via predict.py. Before final
submission we collapse the needed helpers into this file so it's self-contained
(see plan Stage 6).

Run it either way:
    cd submissions/my_team && python train.py          # CWD = team dir (stage 1)
    python run.py train                                # from repo root (stage 1)
    python run.py train --stage 2                      # advance, when you choose to
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

import numpy as np

from model import ModelArchitecture                       # local, self-contained
from engine import get_device, set_seed, train, evaluate  # Person B
from data import build_loaders, ManifestDataset           # Person D
from augment import build_eval_transform                  # Person C
from split_data import progressive_stages                 # Person D
from error_analysis import log_errors                     # Person D
from baseline_naive import main as _naive  # noqa: F401  (floor reference: 0.2230, measured via run.py baseline)

OUTPUT = TEAM_DIR / "weights.joblib"

# Knobs (Person B may tune).
MAX_EPOCHS = 40         # per-stage cap; engine.train early-stops at its plateau before this
IMG_SIZE = 224          # full res by default; drop to 128 for a speed escape hatch
BATCH_SIZE = 64
NUM_WORKERS = 2         # set 0 on Windows if you hit DataLoader worker errors
NAIVE_FLOOR = 0.2230    # measured softmax-regression floor (run.py baseline, 32x32 logreg); CNN must comfortably beat it


def _per_class_acc(preds, labels, n_classes=20):
    """Return list of (class_idx, acc, support) — shows WHERE the model falls."""
    out = []
    for c in range(n_classes):
        m = labels == c
        sup = int(m.sum())
        acc = float((preds[m] == c).mean()) if sup else 0.0
        out.append((c, acc, sup))
    return out


def main(stage=1, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
         num_workers=NUM_WORKERS, epochs=MAX_EPOCHS):
    """Train ONE stage from a fresh init, judge it on the next unseen block, save, STOP."""
    set_seed(42)
    device = get_device()

    stages = list(progressive_stages())          # [(['P0'],'P1'), (['P0','P1'],'P2'), ...]
    if not (1 <= stage <= len(stages)):
        raise SystemExit(f"--stage must be 1..{len(stages)} (got {stage})")
    train_parts, test_part = stages[stage - 1]

    print(f"device={device}  img_size={img_size}  batch={batch_size}  max_epochs={epochs}")
    print(f"[stage {stage}] train={'+'.join(train_parts)}  ->  test={test_part}  "
          f"(fresh random init — previous models forgotten)")

    model = ModelArchitecture()                  # FRESH init every run
    train_loader, eval_loader = build_loaders(
        train_parts, test_part, img_size=img_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    model, _ = train(model, train_loader, eval_loader,
                     epochs=epochs, device=device,
                     log_csv=PROJECT_ROOT / "outputs" / f"train_stage{stage}.csv")

    acc, preds, labels, confs = evaluate(model, eval_loader, device)

    # where does it fall? per-class accuracy on the held-out block + misclassified dump
    print(f"\n=== stage {stage}: held-out {test_part} clean acc = {acc:.4f} "
          f"(naive floor {NAIVE_FLOOR}) ===")
    print("  per-class accuracy (worst first):")
    for c, a, sup in sorted(_per_class_acc(preds, labels), key=lambda t: t[1])[:8]:
        print(f"    class {c:2d}  acc {a:.3f}  (n={sup})")
    log_errors(eval_loader.dataset.samples, preds, labels, confs, tag=f"stage{stage}_{test_part}")

    joblib.dump(model.cpu().state_dict(), OUTPUT)   # CPU state dict via joblib (grader contract)
    print(f"\nSaved -> {OUTPUT}")

    verdict = "GOOD — advance when ready" if acc >= 0.55 else \
              "weak — tune model/aug and rerun this stage" if acc >= NAIVE_FLOOR else \
              "BROKEN — below naive floor, something is wrong"
    print(f"verdict: {verdict}")
    if stage < len(stages):
        print(f"to advance (your call): python run.py train --stage {stage + 1}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    a = ap.parse_args()
    main(stage=a.stage, img_size=a.img_size, batch_size=a.batch,
         num_workers=a.workers, epochs=a.epochs)
