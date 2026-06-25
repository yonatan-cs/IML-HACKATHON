"""
error_analysis.py  —  OWNER: Person D (Error analysis)

Methodology step 5: every Dev / temporary-Test evaluation auto-writes a detailed error
log so the team can eyeball WHAT the model gets wrong and find patterns to fix.

`log_errors(...)` is called automatically by run.py / train.py after each eval. It writes:
  outputs/errors_<tag>.csv      — one row per MISCLASSIFIED image: path,true,pred,confidence
  outputs/confusion_<tag>.csv   — 20x20 confusion matrix (rows=true, cols=pred)

Only run this on Dev/Test blocks you're allowed to inspect — NEVER on the final held-out
Test until the very end (see methodology: test-touched-once).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from labels import IDX_TO_NAME

PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "outputs"


def log_errors(samples, preds, labels, confs, tag: str) -> tuple[Path, Path]:
    """
    Args:
        samples: list of (path, label) in the SAME order as preds (ManifestDataset.samples
                 from a shuffle=False loader). Pass model.predict_loader order.
        preds/labels/confs: np arrays from engine.evaluate.
        tag: short name for the files, e.g. "P1" or "stage3_P2".
    """
    OUT.mkdir(parents=True, exist_ok=True)
    preds, labels, confs = np.asarray(preds), np.asarray(labels), np.asarray(confs)

    err_path = OUT / f"errors_{tag}.csv"
    n_err = 0
    with open(err_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_idx", "true_name", "pred_idx", "pred_name", "confidence"])
        for (path, _), t, p, c in zip(samples, labels, preds, confs):
            if t != p:
                n_err += 1
                w.writerow([str(path), int(t), IDX_TO_NAME[int(t)],
                            int(p), IDX_TO_NAME[int(p)], f"{float(c):.4f}"])

    # confusion matrix (20x20)
    cm = np.zeros((20, 20), dtype=int)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    cm_path = OUT / f"confusion_{tag}.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [IDX_TO_NAME[i] for i in range(20)])
        for i in range(20):
            w.writerow([IDX_TO_NAME[i]] + cm[i].tolist())

    acc = float((preds == labels).mean()) if len(preds) else 0.0
    print(f"  error log [{tag}]: acc={acc:.4f}  errors={n_err}/{len(preds)}  -> {err_path.name}")
    # most-confused pair (off-diagonal) — a quick pattern hint
    off = cm.copy()
    np.fill_diagonal(off, 0)
    if off.sum() > 0:
        i, j = np.unravel_index(off.argmax(), off.shape)
        print(f"    most confused: {IDX_TO_NAME[i]} -> {IDX_TO_NAME[j]} ({off[i, j]}x)")
    return err_path, cm_path
