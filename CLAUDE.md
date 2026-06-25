# OpenWolf

@.wolf/OPENWOLF.md

This project uses OpenWolf for context management. Read and follow .wolf/OPENWOLF.md every session. Check .wolf/cerebrum.md before generating code. Check .wolf/anatomy.md before reading files.


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

IML (67577) Hackathon 2026 — **Challenge 2: Robust Image Classification**. Train a CNN **from random init** to classify images into 20 ImageNet-subset classes, optimizing for *both* clean accuracy *and* robustness to visual manipulations (background swaps, lighting, color distortion).

Final test set = **50% in-domain** (like the train data) + **50% out-of-domain** (held-out augmentations not given to us). Score combines clean accuracy with robustness. The whole game is: don't let the model latch onto background/color shortcuts.

Starter files (`base_model.py`, `labels.py`, `evaluate.py`, `check_submission.py`, `submissions/`, `dataset/labels.json`) come from the Moodle starter zip; the image data (`dataset/train/`, `dataset/validation/`) is downloaded separately and is **not** committed.

## Hard constraints (graded — violating these = disqualified or zero)

- **Train from scratch.** No pretrained weights, no external datasets, no torchvision pretrained backbones, no network downloads during train/load/eval.
- `model.py` must define a class named exactly `ModelArchitecture` (an `nn.Module`) whose `forward` returns **logits of shape `[B, 20]`**. The architecture must be reconstructable from `model.py` alone — do not hide layers in `train.py`.
- `train.py` must save weights to exactly **`weights.joblib`**, as a **CPU** state dict via joblib:
  ```python
  joblib.dump(model.cpu().state_dict(), "weights.joblib")
  ```
  Note: `base_model.py`'s docstring mentions `torch.save`, but the grader (`evaluate.py` / `check_submission.py`) loads with `joblib.load` — **joblib is authoritative**.
- `predict.py` **must not be changed.** It defines class `Model(BaseModel)`; `predict(x)` returns a **Long tensor of class indices `[B]`**, values in `0..19` (argmax of logits — never logits/probabilities).
- `predict.predict` receives `x` of shape **`[B, 3, 224, 224]`**, already normalized with ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)`. The architecture must accept 224×224 input.

## Label mapping

`labels.py` is the source of truth: local index `0..19` = HF ImageNet classes **sorted by HF index**. `dataset/labels.json` agrees with it. Dataset loaders (`ImageNetSubset`) map each class folder name → its sorted local index, so folder names must match `HF_INDEX_TO_NAME` exactly. Run `python labels.py` to print the full index↔name table.

## Where the work happens

Develop inside `submissions/my_team/` (rename to `challenge2_<id>_<id>_<id>` for final submission). Three files to fill in:
- `model.py` — define `ModelArchitecture` (currently `raise NotImplementedError`).
- `train.py` — full training pipeline; must produce `weights.joblib`. `DATA_ROOT = ../../dataset` (relative to the team folder). Use `ImageNetSubset` from `base_model.py` for loading.
- `predict.py` — leave as-is (the fixed grader interface).

`submissions/dummy_baseline/` is a working reference: a zero-weight CNN that always predicts class 0. Read it to see the exact `model.py` / `predict.py` / `train.py` shape the grader expects.

## Commands

Run from the project root (where `evaluate.py`, `labels.py`, `dataset/` live):

```bash
python check_submission.py [team_name]   # structural check: required files, class names, predict() shape/range. No team_name = check all.
python evaluate.py                         # local leaderboard: loads each submission, runs inference on dataset/validation, prints accuracy ranking
python labels.py                           # print local-index → HF-index → class-name table
```

Training (produces `weights.joblib` inside the team folder):
```bash
cd submissions/my_team && python train.py
```

`evaluate.py` reads validation images from `dataset/validation/<class_name>/*.jpg` (Resize 256 → CenterCrop 224 → ToTensor → ImageNet normalize). You must create that split yourself from the raw `train_set` (1000 raw samples/class) before `evaluate.py` will run — step 2 of the recommended pipeline is "split before doing anything else."

## Robustness strategy (the actual challenge)

Standard accuracy is the easy half. The graded edge is invariance to held-out augmentations. Approach: simulate background/lighting/color shifts as **train-time augmentations** (and optionally a separate stress-test val split to measure robustness). The held-out test augmentations are unknown, so over-fitting to one specific augmentation won't generalize — aim for broad, label-preserving perturbations.

## Submission packaging

Final folder `challenge2_<ids>/` contains only: `train.py`, `model.py`, `predict.py`, `weights.joblib`, `README`. Do **not** submit the dataset, venvs, caches, intermediate checkpoints, or anything with absolute local paths. `README` line 1 = names (comma-separated, no spaces), line 2 = IDs, then a blank line, then the model/manipulation write-up.
