# Team setup — IML Challenge 2

Everything in this repo is **code + docs only**. The 2.1GB dataset and all its
regenerable derivatives (splits, augmented copies, validation folder, weights) are **not**
in git — each teammate reproduces them locally. They're seeded, so everyone gets identical
data.

## 1. One-time setup (every teammate)

```bash
# clone
git clone https://github.com/yonatan-cs/IML-HACKATHON.git
cd IML-HACKATHON

# python env (use 3.11 — torch has no 3.14 wheels yet)
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the data (out-of-band — NOT via git)

Download `train_set` (and `augmentations`) from the course link and place them so you have:

```
dataset/train/<class_name>/*.jpg          # 20 classes × 1000 images
dataset/augmentations/color_jitter/...    # provided OOD examples
dataset/augmentations/random_rotation/...
```

## 3. Reproduce the local artifacts (deterministic — identical for all 4)

```bash
python run.py split        # -> dataset/splits.json + dataset/validation/  (seeded 40/15/15/15/15)
python run.py materialize  # -> dataset/train_aug/  (offline augmented twins, ~80s, ~1.5GB)
```

Both are seeded, so your `splits.json` and `train_aug/` match everyone else's exactly.
Re-run them only if the seed or split logic changes.

## 4. Daily pipeline

```bash
python run.py eda          # look at the data (class balance, sample grid, label sanity)
python run.py baseline     # naive logistic-regression floor (the CNN must beat this)
python run.py train        # progressive training -> submissions/my_team/weights.joblib
python run.py robust       # clean vs provided-OOD accuracy
python run.py eval         # provided leaderboard (evaluate.py)
python run.py errors --partition P1   # detailed misclassification log -> outputs/
```

## Who owns what (edit mostly your own file → clean merges)

| Person | Lever | File(s) |
|--------|-------|---------|
| A | model architecture | `submissions/my_team/model.py` |
| B | training / optimization | `engine.py` |
| C | augmentation / robustness | `augment.py`, `make_augmented.py` |
| D | data / methodology / eval | `split_data.py`, `data.py`, `eda.py`, `baseline_naive.py`, `robust_eval.py`, `error_analysis.py` |

`model.py` must stay self-contained (import only torch) — the grader rebuilds it from that
file alone. `predict.py` is frozen, never edit it.

## Git workflow

- **`git pull` before `git push`** — `.wolf/memory.md` and `.wolf/cerebrum.md` use union-merge
  (changes stack, no conflicts).
- Don't commit `weights.joblib` during dev (git-ignored). Near the end, the team picks the
  single best model and force-adds that one file.
- Never commit `dataset/`, `.venv/`, or `outputs/` (all git-ignored).
- Hit a real bug + fixed it? Log it in `.wolf/cerebrum.md` → `## Do-Not-Repeat` (shared,
  conflict-free). `.wolf/buglog.json` is local-only auto-noise — don't rely on it.
