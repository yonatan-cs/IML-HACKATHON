# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-06-28

## User Preferences

- Claude must NEVER appear as a GitHub contributor — even when Claude runs the push. Commits author+committer = the human; no `Co-Authored-By`/Claude/Anthropic trailers anywhere. All work attributed to the human team ("הכל אנחנו").

## Key Learnings

- **Project:** IML-HACKATHON. This human user = **Person A (Architecture / model.py)** as of 2026-06-25 ("תופס פיקוד על A"). Owns model.py only. User does NOT have the dataset locally — a different machine runs training; user's job is to make their file excellent + correct + interview-defensible. (Was recorded as Person B / engine.py earlier — superseded.)

- **Course-coverage map (interview-defensible — verified via NotebookLM 2026-06-25).**
  IN syllabus (safe + defendable): conv+pooling (Max/Avg/Stride, lec 8), **residual/skip** (`y=f(x)+x`, He 2016, Transformers unit — mitigates vanishing gradients), Dropout (~50%), ReLU/softmax, **L2/Ridge = weight decay**, GAP, bias-variance/overfitting, early-stop (Patience), SGD + LR-schedule (`η_t=1/√t`), data aug.
  OUT of syllabus (flag): **BatchNorm** (only LayerNorm taught → defensible only BY ANALOGY), **momentum/Adam/AdamW**, advanced init Kaiming/Xavier/He (only "random init" taught).

- **Dataset layout gotcha:** raw data lives at `dataset/train_set/train/<class>/` (20×1000 jpg), but pipeline (`split_data.py`+`make_augmented.py`) reads `dataset/train/`. Bridge with symlink `dataset/train -> train_set/train` (local-only; `dataset/` gitignored). Without it, `python run.py split` fails: `dataset/train not found`. OOD reference sets at `dataset/augmentations/{color_jitter,random_rotation}/`.

## Do-Not-Repeat

> TEAM CONVENTION: log REAL bug fixes here (shared + union-merged → all 4 get them, no conflicts). Do NOT rely on .wolf/buglog.json — machine-LOCAL JSON, can't union-merge, auto-fills with edit noise. One line per fix: `[date] symptom → fix`.

## Decision Log

- **2026-06-25 — No EDA stage, no `eda.py`** (deleted; run.py has no cmd_eda). Mechanical sanity (class balance, folder name == labels.py, raises on missing/renamed/empty) enforced inside `split_data.build_splits`. Normalization uses fixed ImageNet mean/std (nothing to fit). Keep only a one-time ~2-min manual glance at a few train images + `augmentations/` folder to guide aug choices; focused inspection continues in `error_analysis.py`. Methodology chain: data partition → naive baseline → model selection → error analysis.

- **2026-06-25 — Augmentation = OFFLINE ONLY.** Training feeds originals + offline twins through DETERMINISTIC transform (`build_eval_transform`: Resize256→CenterCrop224→Normalize) — NO online random editing. All variety from `make_augmented` twins. data.py:82 uses build_eval_transform when include_augmented=True. Active filters: **rotate, grayscale, invert, color_jitter, salt_pepper, blur** (mean-fill rotation). **shift+zoom REMOVED** — translate/crop pushes labeled object out of frame → label-corrupting. `build_train`/`build_light_train` kept only for the include_augmented=False fallback (effectively dead).

- **2026-06-25 — Gated curriculum training (NOT auto-progressive).** `run.py train --stage N`: stage1=P0(40%)→testP1, stage2=P0+P1(55%)→testP2, etc. Each run = FRESH random init, train to convergence (engine early-stop), test next unseen 15% block (clean originals), print per-class acc, STOP. Advancing = manual human call (team iterates arch via error analysis between stages; warm-start impossible across arch changes). "Good enough" to advance: clean-Dev ~0.55–0.65, must beat naive floor **0.2230** (measured 2026-06-25; old 0.3453 stale). 0.85 unreachable from-scratch (heavy twin distortion lowers clean acc). Real gate = error analysis shows no systematic shortcut.

- **2026-06-25 — Architecture (Person A).** model.py = small **ResNet-style**: Stem(7×7 s2 + maxpool) → 4 residual stages [64,128,256,512] blocks (2,2,2,2) → GAP → Dropout(0.3) → Linear(512,20). ~11.2M params, self-contained (torch/nn only), forward→[B,20]. Shape math 224→56→56→28→14→7→GAP→20. **BatchNorm chosen** (over GroupNorm/none) despite syllabus-edge — defense: "normalization layer like LayerNorm after Transformer attention, adapted to conv channels." BN buffers ride in state_dict; predict.py `.eval()` (line 58) → inference uses running stats, batch-size-safe. GroupNorm = lower-risk fallback if challenged. WHY rewrite: no-norm deep-from-scratch CNN hard to optimize + under-parameterized.

- **2026-06-25 — Engine defaults (Person B / engine.py), if SGD path used:** lr 0.05, weight_decay 5e-4, lr_step_size auto = epochs//3, grad_clip 1.0, patience 5, label_smoothing 0.1, momentum 0.9. KEEP SGD+momentum (NOT Adam — out of syllabus); momentum = smoothed-update extension of SGD. label_smoothing + StepLR FLAGGED as syllabus-edge. Public signatures STABLE (all call sites use kwargs). MPS-nondeterminism caveat KEPT (trust saved best-weights checkpoint, not rerun-determinism).

- **2026-06-26 — FINAL SUBMITTED model.** ResNet-18 + Squeeze-Excitation (~11.28M, BatchNorm, dropout 0.4), **AdamW lr1e-3 wd1e-2 + cosine + label-smoothing 0.1 + grad-clip 1.0 + early-stop**. Result: held-out clean **0.847, SCORE 0.906** (OOD-mean ~0.96 > clean). Robustness via OFFLINE weighted twins (rotate4/color4/gray2/sp2/blur2/invert1, mean-fill, no bg-swap). Weak class: acoustic_guitar vs laptop/mobile_phone. Submitted `weights.joblib` sha=fe20eb7d…, byte-identical to `submissions/my_team/weights.joblib` (the offline-twin gated model, train_stage4.csv Jun25 22:17).

- **2026-06-26 — Submission packaging FIX + lesson.** Shipped train.py must be the script that ACTUALLY made the weights, not a fresh simplification. Initial shipped train.py (Jun26 09:33) was a collapsed ONLINE-torchvision-aug + random-15%-split rewrite that did NOT train the weights and contradicted README. FIX: rewrote submission train.py as self-contained OFFLINE-twin script (inlined PIL filters rotate/color/gray/sp/blur/invert weights 4/4/2/2/2/1, mean-fill, materialize_twins, stratified split, AdamW/cosine/ls0.1/clip/early-stop) → now matches README + weights. Cleaned README (dropped engine.py + SGD-selectable mentions). weights.joblib UNCHANGED.