# Cerebrum

> OpenWolf's learning memory. Updated automatically as the AI learns from interactions.
> Do not edit manually unless correcting an error.
> Last updated: 2026-06-24

## User Preferences

<!-- How the user likes things done. Code style, tools, patterns, communication. -->

- Claude must NEVER appear as a GitHub contributor — even when Claude runs the push. Commits author+committer = the human; no `Co-Authored-By`/Claude/Anthropic trailers anywhere. All work attributed to the human team ("הכל אנחנו").

## Key Learnings

- **Project:** IML-HACKATHON
- **This human user = Person B** (Training / optimization). Owns `engine.py` ONLY: optimizer,
  LR schedule, early-stop, device select, seeding/reproducibility, checkpointing, metrics.
  NOT augment.py/make_augmented.py (Person C), NOT model.py (A), NOT data/eval (D). When the
  user says "חלק B" they mean Person B's lane, not task section 2.2. The user does NOT have the
  dataset locally — a different machine runs training; user's job is to make engine.py excellent
  + correct + interview-defensible.

- **Course-coverage map (what's interview-defensible — verified via NotebookLM "מבוא ללמידת מכונה" 2026-06-25).**
  IN syllabus (safe to use + defend): conv+pooling (Max/Avg/Stride, lecture 8), **residual/skip
  connections** (`y=f(x)+x`, He 2016, taught in Transformers unit — mitigates vanishing gradients),
  Dropout (~50% typical), ReLU/softmax, **L2/Ridge = weight decay**, GAP/pooling-as-regularizer,
  bias-variance/overfitting, early-stopping (Patience), SGD + LR-schedule (`η_t=1/√t`), data aug.
  OUT of syllabus (interview risk — flag): **BatchNorm** (only LayerNorm taught, in Transformers:
  `LayerNorm(x)=x−mean` → BN defensible only BY ANALOGY), **momentum/Adam** (engine uses SGD+momentum
  — Person B's risk), advanced init Kaiming/Xavier/He (only "random init" taught).
- **User took role A (Architecture / model.py) for the 2026-06-25 session** (was recorded as Person B
  / engine.py). When the user says "תופס פיקוד על A" they mean model.py is theirs now.
- **Dataset layout gotcha:** the downloaded raw data lives at `dataset/train_set/train/<class>/`
  (20 classes × 1000 jpg), but the pipeline code (`split_data.py` + `make_augmented.py`) reads
  from `dataset/train/`. Bridge with a symlink `dataset/train -> train_set/train` (local-only;
  `dataset/` is gitignored). Without it, `python run.py split` fails with
  `dataset/train not found — download train_set first`. Provided OOD reference sets are at
  `dataset/augmentations/{color_jitter,random_rotation}/`.

## Do-Not-Repeat

<!-- Mistakes made and corrected. Each entry prevents the same mistake recurring. -->
<!-- Format: [YYYY-MM-DD] Description of what went wrong and what to do instead. -->

> TEAM CONVENTION: log REAL bug fixes here (this file is shared + union-merged → all 4 get
> them, no conflicts). Do NOT rely on .wolf/buglog.json — it's machine-LOCAL, JSON (can't
> union-merge), and auto-fills with edit noise. One line per fix: `[date] symptom → fix`.

## Decision Log

<!-- Significant technical decisions with rationale. Why X was chosen over Y. -->

- 2026-06-25 — Dropped the preprocessing-stats step and the standalone EDA pass (deleted `eda.py`,
  removed the `eda` subcommand from `run.py`, trimmed SETUP.md/plan.md). Rationale: normalization
  uses fixed ImageNet mean/std constants (nothing to fit), and `eda.py`'s useful parts are already
  covered — `split_data.build_splits` enforces class balance + label/folder sanity (raises on any
  missing/renamed/empty class folder), and visual inspection happens during error analysis
  (`error_analysis.py`). Net: methodology chain is now data partitioning → naive baseline →
  model selection → error analysis.
- [2026-06-25] EDA decision settled: NO standalone EDA stage and NO eda.py (already deleted; run.py has no cmd_eda). Mechanical sanity (class balance, folder name == labels.py) is enforced inside split_data.build_splits. Keep only a one-time ~2-min manual glance at a few train images + the provided augmentations/ folder (Finder, no tooling) to guide augmentation choices; focused inspection continues in error_analysis. (Corrects an earlier note in this session that wrongly said eda.py still exists.)
- [2026-06-25] Augmentation strategy = OFFLINE ONLY. Training feeds originals + offline twins
  through the DETERMINISTIC transform (build_eval_transform: Resize256→CenterCrop224→Normalize)
  — NO online random editing. Changed data.py:82 to use build_eval_transform when
  include_augmented=True (was build_light_train_transform). All variety comes from make_augmented
  twins. Rationale: team wants originals used "as is"; on-the-fly edits unwanted. build_train/
  build_light_train kept only for the include_augmented=False fallback (twins are always
  materialized on the single training laptop, so that path is effectively dead).
- [2026-06-25] Dropped the `zoom` filter from make_augmented.FILTERS (its 60–90% random crop can
  push the labeled object out of frame → label-corrupting). f_zoom function kept but never
  selected. Twins must be regenerated (`python run.py materialize`) since prior twins included zoom.

## Decision Log — gated curriculum (added this session)
- Training workflow = GATED single-stage, NOT auto-progressive. `run.py train --stage N`:
  stage1=P0(40%)→testP1, stage2=P0+P1(55%)→testP2, etc. Each run = FRESH random init
  ("forget previous models"), trains to convergence (engine early-stop), tests next unseen
  15% block (clean originals), prints per-class acc, STOPS. Advancing = manual human call.
  WHY: team iterates architecture via error analysis between stages; warm-start impossible
  across arch changes; only expand data when current block is "good".
- "Good enough" to advance: clean-Dev ~0.55-0.65, must beat naive floor 0.2230 (measured 2026-06-25; the old 0.3453 was stale). 0.85 is
  unreachable from-scratch here (heavy twin distortion lowers clean acc). Real gate =
  error analysis shows no systematic shortcut.
- Aug filters (make_augmented.py offline twins, the ACTIVE path): rotate, grayscale, invert,
  color_jitter, salt_pepper, blur. shift+zoom REMOVED (translate/crop object out of frame).

## Decision Log — architecture rewrite (2026-06-25, Person A)

- model.py REWRITTEN: plain 4-block VGG (no norm, no skips, ~1.2M params) → small **ResNet-style**
  net. Stem(7x7 s2 + maxpool) → 4 residual stages [64,128,256,512], blocks (2,2,2,2) → GAP →
  Dropout(0.3) → Linear(512,20). ~11.2M params. Still self-contained (torch/nn only), forward→[B,20].
- WHY weak before: deep-from-scratch CNN with NO normalization is hard to optimize → plateaus; also
  under-parameterized for 20-class ImageNet. Fix = normalization (easier optimization) + residuals
  (depth without vanishing gradients) + more width (lower bias).
- **BatchNorm chosen** (user decision) over GroupNorm/none despite being syllabus-edge. Interview
  defense = "normalization layer, like the LayerNorm we saw after each Transformer attention block,
  adapted to conv channels." BN buffers ride in state_dict; predict.py calls `.eval()` (line 58) so
  inference uses running stats → batch-size-safe. GroupNorm was the lower-risk alternative if asked.
- Verified syntax+compile + shape math (224→56→56→28→14→7→GAP→20) on this machine. Runtime smoke
  test (`python model.py`) NOT run here — no torch installed locally; run on a teammate machine/Colab.

## Decision Log — training/optimization tuning (2026-06-25, Person B / engine.py)

- engine.py `train` defaults retuned now that model.py has BatchNorm: **lr 0.01→0.05** (BN
  conditions the loss surface → SGD can take larger/cleaner steps; a no-norm net would diverge
  at 0.05), **weight_decay 1e-4→5e-4** (L2/Ridge, pairs with higher LR), **lr_step_size 5→None**
  which auto-derives `epochs//3` so the "~2 LR drops over the run" rule holds even when a caller
  changes `epochs` (baseline_naive passes epochs= but not lr_step_size). grad_clip 1.0 kept as the
  high-LR safety net; patience 5 and label_smoothing 0.1 kept.
- KEEP SGD + momentum 0.9; do NOT switch to Adam (out of syllabus). momentum defended as a standard
  smoothed-update extension of SGD (velocity = decaying avg of past grads). label_smoothing FLAGGED
  as syllabus-edge (soft-target variant of course-taught cross-entropy). StepLR FLAGGED (LR
  scheduling is taught; specific step form is the extension). Kaiming init lives in model.py (A).
- Public signatures kept STABLE: get_device/set_seed/train/evaluate unchanged; all call sites
  (train.py, baseline_naive.py, robust_eval.py, run.py) use keyword args → tuning defaults is safe.
- set_seed now documents that torch.manual_seed also seeds the MPS generator; MPS-nondeterminism
  caveat KEPT (we trust the saved best-weights checkpoint, not rerun-determinism).
