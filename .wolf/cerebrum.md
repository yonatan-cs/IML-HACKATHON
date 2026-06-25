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
