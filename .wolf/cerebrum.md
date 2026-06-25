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
