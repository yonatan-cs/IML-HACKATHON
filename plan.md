IML Challenge 2 — Robust Image Classification: Team Build Plan
Context
4-person team, ~6-7 hours of total active work, deadline Friday 11:00. Goal: a from-scratch CNN that classifies 20 ImageNet-subset classes and stays accurate under held-out visual manipulations (background/lighting/color). Graded test = 50% clean + 50% out-of-distribution (OOD) augmented images.

Everyone owns real, score-moving work (not a trivial pipeline split). Claude builds a runnable skeleton; each person fills one high-leverage lever, measured on a shared harness, in separate files (clean git merges). Compute: M3 Mac (MPS), Windows 32GB (CPU), no NVIDIA → device-agnostic code, small model.

This plan follows the course "ML in Practice" methodology (data partitioning → preprocessing iron-rule → EDA → naive baseline → model selection → error analysis), not just "train a net." The interview will probe this — methodology is part of the grade.

Project Vision (AI-readable — a fresh session reads this and is fully oriented)
Product: PixelPerfect robust classifier. Input [B,3,224,224] ImageNet-normalized → output class indices [B] in 0..19.

Two equally-weighted objectives: (1) clean accuracy, (2) robustness — prediction must not change under label-preserving manipulations (background/lighting/color/blur). Test augmentations are hidden: no overfitting to one kind; need broad label-preserving train-time augmentation to force invariance.

Hard grader constraints (violation = zero/DQ) — authoritative copy in CLAUDE.md:

Train from scratch; no pretrained weights, no external data, no network at train/load/eval.
submissions/<team>/model.py defines class exactly ModelArchitecture (nn.Module), forward→logits [B,20], fully self-contained (imports only torch/stdlib). The grader reconstructs the architecture from this file alone.
train.py saves weights.joblib = joblib.dump(model.cpu().state_dict(), ...) (base_model docstring says torch.save — wrong; grader uses joblib.load).
predict.py frozen — Model(BaseModel), returns argmax indices.
Labels: labels.py (local idx 0..19 = HF classes sorted by HF index) = source of truth; folder names must match HF_INDEX_TO_NAME.
Data: dataset/train/<class>/*.jpg, 1000/class, 20 classes = 20k. Downloaded separately (not committed). A course augmentations/ folder exists — contents unknown until downloaded.

Submission: challenge2_<ids>/ = ONLY train.py, model.py, predict.py, weights.joblib, README. README: line1 names (comma,no-space), line2 IDs, blank line, then model + manipulation write-up. Short interview defends choices → keep explainable.

Methodology — the iron rules (baked into every stage)
These come from the tutorial and are non-negotiable; the harness enforces them so no one accidentally cheats the methodology:

Three-way split: Train / Dev(Validation) / Test. Built once, seeded, stratified per class so each split faithfully represents the whole (no class leakage). Dev = all tuning + error analysis. Local Test = touched ONCE, at the very end, to estimate true generalization — never tuned against (every extra look overfits it and destroys it as a measure). The hidden grader test is the ultimate one-shot test.
Preprocessing iron rule — fit on Train only. Any data-driven decision (normalization stats, class weights, augmentation strength) is decided on Train alone. Dev/Test never inform preprocessing → no contamination. (We use fixed ImageNet mean/std = constants, not fitted → already compliant; if anyone computes dataset stats, Train only.)
EDA before modeling. Physically look at images: backgrounds, lighting, label/folder sanity, class balance. Drives which manipulations matter (Person C) and catches bad data.
Naive baseline before the deep net. A simple model (softmax/logistic regression on downsampled pixels) sets the floor the CNN must beat. If the CNN can't beat it, something is broken.
Error analysis on Dev, manually. Pull misclassified Dev samples, eyeball them, find recurring patterns (e.g. "sky-background ships → airliner"). Feed patterns back into augmentation/architecture. Never error-analyze on Test.
Reproducibility caveat. Seed torch/numpy/random, but MPS/CUDA can give different results with the same seed — so we keep the actual best weights file, not "rerun and hope." Record the config that produced each weights.joblib.
Technical Decisions (plain language + cost-to-undo flags)
Model = hand-built plain CNN (VGG-style), self-contained in model.py. conv→BN→ReLU blocks + maxpool → AdaptiveAvgPool2d(1) → Linear(…,20). Adaptive pool accepts any input (train small, eval 224). Easiest to explain, fastest to train. 🚩 Hard-to-undo: model.py must never import a local helper — grader breaks otherwise.
Train ≈128px, evaluate 224px. ~3× faster on MPS/CPU; adaptive pool absorbs the gap. 🚩 mild train/eval gap — revisit only if clean accuracy underperforms.
Augmentation = torchvision.transforms.v2 only. Already a dep, zero network. No albumentations (extra dep), no wandb (phones home → breaks no-network rule).
Dev code at repo root; submission stays flat. Helpers (data.py, augment.py, engine.py, split_data.py, eda.py, baseline_naive.py, robust_eval.py, error_analysis.py, run.py) sit at root next to base_model.py. Gradeable folder keeps only self-contained model.py + thin train.py + frozen predict.py. 🚩 Assembly cost: collapse pipeline into a self-contained train.py at the end (~30 min). Grader never runs train.py, so this is for honesty/interview.
weights.joblib git-ignored during dev. Binary, changes each run → merge conflicts + bloat. Each person trains locally; the leaderboard ranks; the single best is committed once. 🚩 add submissions/**/weights.joblib to .gitignore.
Logging = CSV per run (+ optional local TensorBoard). No external trackers (network).
Env: Python 3.11, requirements.txt (torch torchvision pillow joblib numpy tqdm). Windows DataLoader workers guarded by if __name__ == "__main__": (spawn).
Repo / Module Layout
repo root/
  base_model.py, labels.py, evaluate.py, check_submission.py   # provided (reuse)
  split_data.py        # NEW — 3-way stratified seeded split + OOD stress build   [Person D]
  eda.py               # NEW — sample grids, class balance, label sanity          [Person D]
  baseline_naive.py    # NEW — softmax-regression floor (pure torch)              [Person D]
  data.py              # NEW — datasets / transforms wiring / loaders             [Person D]
  augment.py           # NEW — train aug pipeline + OOD stress transforms         [Person C]
  engine.py            # NEW — train/eval loops, device, scheduler, seed, ckpt    [Person B]
  robust_eval.py       # NEW — clean-Dev vs OOD-Dev report                        [Person D]
  error_analysis.py    # NEW — dump misclassified Dev images + confusion          [Person D]
  run.py               # NEW — CLI glue: split|eda|baseline|train|eval|errors      [shared]
  requirements.txt     # NEW
  submissions/my_team/
    model.py           # ModelArchitecture — SELF-CONTAINED                       [Person A]
    train.py           # thin trainer → weights.joblib                            [shared]
    predict.py         # FROZEN
Separate files per owner ⇒ parallel work, near-zero merge conflicts.

4-Person Ownership (each lever moves the grade)
Person A — Architecture (model.py): own ModelArchitecture; start from the plain-CNN baseline, improve depth/width/regularization. Stays self-contained, [B,20] logits.
Person B — Training / optimization (engine.py): loop, optimizer, LR schedule, early stop, device auto-select, seeding/reproducibility, checkpointing, metrics.
Person C — Augmentation / robustness (augment.py): the differentiator — train-time manipulations + OOD stress transforms. Owns the 50% OOD score half. Guided by EDA + error analysis.
Person D — Data + methodology + evaluation (split_data.py, data.py, eda.py, baseline_naive.py, robust_eval.py, error_analysis.py): the 3-way stratified split, EDA, naive baseline floor, the shared leaderboard everyone trusts, error-analysis tooling, enforces Test-used-once, and selects/commits the final weights.joblib.
Shared glue (run.py, train.py) built by Claude, rarely edited → low conflict.

Build Plan — small, reviewable stages
Stage 0 — Scaffold (Claude, now, no training): create all NEW files runnable, each owner's lever marked # TODO(owner): … with full docstrings + PyTorch hints; add weights ignore. Deliverable: imports clean on Mac+Windows, check_submission.py my_team passes at dummy level. Review before training.

Stage 1 — Data + split + EDA (Person D): download train_set; run.py split → seeded stratified Train/Dev/Test (+ map Dev to dataset/validation/ for evaluate.py, Test to dataset/test/). run.py eda → class balance + sample grids + label sanity. Inspect the provided augmentations/ folder, report contents. Review: splits representative? data clean?

Stage 2 — Naive baseline (Person D): run.py baseline → softmax regression on downsampled pixels = the floor. Record clean-Dev accuracy. Review: floor number agreed.

Stage 3 — Clean CNN baseline (A+B): train plain CNN on Train, tune on Dev. Must beat the naive floor; aim 50%+ clean-Dev. First real weights.joblib. Review: loop works + beats floor.

Stage 4 — Robustness (C+D): Person C adds augmentation; Person D adds robust_eval (clean-Dev vs OOD-Dev). Retrain. Goal: hold clean accuracy while OOD accuracy climbs. Review: clean vs OOD before/after.

Stage 5 — Error analysis + iterate (all, parallel): run.py errors dumps misclassified Dev samples; team eyeballs for patterns → feeds A's architecture + C's augmentation. Person D's leaderboard picks each round's winner. Time-boxed. Test set stays untouched.

Stage 6 — Package & submit (Person D + Claude): now touch local Test once for the honest final number; assemble self-contained train.py; commit final weights.joblib; write README (names/IDs + manipulation write-up); rename challenge2_<ids>; run check_submission.py + evaluate.py last time.

Tools / connectors / skills
No external connectors/MCP in the pipeline — rules forbid network at train/load/eval. (Drive only as a manual way to pass the dataset between teammates, never called from code.)
TensorBoard (local) — optional loss/acc curves: pip install tensorboard + tensorboard --logdir runs.
Optional interactive demo (tutorial tip): a tiny script that runs predict on an arbitrary image so you can probe behavior on varied inputs (great for the interview).
Colab fallback if MPS/CPU too slow: upload repo+data to Drive, run run.py on a free T4 (same device-agnostic code; re-download data per session).
/code-review on each stage's diff before merge — cheap quality gate.
Git already set for low-conflict parallel work; git pull before git push.
Verification
Structural: python check_submission.py my_team → all [ OK ]; predict(x) returns [B] int tensor in 0..19.
Methodology: split is stratified + seeded (counts equal per class per split); naive baseline number recorded; CNN beats it; error analysis ran on Dev; Test read exactly once.
Functional: run.py train → weights.joblib; evaluate.py lists the team with a real accuracy.
Robustness: robust_eval.py prints clean-Dev vs OOD-Dev; gap shrinks after Stage 4.
Reproducibility: seeds set; the winning weights.joblib is saved/committed (not relied on via rerun, given MPS/CUDA nondeterminism).
Cross-platform: imports + 1-epoch smoke train on M3 (mps) and Windows (cpu), no edits.