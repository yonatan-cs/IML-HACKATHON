"""
engine.py  —  OWNER: Person B (Training / optimization)

Device selection, seeding, and the train/eval loops. Everything here is plain PyTorch
with fuller-than-usual comments (team is new to the API). It already works; Person B's
job is to make training CONVERGE BETTER (see the TODOs in `train`).

NOTE (team rule update, 2026-06-25): we are NO LONGER restricted to course-only methods.
We use best-practice techniques (AdamW, BatchNorm, cosine LR schedule, label smoothing) and
DOCUMENT each one — here in the code and in the team guide — so everyone knows exactly what
we run and why. Nothing is a black box.

Used by: run.py, submissions/my_team/train.py, baseline_naive.py, robust_eval.py.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── device & reproducibility ──────────────────────────────────────────────────

def get_device() -> torch.device:
    """Pick the fastest available backend. Same code runs on every teammate's laptop."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():     # Apple Silicon (M3)
        return torch.device("mps")
    return torch.device("cpu")                # Windows box without NVIDIA


def set_seed(seed: int = 42) -> None:
    """
    Seed Python / NumPy / Torch for reproducibility.

    We seed every RNG that feeds the run: Python's `random` (used by some samplers),
    NumPy (our metrics / any array shuffling), and Torch (weight init, dropout masks,
    DataLoader shuffling). `torch.manual_seed` also seeds the MPS (Apple Silicon)
    generator, so the same seed gives the same *starting point* on every laptop.

    ⚠️ Caveat: GPU/MPS kernels are NOT fully deterministic even with a fixed seed —
    floating-point reductions can run in a different order each time, so two runs can still
    differ by a fraction of a percent. That is why we always SAVE the best weights file
    (checkpointing) rather than relying on "rerun and get the exact same number". Don't panic
    over tiny run-to-run deltas.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                       # also seeds the MPS generator on M-series Macs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── one epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None) -> float:
    """
    Run one pass over `loader`, update weights, return the mean training loss.

    `grad_clip` (max gradient norm): if set, clips the gradient norm AFTER backward() and
    BEFORE step(). This caps how big any single update can be — a safety net that keeps the
    first few high-learning-rate steps from spiking and blowing the weights up.
    """
    model.train()                       # enable dropout
    running = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)               # [B, 20]
        loss = criterion(logits, y)
        loss.backward()                 # backprop
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()                # update weights
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Evaluate WITHOUT updating weights. Returns:
        acc    : float accuracy in [0,1]
        preds  : np.ndarray [N] predicted class indices
        labels : np.ndarray [N] true class indices
        confs  : np.ndarray [N] softmax confidence of the predicted class
    Order matches the loader (use shuffle=False) so error_analysis can map back to paths.
    """
    model.eval()                        # disable dropout
    all_preds, all_labels, all_confs = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.numpy())
        all_confs.append(conf.cpu().numpy())
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    confs = np.concatenate(all_confs) if all_confs else np.array([])
    acc = float((preds == labels).mean()) if len(preds) else 0.0
    return acc, preds, labels, confs


# ── full training loop ────────────────────────────────────────────────────────

def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 15,
    optimizer: str = "adamw",          # "adamw" (default, best convergence) or "sgd"
    lr: float | None = None,           # None ⇒ optimizer-aware default (adamw 1e-3, sgd 0.05)
    weight_decay: float | None = None, # None ⇒ optimizer-aware default (adamw 1e-2, sgd 5e-4)
    momentum: float = 0.9,             # SGD only
    lr_schedule: str = "cosine",       # "cosine" (default) or "step"
    lr_step_size: int | None = None,   # step schedule: None ⇒ auto = epochs // 3
    lr_gamma: float = 0.1,             # step schedule drop factor
    grad_clip: float | None = 1.0,
    label_smoothing: float = 0.1,
    device: torch.device | None = None,
    patience: int = 5,
    log_csv: str | Path | None = None,
):
    """
    Train `model` and return (model, history). Keeps the best-val-accuracy weights
    (checkpointing) and restores them at the end, so the returned model is the best one we
    saw on the validation set — not necessarily the last epoch.

    OPTIMIZER (choose via `optimizer=`):
      • "adamw" (DEFAULT): AdamW = Adam with *decoupled* weight decay. Adam adapts the step
        size per-parameter from running estimates of each gradient's mean and variance, so it
        converges fast and is forgiving about the learning rate — the strongest default for
        training this ResNet from scratch. The "W" means the weight-decay (L2) term is applied
        directly to the weights instead of folded into the gradient, which regularizes more
        correctly than classic Adam. Defaults: lr=1e-3, weight_decay=1e-2.
      • "sgd": plain stochastic gradient descent + momentum 0.9 (momentum keeps a decaying
        average of past gradients — "velocity" — and steps along it, damping mini-batch noise).
        SGD often generalizes a hair better (flatter minima), which can help the 50%
        out-of-distribution half, but needs a bigger LR and more epochs. Defaults: lr=0.05,
        weight_decay=5e-4. Set momentum=0 for textbook vanilla SGD.

    LEARNING RATE: lr=None auto-picks the optimizer's default (adamw 1e-3, sgd 0.05). BatchNorm
      in the model conditions the loss surface so both optimizers stay stable; `grad_clip`
      (max grad norm 1.0) is the safety net for the first few steps.

    LR SCHEDULE (`lr_schedule=`):
      • "cosine" (DEFAULT): CosineAnnealingLR smoothly decays the LR from its start value down
        to ~0 along a cosine curve over the run — fast learning early, gentle fine-tuning late,
        no manual step points to pick. Pairs naturally with AdamW.
      • "step": StepLR multiplies the LR by `lr_gamma` every `lr_step_size` epochs
        (None ⇒ epochs//3, ~2 drops over the run). The classic step-decay schedule.

    REGULARIZATION (all fighting overfitting — 11.2M params on only ~16k images):
      • weight_decay = L2 penalty pulling weights toward 0 (optimizer-aware default).
      • label_smoothing=0.1: replaces the hard 1-of-20 target with 0.9 for the true class and
        0.1 spread over the other 19. Curbs over-confidence → better calibration and a little
        more robustness (the net commits less hard to background/color shortcuts). 0.0 = hard.
      • Dropout (in the model head) + early stopping (below) round out the defenses.

    EARLY STOPPING (`patience`): stop if validation accuracy doesn't improve for `patience`
      consecutive epochs; the best-seen checkpoint is restored before returning.

    Defaults are a strong from-scratch recipe (AdamW + cosine + BatchNorm + label smoothing).
    TODO(Person B): tune per stage — adamw lr ~[5e-4, 3e-3] / wd ~[1e-2, 5e-2];
      sgd lr ~[0.01, 0.1] / wd ~[1e-4, 5e-4]; grad_clip up to 5.0; label_smoothing 0.0–0.2;
      try `optimizer="sgd"` near the end to see if it squeezes a bit more OOD generalization.
    """
    device = device or get_device()
    model = model.to(device)

    # Build the optimizer from the `optimizer=` string, filling optimizer-aware defaults for
    # lr / weight_decay when the caller left them None, then REBIND `optimizer` to the built
    # object so the training loop below uses it directly.
    opt_name = optimizer.lower()
    if opt_name == "adamw":
        lr = 1e-3 if lr is None else lr
        weight_decay = 1e-2 if weight_decay is None else weight_decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        lr = 0.05 if lr is None else lr
        weight_decay = 5e-4 if weight_decay is None else weight_decay
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
    else:
        raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {opt_name!r}")

    # CrossEntropyLoss takes raw logits (it applies log-softmax internally — do NOT softmax
    # first). label_smoothing softens the targets (0.9 true / 0.1 spread) — see docstring.
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # LR schedule, stepped once per epoch below. cosine = smooth decay to ~0; step = drop every k.
    if lr_schedule.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=lr * 0.01)
    elif lr_schedule.lower() == "step":
        step = lr_step_size if lr_step_size is not None else max(1, epochs // 3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=lr_gamma)
    else:
        raise ValueError(f"lr_schedule must be 'cosine' or 'step', got {lr_schedule!r}")

    history = []
    best_acc, best_state, since_improved = -1.0, None, 0

    writer = None
    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(open(log_csv, "w", newline=""))
        writer.writerow(["epoch", "train_loss", "val_acc", "lr"])

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_acc = -1.0
        if val_loader is not None:
            val_acc, *_ = evaluate(model, val_loader, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_acc": val_acc, "lr": cur_lr})
        if writer:
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{val_acc:.4f}", f"{cur_lr:.2e}"])
        print(f"  epoch {epoch:2d}/{epochs}  loss {tr_loss:.4f}  val_acc {val_acc:.4f}")

        # keep best + early stop (only meaningful when we have a val set)
        if val_loader is not None:
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since_improved = 0
            else:
                since_improved += 1
                if since_improved >= patience:
                    print(f"  early stop at epoch {epoch} (no val gain for {patience})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)         # restore best, not last
    return model, history


if __name__ == "__main__":
    # Self-check: tiny fake data trains for 1 epoch on the chosen device (both optimizers).
    from torch.utils.data import TensorDataset
    set_seed(0)
    dev = get_device()
    print("device:", dev)
    X = torch.randn(32, 3, 32, 32)
    Y = torch.randint(0, 20, (32,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=8)
    for opt in ("adamw", "sgd"):
        net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 20))
        _, hist = train(net, loader, loader, epochs=1, optimizer=opt, device=dev)
        assert len(hist) == 1
    print("engine OK (adamw + sgd)")
