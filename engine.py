"""
engine.py  —  OWNER: Person B (Training / optimization)

Device selection, seeding, and the train/eval loops. Everything here is plain PyTorch
with fuller-than-usual comments (team is new to the API). It already works; Person B's
job is to make training CONVERGE BETTER (see the TODOs in `train`).

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

    ⚠️ Caveat from the tutorial: GPU/MPS kernels are NOT fully deterministic even with a
    fixed seed — two runs can differ. So we always SAVE the best weights file rather than
    relying on "rerun and get the same number". Don't panic over tiny run-to-run deltas.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── one epoch ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None) -> float:
    """
    Run one pass over `loader`, update weights, return the mean training loss.

    `grad_clip` (max gradient norm): if set, clips the gradient norm AFTER backward() and
    BEFORE step(). This caps how big any single update can be, which keeps SGD stable when
    the LR is high (0.01) — early in training gradients can spike and blow the weights up.
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
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    lr_step_size: int = 5,
    lr_gamma: float = 0.1,
    grad_clip: float | None = 1.0,
    label_smoothing: float = 0.1,
    device: torch.device | None = None,
    patience: int = 5,
    log_csv: str | Path | None = None,
):
    """
    Train `model` and return (model, history). Keeps the best-val-accuracy weights.

    Optimizer = SGD with momentum (generalizes better than Adam on a from-scratch CNN and
    is course-explainable). LR schedule = StepLR: every `lr_step_size` epochs the LR is
    multiplied by `lr_gamma` (step decay) — learn fast at a high LR early, then fine-tune at
    a low LR. With the defaults (lr 0.01, step 5, gamma 0.1) the LR is 0.01 for epochs 1-5,
    0.001 for 6-10, 0.0001 for 11-15.

    Defaults below are a sane starting point. Person B: tune them.

    TODO(Person B):
      - SGD lr lives in ~[0.003, 0.1]; momentum 0.9 is standard. weight_decay ~[1e-4, 5e-4].
      - StepLR knobs: `lr_step_size` (how often to drop) and `lr_gamma` (drop factor, e.g. 0.1).
        Rule of thumb: step_size ≈ epochs / 3 so you get ~2 drops over the run.
      - `grad_clip` (max grad norm, default 1.0): caps update size so high-LR SGD stays stable.
        Raise toward 5.0 if it's clipping so hard that learning stalls; set None to disable.
      - `label_smoothing` (default 0.1): softens targets to curb over-confidence. Try 0.0–0.2;
        set 0.0 to train on hard labels.
      - early stopping `patience`, label smoothing in the loss, gradient clipping
      - (advanced) mixed precision via torch.cuda.amp on CUDA machines
    """
    device = device or get_device()
    model = model.to(device)

    # label_smoothing softens the targets (e.g. 0.9 for the true class, 0.1 spread over the
    # rest) instead of a hard 1/0. This curbs over-confidence → better generalization and a
    # bit more robustness (the model commits less hard to background/color shortcuts).
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)  # wants logits
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    # step decay: LR *= lr_gamma every lr_step_size epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

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
