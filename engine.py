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

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """Run one pass over `loader`, update weights, return the mean training loss."""
    model.train()                       # enable dropout
    running = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)               # [B, 20]
        loss = criterion(logits, y)
        loss.backward()                 # backprop
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
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    patience: int = 5,
    log_csv: str | Path | None = None,
):
    """
    Train `model` and return (model, history). Keeps the best-val-accuracy weights.

    Defaults below are a sane starting point. Person B: tune them.

    TODO(Person B):
      - optimizer: AdamW (here) vs SGD(momentum=0.9, nesterov) — SGD often generalizes better
      - lr schedule: cosine (here) vs OneCycleLR vs StepLR; try lr in [3e-4, 3e-3]
      - early stopping `patience`, label smoothing in the loss, gradient clipping
      - (advanced) mixed precision via torch.cuda.amp on CUDA machines
    """
    device = device or get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()             # standard multi-class loss (wants logits)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_acc, best_state, since_improved = -1.0, None, 0

    writer = None
    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        writer = csv.writer(open(log_csv, "w", newline=""))
        writer.writerow(["epoch", "train_loss", "val_acc", "lr"])

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
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
    # Self-check: tiny fake data trains for 1 epoch on the chosen device.
    from torch.utils.data import TensorDataset
    set_seed(0)
    dev = get_device()
    print("device:", dev)
    X = torch.randn(32, 3, 32, 32)
    Y = torch.randint(0, 20, (32,))
    net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 20))
    loader = DataLoader(TensorDataset(X, Y), batch_size=8)
    _, hist = train(net, loader, loader, epochs=1, device=dev)
    assert len(hist) == 1
    print("engine OK")
