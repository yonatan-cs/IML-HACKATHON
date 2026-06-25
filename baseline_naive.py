"""
baseline_naive.py  —  OWNER: Person D (Naive baseline)

Methodology step 4: a DUMB model first, to set the floor the CNN must beat. If the deep
net can't beat softmax-regression-on-tiny-pixels, something is broken (bug, bad labels,
broken loader) — not "the net needs tuning".

Model = one Linear layer on a downsampled flattened image = multinomial logistic
regression. Pure torch, trains in seconds. Trains on P0, tests on P1 (the first
progressive step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from data import ManifestDataset
from engine import get_device, set_seed, train, evaluate

SMALL = 32   # downsample to 32x32 → 3*32*32 = 3072 features


def _tiny_transform():
    """Resize to 32x32, to float tensor, flatten to a 3072-vector. No normalization needed."""
    return v2.Compose([
        v2.Resize((SMALL, SMALL), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda t: t.reshape(-1)),     # [3,32,32] -> [3072]
    ])


def main(epochs: int = 8):
    set_seed(42)
    device = get_device()

    tf = _tiny_transform()
    train_ds = ManifestDataset(["P0"], tf)
    test_ds = ManifestDataset(["P1"], tf)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = nn.Linear(3 * SMALL * SMALL, 20)     # logistic regression
    model, _ = train(model, train_loader, test_loader, epochs=epochs, lr=1e-3, device=device)

    acc, *_ = evaluate(model, test_loader, device)
    print(f"\nNAIVE BASELINE (logistic regression, {SMALL}x{SMALL}) clean-Dev acc = {acc:.4f}")
    print(f"  random guessing = {1/20:.4f}. The CNN must comfortably beat {acc:.4f}.")
    return acc


if __name__ == "__main__":
    main()
