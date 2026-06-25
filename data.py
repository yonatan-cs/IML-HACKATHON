"""
data.py  —  OWNER: Person D (Data)

Turns the split manifest (dataset/splits.json) into PyTorch Datasets / DataLoaders.

Key design: we load from the manifest (a list of (path,label)) rather than copying files
around. One `ManifestDataset` works for train OR eval — you just pass a different transform.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from augment import build_train_transform, build_light_train_transform, build_eval_transform
from split_data import PROJECT_ROOT, load_splits
from make_augmented import AUG_DIR


class ManifestDataset(Dataset):
    """
    Dataset over a set of split partitions.

    Args:
        partitions: e.g. ["P0","P1"] — their images are concatenated.
        transform:  an augment.py pipeline (train = aug, eval = deterministic).
        include_augmented: if True, ALSO add each image's offline augmented twins from
            dataset/train_aug/ (created by make_augmented.py). The twin inherits its
            original's partition, so this is leakage-safe: ONLY pass include_augmented=True
            for TRAIN partitions, never for the eval/test block.
    Yields (image_tensor, label). `self.samples` keeps (abs_path, label) in order so
    error_analysis can map predictions back to files (use shuffle=False for eval).
    """

    def __init__(self, partitions: list[str], transform, include_augmented: bool = False):
        splits = load_splits()
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for p in partitions:
            for rel, label in splits["partitions"][p]:
                path = PROJECT_ROOT / rel
                self.samples.append((path, int(label)))
                if include_augmented:
                    # twins live at dataset/train_aug/<class>/<stem>__aug*.jpg
                    twin_dir = AUG_DIR / path.parent.name
                    for twin in sorted(twin_dir.glob(f"{path.stem}__aug*.jpg")):
                        self.samples.append((twin, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_loaders(
    train_partitions: list[str],
    eval_partition: str | None = None,
    *,
    img_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    include_augmented: bool = True,
):
    """
    Build (train_loader, eval_loader). eval_loader is None if eval_partition is None.

    include_augmented=True (default) pulls in the offline augmented twins (dataset/train_aug/)
    for the TRAIN set and uses a LIGHT online transform (the heavy filters are already baked
    into the twins). If train_aug/ doesn't exist yet, no twins are added — training still
    works on originals with the light transform. Run `python run.py materialize` first to
    generate the twins. The eval set is ALWAYS clean originals (no twins, no leakage).

    ⚠️ Windows: DataLoader workers use 'spawn', so any script that calls this MUST be
    guarded by `if __name__ == "__main__":` (run.py / train.py already are). If you hit
    worker errors on Windows, set num_workers=0.
    """
    train_tf = build_light_train_transform(img_size) if include_augmented else build_train_transform(img_size)
    train_ds = ManifestDataset(train_partitions, train_tf, include_augmented=include_augmented)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )

    eval_loader = None
    if eval_partition is not None:
        eval_ds = ManifestDataset([eval_partition], build_eval_transform(img_size))  # clean only
        eval_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False,   # shuffle=False → maps to paths
            num_workers=num_workers, pin_memory=False,
        )
    return train_loader, eval_loader


if __name__ == "__main__":
    # Self-check: needs dataset/splits.json (run `python run.py split` first).
    try:
        tl, el = build_loaders(["P0"], "P1", batch_size=8, num_workers=0)
        xb, yb = next(iter(tl))
        assert xb.shape[1:] == (3, 224, 224), xb.shape
        print("data OK — train batch", tuple(xb.shape), "| eval samples", len(el.dataset))
    except FileNotFoundError as e:
        print("(skipped self-check)", e)
