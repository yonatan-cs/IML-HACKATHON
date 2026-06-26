"""
train.py  —  SELF-CONTAINED training script (IML 67577, Challenge 2).

Trains `ModelArchitecture` (model.py, same folder) FROM RANDOM INIT and saves the CPU weights
to weights.joblib via joblib — the exact artifact predict.py loads. The grader never runs this
file (it only loads weights.joblib through predict.py); we ship it so the pipeline that produced
the submitted model is documented and runnable on its own. Imports ONLY torch / torchvision /
stdlib / joblib / numpy / PIL and the local model.py — no other project files.

Recipe that produced the submitted model: a small ResNet + Squeeze-and-Excitation network,
trained with AdamW + cosine LR + label smoothing + gradient clipping + early stopping, on the
ORIGINAL training images PLUS pre-baked OFFLINE augmented "twins" of each training image. The
twins are the robustness mechanism (50% of the grade is held-out manipulations): each twin
applies a weighted-random subset of label-preserving filters — mean-fill rotation, colour
jitter, grayscale, salt-pepper, blur, inversion — so the network is forced to be invariant to
background / lighting / colour / orientation shifts it never sees a clean copy of.

Why OFFLINE twins (data duplication) rather than on-the-fly augmentation: the twins are baked
to disk once, seeded per file, so they are reproducible byte-for-byte across machines and can
be opened and inspected during error analysis. Originals pass through unchanged. A twin is only
ever added to the TRAIN split (never validation) -> no leakage.

Run (from a folder where the dataset is reachable):
    python train.py --data /path/to/dataset/train --img-size 224 --epochs 25

Reproducibility note: MPS/CUDA kernels are not bit-deterministic, so a rerun does NOT reproduce
the exact weights — that is why we keep the actual trained weights.joblib rather than relying on
"rerun and hope".
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image, ImageOps, ImageEnhance, ImageStat, ImageFilter

from model import ModelArchitecture   # local, same folder — self-contained

# Local class index 0..19 = the 20 HF ImageNet classes sorted by HF index (matches
# dataset/labels.json). A training subfolder's name maps to its position here.
CLASSES = [
    "goldfish", "bald_eagle", "toucan", "jellyfish", "tiger", "african_elephant",
    "acoustic_guitar", "airliner", "balloon", "lighthouse", "castle", "mobile_phone",
    "container_ship", "french_horn", "laptop", "sports_car", "mushroom", "lemon",
    "pizza", "daisy",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── offline augmentation filters (PIL -> PIL, RGB in/out) ──────────────────────
# Each twin applies a weighted-random 1..3-subset of these. Rotation and colour jitter carry
# the highest weight because the provided augmentations/ folder (random_rotation, color_jitter)
# confirms those are real held-out test axes. Inversion is kept rare — it is label-risky
# (a yellow lemon becomes blue). Rotation fills the empty corners with the image's OWN mean
# colour, not black: black corners would be a fake "background" cue the model could cheat on.

def _mean_color(img: Image.Image) -> tuple[int, int, int]:
    m = ImageStat.Stat(img).mean
    return tuple(int(round(c)) for c in m[:3])


def f_rotate(img, rng):
    angle = rng.choice([45, 90, 180, 270])
    return img.rotate(angle, fillcolor=_mean_color(img), expand=False)


def f_color_jitter(img, rng):
    for Enh in (ImageEnhance.Color, ImageEnhance.Brightness, ImageEnhance.Contrast):
        img = Enh(img).enhance(rng.uniform(0.5, 1.5))
    return img


def f_grayscale(img, rng):
    return ImageOps.grayscale(img).convert("RGB")


def f_salt_pepper(img, rng, amount: float = 0.02):
    arr = np.array(img)
    r = np.random.default_rng(rng.randint(0, 2 ** 31 - 1)).random(arr.shape[:2])
    arr[r < amount / 2] = 0           # pepper
    arr[r > 1 - amount / 2] = 255     # salt
    return Image.fromarray(arr)


def f_blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.5)))


def f_invert(img, rng):
    return ImageOps.invert(img)


# (name, fn, weight). Canonical order = geometric -> photometric/noise; chosen filters are
# applied in this order regardless of pick order.
FILTERS = [
    ("rotate",       f_rotate,       4),
    ("color_jitter", f_color_jitter, 4),
    ("grayscale",    f_grayscale,    2),
    ("salt_pepper",  f_salt_pepper,  2),
    ("blur",         f_blur,         2),
    ("invert",       f_invert,       1),
]
MAX_FILTERS_PER_IMAGE = 3


def _weighted_sample_without_replacement(rng, idxs, weights, k):
    idxs, weights = list(idxs), list(weights)
    out = []
    for _ in range(min(k, len(idxs))):
        total = sum(weights)
        r = rng.uniform(0, total)
        acc = 0.0
        for j, w in enumerate(weights):
            acc += w
            if r <= acc:
                out.append(idxs.pop(j))
                weights.pop(j)
                break
    return out


def apply_random_filters(img: Image.Image, rng: random.Random) -> Image.Image:
    """Pick 1..MAX weighted filters without replacement, apply them in canonical order."""
    k = rng.randint(1, MAX_FILTERS_PER_IMAGE)
    chosen = _weighted_sample_without_replacement(
        rng, range(len(FILTERS)), [w for _, _, w in FILTERS], k)
    out = img
    for i in sorted(chosen):
        out = FILTERS[i][1](out, rng)
    return out.convert("RGB")


# ── data ───────────────────────────────────────────────────────────────────────

def list_samples(root: str | Path) -> list[tuple[Path, int]]:
    """Collect (image_path, label) from <root>/<class_name>/*.{jpg,jpeg,png}."""
    root = Path(root)
    idx = {c: i for i, c in enumerate(CLASSES)}
    samples: list[tuple[Path, int]] = []
    for c in CLASSES:
        d = root / c
        for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png"):
            for p in sorted(d.glob(ext)):
                samples.append((p, idx[c]))
    if not samples:
        raise FileNotFoundError(f"No images under {root} (expected <class_name>/*.jpg). "
                                f"Pass --data pointing at the dataset/train folder.")
    return samples


def stratified_split(samples, val_frac, seed):
    """Per-class hold-out so each split mirrors the full class distribution (no leakage)."""
    by_cls: dict[int, list] = {}
    for p, y in samples:
        by_cls.setdefault(y, []).append((p, y))
    rng = random.Random(seed)
    train, val = [], []
    for y in sorted(by_cls):
        items = sorted(by_cls[y], key=lambda t: t[0].name)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_frac))
        val += items[:n_val]
        train += items[n_val:]
    return train, val


def materialize_twins(train_orig, aug_root, copies, seed):
    """Pre-bake `copies` augmented twins per TRAIN original to disk (seeded per file → identical
    across machines, inspectable). Skips files already written. Returns (twin_path, label) list.
    Twins are built ONLY for the train split, so they can never leak into validation."""
    aug_root = Path(aug_root)
    out, made = [], 0
    for p, y in train_orig:
        dst_dir = aug_root / p.parent.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for i in range(copies):
            dst = dst_dir / f"{p.stem}__aug{i}.jpg"
            if not dst.exists():
                rng = random.Random(f"{seed}:{p.name}:{i}")
                apply_random_filters(Image.open(p).convert("RGB"), rng).save(dst, quality=90)
                made += 1
            out.append((dst, y))
    print(f"twins ready: {len(out)} total ({made} newly written) -> {aug_root}")
    return out


class ImgDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert("RGB")), y


def base_transform(img_size: int) -> v2.Compose:
    """Deterministic: Resize 256 -> CenterCrop 224 -> ImageNet normalize. Mirrors the grader.
    Used for BOTH originals and twins — all augmentation variety is already baked into the twins,
    so we do NOT re-augment online (that would compound the distortion)."""
    resize = int(round(img_size * 256 / 224))
    return v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def val_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/train", help="path to dataset/train (<class>/*.jpg)")
    ap.add_argument("--aug-dir", default="dataset/train_aug", help="where offline twins are written")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--copies", type=int, default=2, help="augmented twins per training image")
    ap.add_argument("--patience", type=int, default=5)
    a = ap.parse_args()

    set_seed(42)
    device = get_device()
    print(f"device={device}  img_size={a.img_size}  batch={a.batch}  epochs={a.epochs}")

    # Stratified, seeded split at the ORIGINAL-image level (val = clean originals, for early stop).
    originals = list_samples(a.data)
    train_orig, val_orig = stratified_split(originals, a.val_frac, seed=42)

    # Robustness mechanism: pre-bake offline twins for the TRAIN originals, add them to train.
    twins = materialize_twins(train_orig, a.aug_dir, copies=a.copies, seed=42)
    train_samples = train_orig + twins

    tf = base_transform(a.img_size)                      # deterministic, for originals + twins
    train_ds = ImgDataset(train_samples, tf)
    val_ds = ImgDataset(val_orig, tf)                    # clean originals only — no twins, no leakage
    print(f"train={len(train_ds)} ({len(train_orig)} orig + {len(twins)} twins)  "
          f"val={len(val_ds)}  classes={len(CLASSES)}")

    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                              num_workers=a.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = ModelArchitecture().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs, eta_min=1e-5)

    best_acc, best_state, since_improved = -1.0, None, 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        acc = val_accuracy(model, val_loader, device)
        print(f"epoch {epoch:2d}/{a.epochs}  loss {running / max(n, 1):.4f}  val_acc {acc:.4f}")

        if acc > best_acc:                       # keep the best-val checkpoint (not the last)
            best_acc, since_improved = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improved += 1
            if since_improved >= a.patience:     # early stopping
                print(f"early stop at epoch {epoch} (no val gain for {a.patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Grader contract: CPU state dict saved via joblib to exactly weights.joblib.
    joblib.dump(model.cpu().state_dict(), "weights.joblib")
    print(f"saved weights.joblib  (best val_acc {best_acc:.4f})")


if __name__ == "__main__":
    main()
