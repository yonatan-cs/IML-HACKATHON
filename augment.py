"""
augment.py  —  OWNER: Person C (Augmentation / robustness)

THE differentiator of this challenge. The hidden test is 50% out-of-distribution
(background / lighting / color / rotation shifts). We can't see those exact transforms,
so we generate LOTS of varied data from the given 20k images and force the network to be
invariant to them.

Everything uses torchvision.transforms.v2 (already installed, no network). Two custom
transforms (mean-fill rotation, salt-&-pepper) are implemented below because torchvision
doesn't ship them exactly as the team asked.

Pipelines exported:
  build_train_transform(img_size)  -> heavy random augmentation (used for TRAINING)
  build_eval_transform(img_size)   -> deterministic, matches the grader (used for VAL/TEST)
  OOD_TRANSFORMS                    -> dict of single named manipulations (robustness probes)

All pipelines END with ToImage→float→Normalize(ImageNet) so the model gets a normalized
[3,H,W] tensor, exactly like predict.py expects.
"""
from __future__ import annotations

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ── two custom transforms the team asked for ──────────────────────────────────

class MeanFillRotation(v2.Transform):
    """
    Rotate by a random angle chosen from `angles`, filling the empty corners with the
    image's OWN mean color (not black — black would be a fake 'background' cue the model
    could cheat on).

    Person C TODO: tune `angles`. 90/180/270 are 'free' (no empty corners); 45 creates
    triangular fill regions — visually strong, test with error analysis whether it helps.
    """

    def __init__(self, angles=(45, 90, 180, 270)):
        super().__init__()
        self.angles = list(angles)

    def transform(self, inpt, params):
        angle = float(self.angles[torch.randint(len(self.angles), (1,)).item()])
        # per-image mean color as the fill (compute on the tensor, per channel)
        t = v2.functional.to_image(inpt) if not isinstance(inpt, torch.Tensor) else inpt
        t = v2.functional.to_dtype(t, torch.float32, scale=True)
        fill = t.mean(dim=(-1, -2)).tolist()      # [C] mean per channel
        return v2.functional.rotate(t, angle, fill=fill)


class SaltPepperNoise(v2.Transform):
    """
    Randomly set a fraction `amount` of pixels to 0 (pepper) or 1 (salt). Simulates sensor
    noise / compression artifacts. Operates on a float tensor in [0,1].
    """

    def __init__(self, amount: float = 0.02):
        super().__init__()
        self.amount = amount

    def transform(self, inpt, params):
        t = v2.functional.to_image(inpt) if not isinstance(inpt, torch.Tensor) else inpt
        t = v2.functional.to_dtype(t, torch.float32, scale=True).clone()
        mask = torch.rand_like(t[:1])             # one mask shared across channels [1,H,W]
        t[:, (mask[0] < self.amount / 2)] = 0.0   # pepper
        t[:, (mask[0] > 1 - self.amount / 2)] = 1.0  # salt
        return t


# ── pipelines ─────────────────────────────────────────────────────────────────

def _normalize_tail():
    """Shared ending: ensure float tensor in [0,1], then ImageNet-normalize."""
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_train_transform(img_size: int = 224) -> v2.Compose:
    """
    Heavy random augmentation for TRAINING. Each image independently gets a random subset
    of manipulations every epoch → effectively unlimited augmented data.

    Person C TODO: this is the main lever. Tune the probabilities (`p=`), add/remove
    transforms, and use error analysis to drop any that hurt a specific class
    (e.g. color inversion turns a yellow lemon blue → may mislabel). Candidates the team
    listed: grayscale, rotation(45/90/180 mean-fill), salt&pepper, inversion,
    color jitter. Provided OOD examples confirm color_jitter + rotation are real test ops.
    (shift + zoom removed per Person D — they translate/crop the object out of frame.)
    """
    resize = int(round(img_size * 256 / 224))
    return v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(img_size),
        v2.RandomHorizontalFlip(p=0.5),
        # rotation + colour jitter get the highest probabilities: they are the two
        # CONFIRMED real test axes (dataset/augmentations/{random_rotation,color_jitter}).
        v2.RandomApply([MeanFillRotation(angles=(45, 90, 180, 270))], p=0.5),
        v2.RandomApply([v2.ColorJitter(0.5, 0.5, 0.5, 0.1)], p=0.8),       # lighting/color
        v2.RandomGrayscale(p=0.15),
        v2.RandomApply([v2.RandomInvert()], p=0.08),                       # 🚩 label-risky, rare
        v2.RandomApply([SaltPepperNoise(amount=0.02)], p=0.2),
        *_normalize_tail(),
    ])


def build_light_train_transform(img_size: int = 224) -> v2.Compose:
    """
    MILD training transform — use this WITH the offline augmented dataset
    (`make_augmented.py` already baked the color/rotation/noise filters into the images on
    disk, so we must NOT heavily re-augment or we compound the distortion). Just a little
    geometric jitter + normalize. This is the default when training includes train_aug/.
    """
    resize = int(round(img_size * 256 / 224))
    return v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(img_size),
        v2.RandomHorizontalFlip(p=0.5),
        *_normalize_tail(),
    ])


def build_eval_transform(img_size: int = 224) -> v2.Compose:
    """
    Deterministic transform for VALIDATION / TEST. Mirrors evaluate.py exactly
    (Resize 256 → CenterCrop 224 → ToTensor → Normalize) so local numbers match the grader.
    """
    resize = int(round(img_size * 256 / 224))     # keep the 256/224 ratio if img_size changes
    return v2.Compose([
        v2.Resize(resize, antialias=True),
        v2.CenterCrop(img_size),
        *_normalize_tail(),
    ])


# Single-manipulation probes for robust_eval.py — measure invariance one axis at a time.
# These are kept aligned with the ACTIVE offline twins (make_augmented.py FILTERS): the
# probe strengths roughly mirror the baked-in twin ranges so local robustness numbers
# track what training actually saw. color_jitter + rotation are the confirmed real test
# axes, so we probe them at full strength.
OOD_TRANSFORMS = {
    "grayscale":      v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                                  v2.RandomGrayscale(p=1.0), *_normalize_tail()]),
    "color_jitter":   v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                                  v2.ColorJitter(0.5, 0.5, 0.5, 0.2), *_normalize_tail()]),
    "rotation":       v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                                  MeanFillRotation(angles=(45, 90, 180, 270)), *_normalize_tail()]),
    "invert":         v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                                  v2.RandomInvert(p=1.0), *_normalize_tail()]),
    "salt_pepper":    v2.Compose([v2.Resize(256, antialias=True), v2.CenterCrop(224),
                                  SaltPepperNoise(0.04), *_normalize_tail()]),
}


if __name__ == "__main__":
    # Self-check: every pipeline returns a normalized [3,224,224] tensor.
    from PIL import Image
    dummy = Image.new("RGB", (300, 250), (120, 90, 200))
    for name, tf in {"train": build_train_transform(), "eval": build_eval_transform(),
                     **OOD_TRANSFORMS}.items():
        out = tf(dummy)
        assert out.shape == (3, 224, 224), (name, out.shape)
    print("augment OK — all pipelines output [3,224,224]")
