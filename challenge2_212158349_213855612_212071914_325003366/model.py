"""
model.py  —  ModelArchitecture (CNN classifier)

Defines `ModelArchitecture`: the CNN the grader will reconstruct and load weights into.

╔══════════════════════════════════════════════════════════════════════════════╗
║ HARD RULES (break one → grader fails or you score zero):                       ║
║  1. The class MUST be named exactly `ModelArchitecture` and subclass nn.Module.║
║  2. `forward(x)` takes x of shape [B, 3, 224, 224] and returns LOGITS [B, 20]. ║
║     (Raw scores — NOT softmax/probabilities. predict.py does the argmax.)      ║
║  3. This file MUST be self-contained: import ONLY torch / torch.nn / stdlib.   ║
║     Never `from data import ...` etc. — the grader copies just this file.      ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE: a small ResNet (ResNet-18 shape) with Squeeze-and-Excitation channel attention,
trained from random init.

  - Conv 3x3 + ReLU + MaxPool + Global Average Pool.
  - Residual / skip connections  y = f(x) + x  (He et al., 2016): give gradients a direct path
        backward so the added depth is trainable from random init (mitigates vanishing gradients).
  - BatchNorm after each conv: normalizes activations per channel (across the batch and spatial
        positions), conditioning the loss surface so the deep network optimizes cleanly and fast.
  - Squeeze-and-Excitation (Hu et al., 2018) inside each residual block, after the two convs and
        before the residual add: a small bottleneck MLP reads the global average of each feature
        map and outputs one 0..1 gate per channel, then rescales the channels by it:
            s = sigmoid( W2 · ReLU( W1 · GlobalAvgPool(x) ) )   # [B, C, 1, 1]
            out = out * s                                       # channel-wise gate
        This up-weights informative channels and suppresses distractor channels, improving
        robustness to colour / lighting / background shifts (the out-of-distribution half of the
        score) at low cost (bottleneck reduction=16, ~0.09M params).
  - Dropout (0.4) on the head + L2 weight decay (in training) + Global Average Pool (no large
        fully-connected layers before the final 512->20 linear) regularize the ~11.3M parameters.
  - Kaiming/He weight initialization (fan-out, ReLU).
"""

import torch
import torch.nn as nn


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    """3x3 conv, padding 1 (keeps H,W unless stride>1). bias=False — the following
    BatchNorm has its own shift, so a conv bias would be redundant."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018). Learns one gate per channel:
        squeeze:   z = GlobalAvgPool(x)               # [B,C,1,1]  — global context per channel
        excite:    s = sigmoid(W2 · ReLU(W1 · z))     # [B,C,1,1]  — 0..1 importance per channel
        recalibrate: out = x * s                       # up-weight useful channels, damp distractors
    The two FC layers form a bottleneck (C → C/reduction → C), so it's cheap (~C²/reduction
    params, no spatial cost). Implemented with 1x1 convs so it works on the [B,C,H,W] map directly.
    """

    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        hidden = max(ch // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))     # broadcast [B,C,1,1] gate over H,W


class ResidualBlock(nn.Module):
    """
    Basic residual block (He et al., 2016): two (3x3 conv → BN → ReLU) with a skip
    connection that adds the input back in:  out = ReLU( F(x) + identity ).

    The skip lets gradients flow straight past the block, so vanishing gradients don't
    kill the deep layers and the depth stays trainable from random init.
    When the block changes the channel count or downsamples (stride=2), the identity branch
    is projected by a 1x1 conv (+BN) so the two tensors have matching shapes before the add.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch)            # channel attention on F(x) before the residual add

        # Identity branch: only project when the shape actually changes.
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)              # SE channel attention (recalibrate channels before the add)
        out = out + identity            # residual add
        return self.relu(out)


class ModelArchitecture(nn.Module):
    """
    Small ResNet for 224×224 RGB input → 20 logits.

    Pipeline (channels in []):
        224×224×3
          → stem  (7x7 s2 conv → BN → ReLU → 3x3 s2 maxpool)   → 56×56  [64]
          → stage1 (2 residual blocks, stride 1)                → 56×56  [64]
          → stage2 (2 residual blocks, first stride 2)          → 28×28  [128]
          → stage3 (2 residual blocks, first stride 2)          → 14×14  [256]
          → stage4 (2 residual blocks, first stride 2)          →  7×7   [512]
          → AdaptiveAvgPool2d(1) → Dropout(0.4) → Linear        → 20 logits

    `AdaptiveAvgPool2d(1)` collapses any H×W to 1×1, so the net accepts ANY input size
    (we may train at 128 for speed; the grader feeds 224 — both work).

    NOTE on the grader contract: predict.py builds the net via `ModelArchitecture()` with NO
    args, then load_state_dict(...). So these DEFAULT constructor args (base=64, blocks (2,2,2,2),
    num_classes=20) define the exact architecture weights are loaded into — train.py must build
    the net the same way. `dropout` is the only safe knob to vary without touching state_dict keys
    or tensor shapes (Dropout holds no parameters), so tuning it never breaks weight loading.
    """

    def __init__(self, num_classes: int = 20, base: int = 64,
                 blocks_per_stage=(2, 2, 2, 2), dropout: float = 0.4):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        # Stem: aggressive early downsample (224 → 56) so the deep stages stay cheap on CPU/MPS.
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 residual stages; the first block of stages 2-4 halves H,W (stride 2) and doubles channels.
        self.stage1 = self._make_stage(c1, c1, blocks_per_stage[0], stride=1)
        self.stage2 = self._make_stage(c1, c2, blocks_per_stage[1], stride=2)
        self.stage3 = self._make_stage(c2, c3, blocks_per_stage[2], stride=2)
        self.stage4 = self._make_stage(c3, c4, blocks_per_stage[3], stride=2)

        # Global average pool → one number per channel (pooling fights overfitting; size-agnostic).
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head. Dropout (0.4) regularizes the ~11.3M parameters.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

        self._init_weights()

        # `dropout` is the only constructor arg that does not change state_dict keys/shapes
        # (Dropout holds no parameters), so it can be retuned without regenerating the weights.

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        """First block may downsample / change width; the rest keep shape (stride 1, out→out)."""
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming init for convs (good default for ReLU); BN weight=1/bias=0; Linear ~N(0,0.01)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        logits = self.classifier(x)     # [B, 20]
        return logits


if __name__ == "__main__":
    # Smoke test: shape contract. Run `python model.py` to sanity-check.
    net = ModelArchitecture()
    net.eval()                          # eval so BatchNorm uses running stats (batch-size-safe)
    out = net(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 20), out.shape
    print("OK  logits shape:", tuple(out.shape), "| params:",
          sum(p.numel() for p in net.parameters()))
