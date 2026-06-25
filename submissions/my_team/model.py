"""
model.py  —  OWNER: Person A (Architecture)

Defines `ModelArchitecture`: the CNN the grader will reconstruct and load weights into.

╔══════════════════════════════════════════════════════════════════════════════╗
║ HARD RULES (break one → grader fails or you score zero):                       ║
║  1. The class MUST be named exactly `ModelArchitecture` and subclass nn.Module.║
║  2. `forward(x)` takes x of shape [B, 3, 224, 224] and returns LOGITS [B, 20]. ║
║     (Raw scores — NOT softmax/probabilities. predict.py does the argmax.)      ║
║  3. This file MUST be self-contained: import ONLY torch / torch.nn / stdlib.   ║
║     Never `from data import ...` etc. — the grader copies just this file.      ║
║  4. No BatchNorm in the baseline (team hasn't covered it; keep it explainable).║
║     Add it later ONLY if you've learned it and can defend it in the interview. ║
╚══════════════════════════════════════════════════════════════════════════════╝

This baseline already runs and trains. Person A's job is to IMPROVE it (see TODOs):
deeper/wider blocks, dropout rate, etc. — while keeping the 4 rules above.
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, n_convs: int = 2) -> nn.Sequential:
    """
    One VGG-style stage: `n_convs` × (3x3 conv → ReLU), then halve H,W with MaxPool.

    Why this shape: stacking small 3x3 convs is the standard, easy-to-explain way to
    grow the receptive field. ReLU is the nonlinearity. MaxPool(2) downsamples so deeper
    layers see larger context with fewer pixels. No BatchNorm on purpose (see header).
    """
    layers = []
    c = in_ch
    for _ in range(n_convs):
        layers += [nn.Conv2d(c, out_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        c = out_ch
    layers += [nn.MaxPool2d(kernel_size=2)]   # halves spatial size
    return nn.Sequential(*layers)


class ModelArchitecture(nn.Module):
    """
    Plain VGG-style CNN, from random init.

    Pipeline for a 224×224 input:
        224 →[block1]→ 112 →[block2]→ 56 →[block3]→ 28 →[block4]→ 14
        → AdaptiveAvgPool2d(1) → 256-vector → Dropout → Linear → 20 logits

    `AdaptiveAvgPool2d(1)` collapses any HxW feature map to 1×1, so the model accepts
    ANY input size (we may train at 128 but the grader feeds 224 — both work).
    """

    def __init__(self, num_classes: int = 20, dropout: float = 0.35):
        super().__init__()

        # Feature extractor — channels grow as spatial size shrinks (standard CNN funnel).
        self.features = nn.Sequential(
            conv_block(3,   32, n_convs=2),   # -> 112x112
            conv_block(32,  64, n_convs=2),   # ->  56x56
            conv_block(64,  128, n_convs=2),  # ->  28x28
            conv_block(128, 256, n_convs=1),  # ->  14x14
        )

        # Global average pool → one number per channel. Robust to input size.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head. Dropout fights overfitting (we have only 20k images).
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

        # TODO(Person A): try to BEAT this baseline. Ideas, cheapest first:
        #   - bump dropout (0.3 → 0.5) if you see train >> val accuracy (overfit)
        #   - add a 5th conv_block (256→512) for more capacity
        #   - widen channels (32→64 start) — slower but stronger
        #   - add a hidden Linear(256→256)+ReLU in the head
        # Re-run `python run.py train` after each change and check the leaderboard.

    def _init_weights(self) -> None:
        """Kaiming init for convs/linears — good default for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        logits = self.classifier(x)   # [B, 20]
        return logits


if __name__ == "__main__":
    # Smoke test: shape contract. Run `python model.py` to sanity-check.
    net = ModelArchitecture()
    out = net(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 20), out.shape
    print("OK  logits shape:", tuple(out.shape), "| params:",
          sum(p.numel() for p in net.parameters()))
