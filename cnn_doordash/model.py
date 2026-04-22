"""CNN architecture for menu-item price regression."""

from __future__ import annotations

import torch
from torch import nn


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    """Two 3x3 convs (Conv -> BN -> ReLU) followed by 2x2 max-pool."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class PriceCNN(nn.Module):
    """From-scratch CNN: 224x224 RGB image -> scalar dollar price."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),    # 224 -> 112
            _conv_block(32, 64),   # 112 ->  56
            _conv_block(64, 128),  #  56 ->  28
            _conv_block(128, 256), #  28 ->  14
            _conv_block(256, 512), #  14 ->   7
            nn.AdaptiveAvgPool2d(1),  # 7 -> 1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Explicit Kaiming init matches our ReLU non-linearities and keeps
        # activation magnitudes from exploding through deep stacks.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
