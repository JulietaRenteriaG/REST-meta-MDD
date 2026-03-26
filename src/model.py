import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2D + BatchNorm + ReLU + optional MaxPool."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ReHoCNN(nn.Module):
    """
    2.5D CNN para clasificacion MDD vs HC.
    Input:  (B, 3, 64, 64)  — 3 slices ortogonales del volumen ReHo
    Output: (B, 2)           — logits [HC, MDD]
    """
    def __init__(self, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock( 3, 32, pool=True),    # → (B, 32, 32, 32)
            nn.Dropout2d(0.1),
            ConvBlock(32, 64, pool=True),    # → (B, 64, 16, 16)
            nn.Dropout2d(0.2),
            ConvBlock(64, 64, pool=True),    # → (B, 64, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),         # → (B, 64, 4, 4)
            nn.Flatten(),                    # → (B, 1024)
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = ReHoCNN()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total:,}")

    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(model)