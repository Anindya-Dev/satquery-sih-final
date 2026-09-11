"""U-Net segmentation model for SAR flood mapping."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two stacked (Conv2d -> BatchNorm -> ReLU) blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSegmentor(nn.Module):
    """3-level U-Net.

    Uses three downsampling stages (factor 8), so it accepts inputs whose
    height and width are divisible by 8 (e.g. 120x120).
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

    @torch.no_grad()
    def segment(self, x, threshold=0.5):
        """Run inference and return a boolean mask (sigmoid + threshold).

        Accepts a ``(H, W)``, ``(C, H, W)``, or ``(B, C, H, W)`` float32 input
        and returns a NumPy boolean mask of matching spatial shape.
        """
        self.eval()
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        mask = probs > threshold
        return mask.squeeze().cpu().numpy()
