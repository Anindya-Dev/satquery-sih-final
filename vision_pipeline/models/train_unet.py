"""Training loop for the U-Net segmentation model."""

import torch
import torch.nn as nn

from vision_pipeline.models.unet_segmentor import UNetSegmentor


def make_synthetic_batch(batch_size=2, h=120, w=120):
    """Build synthetic (input, label) pairs with a flood-like center region."""
    x = torch.rand(batch_size, 1, h, w)
    y = torch.zeros(batch_size, 1, h, w)
    y[:, :, 40:80, 40:80] = 1.0
    # Make the input correlate with the label: lower backscatter in flood zone.
    x[:, :, 40:80, 40:80] = x[:, :, 40:80, 40:80] * 0.3
    return x, y


def train_unet(model=None, epochs=10, batch_size=2, lr=1e-3, seed=0):
    """Train (or fine-tune) a UNetSegmentor on synthetic flood masks.

    Returns ``(model, losses)`` where ``losses`` is the per-epoch BCE loss.
    """
    torch.manual_seed(seed)
    if model is None:
        model = UNetSegmentor(in_channels=1, out_channels=1, base_channels=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(epochs):
        x, y = make_synthetic_batch(batch_size=batch_size)
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return model, losses


if __name__ == "__main__":
    model, losses = train_unet()
    print("losses:", losses)
    print("final loss:", losses[-1])
