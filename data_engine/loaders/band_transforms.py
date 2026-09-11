"""Band transform utilities."""

import numpy as np
import torch
import torch.nn.functional as F


def upsample_band(band_tensor: torch.Tensor, target_size=(120, 120)) -> torch.Tensor:
    """
    Upsamples 20m (60x60) or 60m (20x20) band tensors to the 10m grid (120x120).
    Expected input shape: (1, H, W) -> Output shape: (1, 120, 120)
    """
    if band_tensor.shape[-2:] == target_size:
        return band_tensor

    # F.interpolate requires a 4D batch tensor: (Batch, Channels, Height, Width)
    tensor_4d = band_tensor.unsqueeze(0)
    upsampled = F.interpolate(
        tensor_4d,
        size=target_size,
        mode="bilinear",
        align_corners=False
    )
    return upsampled.squeeze(0)


def normalize_optical(band_tensor: torch.Tensor) -> torch.Tensor:
    """
    Clamps and normalizes Sentinel-2 surface reflectance [0, 10000] -> [0.0, 1.0].
    """
    return torch.clamp(band_tensor / 10000.0, 0.0, 1.0)


def process_sar_array(sar_data: np.ndarray) -> torch.Tensor:
    """
    Converts raw Sentinel-1 backscatter to Decibels (dB), clips noise [-35, 0],
    and normalizes to [0.0, 1.0]. Output shape: (1, H, W)
    """
    # 1. Prevent log of zero or negative numbers
    raw_clipped = np.clip(sar_data, a_min=1e-5, a_max=None)

    # 2. Linear power to Decibel (dB) scale
    sar_db = 10.0 * np.log10(raw_clipped)

    # 3. Standard Sentinel-1 noise floor clipping: [-35 dB, 0 dB]
    sar_db = np.clip(sar_db, -35.0, 0.0)

    # 4. Scale to [0.0, 1.0] range
    sar_norm = (sar_db - (-35.0)) / (0.0 - (-35.0))

    return torch.from_numpy(sar_norm).float().unsqueeze(0)