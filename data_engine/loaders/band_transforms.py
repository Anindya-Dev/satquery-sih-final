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


def process_sar_array(sar_data: np.ndarray, return_raw_db: bool = False) -> torch.Tensor:
    """
    Converts raw Sentinel-1 linear backscatter to Decibels (dB).
    - If return_raw_db=True: returns un-normalized dB clipped to [-35.0, 0.0] dB.
    - If return_raw_db=False: scales [-35.0, 0.0] dB into [0.0, 1.0].
    """
    raw_clipped = np.clip(sar_data, a_min=1e-6, a_max=None)
    sar_db = 10.0 * np.log10(raw_clipped)
    sar_db = np.clip(sar_db, -35.0, 0.0)

    if return_raw_db:
        return torch.from_numpy(sar_db).float().unsqueeze(0)

    # Min-max normalization: [-35 dB, 0 dB] -> [0.0, 1.0]
    sar_norm = (sar_db - (-35.0)) / (0.0 - (-35.0))
    return torch.from_numpy(sar_norm).float().unsqueeze(0)