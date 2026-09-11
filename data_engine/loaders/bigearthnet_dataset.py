import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio

from .band_transforms import upsample_band, normalize_optical, process_sar_array

# BigEarthNet 12-Band standard Sentinel-2 order
S2_BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']

# Index mapping required by Member 3 (GIS / CV Specialist)
CHANNEL_MAP = {
    "optical": {band: idx for idx, band in enumerate(S2_BANDS)},
    "sar": {"VV": 0, "VH": 1},
    "fused": {**{band: idx for idx, band in enumerate(S2_BANDS)}, "VV": 12, "VH": 13}
}


class BigEarthNetMMDataset(Dataset):
    def __init__(self, s2_root=None, s1_root=None, mode="fused", target_size=(120, 120)):
        """
        Args:
            s2_root (str): Path to directory holding Sentinel-2 patch folders.
            s1_root (str): Path to directory holding Sentinel-1 patch folders.
            mode (str): 'optical', 'sar', or 'fused'.
            target_size (tuple): Target spatial resolution (120, 120).
        """
        self.s2_root = s2_root
        self.s1_root = s1_root
        self.mode = mode.lower()
        self.target_size = target_size
        self.patch_pairs = []

        if s2_root and os.path.exists(s2_root):
            s2_folders = sorted(glob.glob(os.path.join(s2_root, "*")))
            for s2_dir in s2_folders:
                if not os.path.isdir(s2_dir):
                    continue
                patch_name = os.path.basename(s2_dir)
                s1_dir = None
                if s1_root and os.path.exists(s1_root):
                    candidate = os.path.join(s1_root, patch_name)
                    if os.path.exists(candidate):
                        s1_dir = candidate
                    else:
                        pattern = os.path.join(s1_root, f"*{patch_name.split('_')[-2]}_{patch_name.split('_')[-1]}")
                        matches = glob.glob(pattern)
                        if matches:
                            s1_dir = matches[0]

                self.patch_pairs.append({
                    "patch_id": patch_name,
                    "s2_path": s2_dir,
                    "s1_path": s1_dir
                })

    def __len__(self):
        # Demo fallback: return 8 mock patches if local folders are empty
        return len(self.patch_pairs) if self.patch_pairs else 8

    def _read_band(self, path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
        return torch.from_numpy(data).unsqueeze(0)

    def _load_s2(self, s2_dir):
        tensors = []
        for band in S2_BANDS:
            matches = glob.glob(os.path.join(s2_dir, f"*_{band}.tif"))
            if not matches:
                tensors.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
            else:
                band_t = self._read_band(matches[0])
                band_t = upsample_band(band_t, self.target_size)
                band_t = normalize_optical(band_t)
                tensors.append(band_t)
        return torch.cat(tensors, dim=0)  # Shape: (12, 120, 120)

    def _load_s1(self, s1_dir):
        tensors = []
        for band in S1_BANDS:
            matches = glob.glob(os.path.join(s1_dir, f"*_{band}.tif")) if s1_dir else []
            if not matches:
                tensors.append(torch.zeros((1, *self.target_size), dtype=torch.float32))
            else:
                with rasterio.open(matches[0]) as src:
                    data = src.read(1).astype(np.float32)
                tensors.append(process_sar_array(data))
        return torch.cat(tensors, dim=0)  # Shape: (2, 120, 120)

    def __getitem__(self, idx):
        if not self.patch_pairs:
            # Synthetic generation to ensure live demos never crash without data
            patch_id = f"demo_patch_{idx:03d}"
            s2_tensor = torch.rand((12, *self.target_size), dtype=torch.float32)
            s1_tensor = torch.rand((2, *self.target_size), dtype=torch.float32)
        else:
            item = self.patch_pairs[idx]
            patch_id = item["patch_id"]
            s2_tensor = self._load_s2(item["s2_path"]) if self.mode in ["optical", "fused"] else None
            s1_tensor = self._load_s1(item["s1_path"]) if self.mode in ["sar", "fused"] else None

        if self.mode == "optical":
            tensor = s2_tensor
        elif self.mode == "sar":
            tensor = s1_tensor
        elif self.mode == "fused":
            if s2_tensor is None:
                s2_tensor = torch.zeros((12, *self.target_size), dtype=torch.float32)
            if s1_tensor is None:
                s1_tensor = torch.zeros((2, *self.target_size), dtype=torch.float32)
            tensor = torch.cat([s2_tensor, s1_tensor], dim=0)  # Shape: (14, 120, 120)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Must be 'optical', 'sar', or 'fused'.")

        return {
            "patch_id": patch_id,
            "image": tensor,
            "mode": self.mode
        }


def get_dataloader(s2_root=None, s1_root=None, mode="fused", batch_size=4, shuffle=False):
    """Convenience factory function for the DataLoader."""
    dataset = BigEarthNetMMDataset(s2_root=s2_root, s1_root=s1_root, mode=mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )