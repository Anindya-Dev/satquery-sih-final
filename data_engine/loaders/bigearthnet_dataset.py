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

def get_member3_bitemporal_numpy(s2_root=None, s1_root=None, simulate_flood=True, sar_mode="raw_db"):
    """
    Returns an aligned NumPy float32 array formatted specifically for Member 3:
    Shape: (6, 120, 120)
    Channels: [NIR, RED, GREEN, BLUE, VV_pre, VV_post]
    
    sar_mode:
        - "raw_db": SAR in physical Decibels [-35.0, 0.0] dB (ideal for physical threshold rules: Δ >= 3.0 dB)
        - "normalized": SAR scaled to [0.0, 1.0] (ideal for neural networks / U-Net)
    """
    loader = get_dataloader(s2_root=s2_root, s1_root=s1_root, mode="fused", batch_size=2)
    iterator = iter(loader)
    
    batch_1 = next(iterator)
    tensor_pre = batch_1["image"][0]  # (14, 120, 120)

    # BigEarthNet indices: NIR=B08 (idx 7), RED=B04 (idx 3), GREEN=B03 (idx 2), BLUE=B02 (idx 1)
    nir   = tensor_pre[7:8]
    red   = tensor_pre[3:4]
    green = tensor_pre[2:3]
    blue  = tensor_pre[1:2]
    
    # Internal tensor is normalized [0, 1] from [-35, 0] dB
    vv_norm = tensor_pre[12:13]
    # Reconstruct dB scale
    vv_raw_db = (vv_norm * 35.0) - 35.0

    if sar_mode == "raw_db":
        # Ensure baseline is safely above floor so a drop doesn't collapse against -35 dB
        vv_pre = torch.clamp(vv_raw_db, -25.0, -5.0)
        if simulate_flood:
            vv_post = vv_pre.clone()
            # Specular water reflection drops backscatter by 6.0 dB in the flood zone
            vv_post[:, 40:80, 40:80] = vv_post[:, 40:80, 40:80] - 6.0
        else:
            if batch_1["image"].shape[0] > 1:
                raw_second = (batch_1["image"][1][12:13] * 35.0) - 35.0
                vv_post = torch.clamp(raw_second, -35.0, 0.0)
            else:
                vv_post = vv_pre.clone()
    else:  # "normalized" [0, 1]
        vv_pre = vv_norm.clone()
        if simulate_flood:
            vv_post = vv_pre.clone()
            # 6 dB drop in a 35 dB range is 6/35 ≈ 0.171
            vv_post[:, 40:80, 40:80] = torch.clamp(vv_post[:, 40:80, 40:80] - (6.0 / 35.0), 0.0, 1.0)
        else:
            vv_post = batch_1["image"][1][12:13] if batch_1["image"].shape[0] > 1 else vv_pre.clone()

    bitemporal_tensor = torch.cat([nir, red, green, blue, vv_pre, vv_post], dim=0)
    return bitemporal_tensor.cpu().numpy().astype(np.float32)

def get_member3_optical_sar_pair(s2_root=None, s1_root=None, sar_mode="raw_db"):
    """
    Returns a single co-registered Optical + SAR pair formatted for Member 3's fusion specialist:
    Shape: (5, 120, 120)
    Channels: [NIR, RED, GREEN, BLUE, VV]
    """
    # Slice the first 5 channels [NIR, RED, GREEN, BLUE, VV_pre] from the bitemporal array
    bitemp = get_member3_bitemporal_numpy(
        s2_root=s2_root,
        s1_root=s1_root,
        simulate_flood=False,
        sar_mode=sar_mode
    )
    return bitemp[:5].copy()

def load_from_task_spec(task_spec, s2_root=None, s1_root=None, simulate_flood=True, sar_mode="raw_db"):
    """
    Router adapter: maps Member 2's TaskSpec object directly into 
    the aligned tensors/arrays expected by Member 3.
    """
    modality = getattr(task_spec, "modality", "CROSS_MODAL_PAIR")
    if hasattr(modality, "value"):  # Handle Enum if Member 2 used Enum
        modality = modality.value
    modality = str(modality).upper()

    # Case 1: Bi-temporal flood detection -> Member 3's bitemporal numpy array
    if modality == "BITEMPORAL_PAIR":
        return get_member3_bitemporal_numpy(
            s2_root=s2_root,
            s1_root=s1_root,
            simulate_flood=simulate_flood,
            sar_mode=sar_mode
        )

    # Case 2: Standard modalities -> mapped to data loader mode
    mode_mapping = {
        "OPTICAL": "optical",
        "SAR": "sar",
        "CROSS_MODAL_PAIR": "fused"
    }
    target_mode = mode_mapping.get(modality, "fused")

    loader = get_dataloader(s2_root=s2_root, s1_root=s1_root, mode=target_mode, batch_size=1)
    batch = next(iter(loader))
    
    return {
        "patch_id": batch["patch_id"][0],
        "image": batch["image"].squeeze(0),  # Shape: (C, 120, 120)
        "modality": modality,
        "bands": getattr(getattr(task_spec, "parameters", None), "bands_required", None)
    }