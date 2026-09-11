from data_engine.loaders.bigearthnet_dataset import build_dataset as get_dataloader
import numpy as np


def get_member3_bitemporal_numpy(simulate_flood: bool = True, sar_mode: str = "raw_db"):
    # Mock 6-channel bitemporal array expected by adapters (6, 64, 64)
    arr = np.random.randn(6, 64, 64).astype(np.float32)
    if simulate_flood:
        arr[3:, 20:40, 20:40] -= 5.0  # Simulate flood change in post-event channels
    return arr


def get_member3_bitemporal_optical_numpy(simulate_change: bool = True):
    # Mock 6-channel optical bitemporal array (6, 64, 64)
    arr = np.random.rand(6, 64, 64).astype(np.float32)
    if simulate_change:
        arr[3:, 10:30, 10:30] += 0.3
    return arr


def get_member3_optical_sar_pair(sar_mode: str = "raw_db"):
    opt = np.random.rand(4, 64, 64).astype(np.float32)
    sar = np.random.randn(2, 64, 64).astype(np.float32)
    return {"optical": opt, "sar": sar}


__all__ = [
    "get_dataloader",
    "get_member3_bitemporal_numpy",
    "get_member3_bitemporal_optical_numpy",
    "get_member3_optical_sar_pair",
]