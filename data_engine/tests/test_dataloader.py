import os
import sys
from types import SimpleNamespace
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data_engine.loaders import (
    CHANNEL_MAP,
    get_dataloader,
    get_member3_bitemporal_numpy,
    load_from_task_spec,
)


def run_tests():
    print("Running Data Pipeline & Integration Verification...\n")

    # 1. Base Dataloader Verification
    fused_loader = get_dataloader(mode="fused", batch_size=2)
    sample_fused = next(iter(fused_loader))
    assert sample_fused["image"].shape == (2, 14, 120, 120), "Fused shape mismatch!"
    print(" [PASS] Base Loader (Fused): (2, 14, 120, 120)")

    # 2. Member 3 Bitemporal Array & Physical dB Verification
    data_np = get_member3_bitemporal_numpy(simulate_flood=True, sar_mode="raw_db")
    assert isinstance(data_np, np.ndarray), "Output is not a numpy array!"
    assert data_np.dtype == np.float32, f"Expected float32, got {data_np.dtype}"
    assert data_np.shape == (6, 120, 120), f"Expected (6, 120, 120), got {data_np.shape}"
    print(" [PASS] Member 3 Array: shape (6, 120, 120), dtype float32")

    # Check the simulated 6 dB flood drop: (VV_pre - VV_post) >= 3.0 dB
    vv_pre = data_np[4]
    vv_post = data_np[5]
    delta_flood = (vv_pre - vv_post)[40:80, 40:80]
    assert np.all(delta_flood >= 3.0), "Simulated flood drop is less than 3.0 dB!"
    print(" [PASS] Member 3 SAR physical drop verified (delta >= 3.0 dB in flood zone)")

    # 3. Member 2 Router (TaskSpec) Adapter Verification
    # Test BITEMPORAL_PAIR
    mock_bitemp_spec = SimpleNamespace(modality="BITEMPORAL_PAIR")
    res_bitemp = load_from_task_spec(mock_bitemp_spec)
    assert isinstance(res_bitemp, np.ndarray) and res_bitemp.shape == (6, 120, 120)
    print(" [PASS] Router Adapter -> BITEMPORAL_PAIR: (6, 120, 120) numpy array")

    # Test OPTICAL
    mock_opt_spec = SimpleNamespace(
        modality="OPTICAL",
        parameters=SimpleNamespace(bands_required=["B02", "B03", "B04", "B08"])
    )
    res_opt = load_from_task_spec(mock_opt_spec)
    assert res_opt["image"].shape == (12, 120, 120)
    print(" [PASS] Router Adapter -> OPTICAL: (12, 120, 120) tensor")

    # Test CROSS_MODAL_PAIR (Fused)
    mock_fused_spec = SimpleNamespace(modality="CROSS_MODAL_PAIR")
    res_fused = load_from_task_spec(mock_fused_spec)
    assert res_fused["image"].shape == (14, 120, 120)
    print(" [PASS] Router Adapter -> CROSS_MODAL_PAIR: (14, 120, 120) tensor")

    print("\n--- ALL INTEGRATION TESTS PASSED SUCCESSFULLY ---")


if __name__ == "__main__":
    run_tests()