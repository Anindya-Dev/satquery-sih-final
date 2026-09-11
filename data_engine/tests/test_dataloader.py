import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data_engine.loaders import CHANNEL_MAP, get_dataloader


def run_tests():
    print("Running DataLoader pipeline verification...\n")

    # 1. Test Fused Mode -> (Batch, 14, 120, 120)
    fused_loader = get_dataloader(mode="fused", batch_size=2)
    sample_fused = next(iter(fused_loader))
    assert sample_fused["image"].shape == (2, 14, 120, 120), (
        f"Fused shape mismatch! Got {sample_fused['image'].shape}"
    )
    print(" [PASS] Fused loader yielded shape (2, 14, 120, 120)")

    # 2. Test Optical Mode -> (Batch, 12, 120, 120)
    optical_loader = get_dataloader(mode="optical", batch_size=2)
    sample_opt = next(iter(optical_loader))
    assert sample_opt["image"].shape == (2, 12, 120, 120), (
        f"Optical shape mismatch! Got {sample_opt['image'].shape}"
    )
    print(" [PASS] Optical loader yielded shape (2, 12, 120, 120)")

    # 3. Test SAR Mode -> (Batch, 2, 120, 120)
    sar_loader = get_dataloader(mode="sar", batch_size=2)
    sample_sar = next(iter(sar_loader))
    assert sample_sar["image"].shape == (2, 2, 120, 120), (
        f"SAR shape mismatch! Got {sample_sar['image'].shape}"
    )
    print(" [PASS] SAR loader yielded shape (2, 2, 120, 120)")

    print("\nChannel Mapping for Member 3 (GIS):")
    print(f"Optical Map: {CHANNEL_MAP['optical']}")
    print(f"SAR Map:     {CHANNEL_MAP['sar']}")
    print("\n--- ALL TESTS PASSED SUCCESSFULLY ---")


if __name__ == "__main__":
    run_tests()