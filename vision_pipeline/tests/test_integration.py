import numpy as np

from vision_pipeline.adapters import (
    bitemporal_array_to_imagery,
    optical_sar_pair_to_imagery,
)
from vision_pipeline.evidence.evidence_generator import generate_evidence


def test_flood_tile_validation_raw_db():
    # Mirror Member 1's get_member3_bitemporal_numpy(sar_mode="raw_db",
    # simulate_flood=True): (6, 120, 120) float32 with VV_pre in [-25, -5] dB
    # and a -6 dB drop in the center flood zone.
    data = np.zeros((6, 120, 120), dtype=np.float32)
    data[0:4] = 0.5  # optical bands, arbitrary [0, 1]
    data[4] = -10.0  # VV_pre, raw dB
    data[5] = -10.0  # VV_post = VV_pre...
    data[5, 40:80, 40:80] = -16.0  # ...minus 6 dB in the flood zone

    imagery = bitemporal_array_to_imagery(data)
    task_spec = {
        "primary_tool": "sar_flood_extractor",
        "evidence_requested": ["inundation_mask", "flooded_area_km2", "confidence_mean"],
    }

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    results = evidence[0]["results"]
    mask = results["inundation_mask"]

    assert mask.dtype == np.bool_
    assert mask[40:80, 40:80].all()
    assert int(mask.sum()) == 1600  # 40x40 flood zone only

    # 1600 pixels * 0.0001 km2/pixel = 0.16 km2
    assert abs(results["flooded_area_km2"] - 0.16) < 1e-6

    # drop = 6 dB -> confidence 1.0 in flood zone, 0.0 elsewhere
    assert abs(results["confidence_mean"] - (1600 / 14400)) < 1e-6


def test_fusion_tile_validation_raw_db():
    # Mirror Member 1's get_member3_optical_sar_pair(sar_mode="raw_db"):
    # (5, 120, 120) float32 with NIR, RED, GREEN, BLUE, and VV in raw dB.
    data = np.zeros((5, 120, 120), dtype=np.float32)
    data[2][10, 10] = 0.6  # GREEN high, NIR low -> NDWI > 0 (water)
    data[4] = -20.0  # VV baseline (low backscatter)
    data[4][20, 20] = -5.0  # one built-up pixel (high backscatter)

    imagery = optical_sar_pair_to_imagery(data)
    task_spec = {"primary_tool": "optical_sar_fusion_specialist"}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    results = evidence[0]["results"]
    assert bool(results["water_mask"][10, 10])
    assert bool(results["built_up_mask"][20, 20])
