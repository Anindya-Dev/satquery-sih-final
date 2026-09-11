import numpy as np

from vision_pipeline.evidence.evidence_generator import generate_evidence


def test_sar_flood_extractor():
    vv_pre = np.array([[-5.0, -8.0], [-10.0, -9.0]], dtype=np.float32)
    vv_post = np.array([[-9.0, -9.0], [-13.0, -20.0]], dtype=np.float32)

    task_spec = {
        "primary_tool": "sar_flood_extractor",
        "evidence_requested": ["inundation_mask", "flooded_area_km2"],
    }
    imagery = {"sar": {"VV_pre": vv_pre, "VV_post": vv_post}}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    entry = evidence[0]
    assert entry["tool"] == "sar_flood_extractor"
    assert entry["target"] == "flood"

    expected_mask = np.array([[True, False], [True, True]])
    assert np.array_equal(entry["results"]["inundation_mask"], expected_mask)
    assert abs(entry["results"]["flooded_area_km2"] - 0.0003) < 1e-9
    assert abs(entry["results"]["confidence_mean"] - 0.65) < 1e-6

    expected_math = (
        "flood if (VV_pre - VV_post) >= 3.0 dB; "
        "confidence = min(1.0,(VV_pre-VV_post)/5.0)"
    )
    assert entry["math"] == expected_math


def test_spectral_indices_calculator():
    nir = np.array([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    red = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    green = np.array([[0.5, 0.2], [0.0, 1.0]], dtype=np.float32)

    task_spec = {
        "primary_tool": "spectral_indices_calculator",
        "parameters": {"indices_requested": ["NDVI", "NDWI"]},
    }
    imagery = {"optical": {"NIR": nir, "RED": red, "GREEN": green}}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 2
    assert evidence[0]["target"] == "vegetation"
    assert "NDVI_map" in evidence[0]["results"]
    assert evidence[0]["results"]["confidence_mean"] == 0.75
    assert evidence[1]["target"] == "water"
    assert "NDWI_map" in evidence[1]["results"]


def test_bitemporal_change_detector():
    nir_t1 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    red_t1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    nir_t2 = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    red_t2 = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32)

    task_spec = {
        "primary_tool": "bitemporal_change_detector",
        "evidence_requested": ["change_mask", "changed_area_km2"],
    }
    imagery = {
        "optical": {
            "t1": {"NIR": nir_t1, "RED": red_t1},
            "t2": {"NIR": nir_t2, "RED": red_t2},
        }
    }

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    entry = evidence[0]
    assert entry["tool"] == "bitemporal_change_detector"
    assert entry["target"] == "change"

    expected_mask = np.array([[False, True], [True, True]])
    assert np.array_equal(entry["results"]["change_mask"], expected_mask)
    assert abs(entry["results"]["changed_area_km2"] - 0.0003) < 1e-9
    assert abs(entry["results"]["confidence_mean"] - 0.75) < 1e-6


def test_optical_sar_fusion_specialist():
    nir = np.array([[1.0, 0.0]], dtype=np.float32)
    green = np.array([[0.0, 1.0]], dtype=np.float32)
    vv = np.array([[-2.0, -20.0]], dtype=np.float32)

    task_spec = {"primary_tool": "optical_sar_fusion_specialist"}
    imagery = {
        "optical": {"NIR": nir, "GREEN": green},
        "sar": {"VV": vv},
    }

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    results = evidence[0]["results"]
    assert np.array_equal(results["water_mask"], np.array([[False, True]]))
    assert np.array_equal(results["built_up_mask"], np.array([[True, False]]))
    assert "confidence_mean" in results


def test_grounding_rs_specialist():
    nir = np.zeros((3, 3), dtype=np.float32)
    green = np.zeros((3, 3), dtype=np.float32)
    green[1, 1] = 0.5

    task_spec = {
        "primary_tool": "grounding_rs_specialist",
        "parameters": {"grounding_target": "water"},
    }
    imagery = {"optical": {"NIR": nir, "GREEN": green}}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    boxes = evidence[0]["results"]["bounding_boxes"]
    assert len(boxes) == 1
    assert boxes[0]["x_min"] == 1
    assert boxes[0]["y_min"] == 1
    assert boxes[0]["x_max"] == 1
    assert boxes[0]["y_max"] == 1
    assert boxes[0]["area_pixels"] == 1


def test_grounding_rs_specialist_vegetation():
    nir = np.zeros((3, 3), dtype=np.float32)
    red = np.zeros((3, 3), dtype=np.float32)
    nir[1, 1] = 1.0

    task_spec = {
        "primary_tool": "grounding_rs_specialist",
        "parameters": {"grounding_target": "vegetation"},
    }
    imagery = {"optical": {"NIR": nir, "RED": red}}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    boxes = evidence[0]["results"]["bounding_boxes"]
    assert len(boxes) == 1
    assert boxes[0]["x_min"] == 1
    assert boxes[0]["y_min"] == 1


def test_grounding_rs_specialist_built_up():
    vv = np.full((3, 3), -20.0, dtype=np.float32)
    vv[1, 1] = -5.0

    task_spec = {
        "primary_tool": "grounding_rs_specialist",
        "parameters": {"grounding_target": "built_up"},
    }
    imagery = {"sar": {"VV": vv}}

    evidence = generate_evidence(task_spec, imagery)

    assert len(evidence) == 1
    boxes = evidence[0]["results"]["bounding_boxes"]
    assert len(boxes) == 1
    assert boxes[0]["x_min"] == 1
    assert boxes[0]["y_min"] == 1


def test_unknown_tool_returns_empty():
    task_spec = {"primary_tool": "nonsense"}
    assert generate_evidence(task_spec, {}) == []
