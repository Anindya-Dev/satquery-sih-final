import numpy as np

from vision_pipeline.sar.backscatter import detect_built_up, sar_linear_to_db
from vision_pipeline.sar.flood_detection import flood_detection_rule_based


def test_linear_to_db():
    linear = np.array([[1.0, 0.1], [10.0, 0.0]], dtype=np.float32)

    db = sar_linear_to_db(linear)

    eps = 1e-6
    expected = np.array([
        [10.0 * np.log10(1.0 + eps), 10.0 * np.log10(0.1 + eps)],
        [10.0 * np.log10(10.0 + eps), -60.0],
    ], dtype=np.float64)
    np.testing.assert_allclose(db, expected, atol=1e-5)


def test_flood_detection_rule_based_simple():
    vv_pre = np.array([[-5.0, -8.0], [-10.0, -9.0]], dtype=np.float32)
    vv_post = np.array([[-9.0, -9.0], [-13.0, -20.0]], dtype=np.float32)

    flood_mask, confidence = flood_detection_rule_based(vv_pre, vv_post)

    expected_mask = np.array([[True, False], [True, True]])
    assert np.array_equal(flood_mask, expected_mask)
    assert abs(float(np.mean(confidence)) - 0.65) < 1e-6


def test_detect_built_up():
    vv_db = np.array([[-20.0, -10.0, -5.0]], dtype=np.float32)

    mask, confidence = detect_built_up(vv_db)

    expected_mask = np.array([[False, True, True]])
    assert np.array_equal(mask, expected_mask)

    expected_confidence = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(confidence, expected_confidence, atol=1e-6)
