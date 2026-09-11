import numpy as np

from vision_pipeline.optical.change_detection import detect_change_ndvi


def test_detect_change_ndvi():
    nir_t1 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    red_t1 = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    nir_t2 = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    red_t2 = np.array([[0.0, 0.0], [0.0, 0.5]], dtype=np.float32)

    change_mask, confidence = detect_change_ndvi(nir_t1, red_t1, nir_t2, red_t2)

    expected_mask = np.array([[False, True], [True, True]])
    assert np.array_equal(change_mask, expected_mask)

    expected_confidence = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(confidence, expected_confidence, atol=1e-6)
