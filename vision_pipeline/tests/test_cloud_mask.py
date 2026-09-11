import numpy as np

from vision_pipeline.optical.cloud_mask import mask_clouds


def test_mask_clouds_basic():
    red = np.array([[0.9, 0.65], [0.5, 0.2]], dtype=np.float32)
    green = np.array([[0.9, 0.65], [0.5, 0.2]], dtype=np.float32)
    blue = np.array([[0.9, 0.65], [0.5, 0.2]], dtype=np.float32)

    cloud_mask, confidence = mask_clouds(red, green, blue)

    expected_mask = np.array([[True, True], [False, False]])
    assert np.array_equal(cloud_mask, expected_mask)

    expected_confidence = np.array([[1.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(confidence, expected_confidence, atol=1e-6)
