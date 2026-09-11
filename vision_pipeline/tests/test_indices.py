import numpy as np

from vision_pipeline.optical.indices import compute_evi, compute_ndvi, compute_ndwi


def test_compute_ndvi_basic():
    red = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
    nir = np.array([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32)

    ndvi, mean_confidence = compute_ndvi(nir, red)

    expected = np.array([[0.999999, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(ndvi, expected, atol=1e-6)
    assert mean_confidence == 0.75


def test_compute_ndwi_basic():
    green = np.array([[0.5, 0.2], [0.0, 1.0]], dtype=np.float32)
    nir = np.array([[0.1, 0.2], [0.0, 0.5]], dtype=np.float32)

    ndwi, mean_confidence = compute_ndwi(nir, green)

    eps = 1e-6
    expected = np.array([
        [0.4 / (0.6 + eps), 0.0],
        [0.0, 0.5 / (1.5 + eps)],
    ], dtype=np.float64)
    np.testing.assert_allclose(ndwi, expected, atol=1e-6)
    assert mean_confidence == 0.75


def test_compute_evi_simple():
    blue = np.array([[0.0]], dtype=np.float32)
    red = np.array([[1.0]], dtype=np.float32)
    nir = np.array([[3.0]], dtype=np.float32)

    evi, mean_confidence = compute_evi(nir, red, blue)

    assert abs(float(evi[0, 0]) - 0.5) < 1e-6
    assert mean_confidence == 1.0


def test_denominator_edge_cases():
    red = np.array([[0.0]], dtype=np.float32)
    nir = np.array([[0.0]], dtype=np.float32)

    ndvi, mean_confidence = compute_ndvi(nir, red)

    assert float(ndvi[0, 0]) == 0.0
    assert mean_confidence == 0.0


def test_nan_handling():
    red = np.array([[np.nan]], dtype=np.float32)
    nir = np.array([[1.0]], dtype=np.float32)

    ndvi, mean_confidence = compute_ndvi(nir, red)

    assert np.isnan(ndvi[0, 0])
    assert mean_confidence == 0.0


def test_dtype_validation():
    red = np.array([[0, 1], [2, 0]])
    nir = np.array([[1, 1], [2, 0]])

    ndvi, _ = compute_ndvi(nir, red)

    assert ndvi.dtype == np.float32
