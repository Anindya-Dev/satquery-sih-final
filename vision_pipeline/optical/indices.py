"""Deterministic optical spectral indices (NDVI, NDWI, EVI).

All functions accept single-band 2D arrays of shape (H, W) and cast them to
float32. Each returns a tuple ``(index, mean_confidence)`` where
``mean_confidence`` is the fraction of pixels whose denominator magnitude is
above ``DENOM_CONFIDENCE_THRESHOLD``.

NaN policy: NaN values propagate into the index output (standard NumPy
arithmetic). A NaN denominator is treated as untrustworthy, so it is excluded
from the valid-pixel count used to compute ``mean_confidence``.
"""

import numpy as np

EPSILON = 1e-6
DENOM_CONFIDENCE_THRESHOLD = 1e-3


def _as_float32(array):
    """Cast any array-like input to float32."""
    return np.asarray(array, dtype=np.float32)


def _mean_confidence(denominator):
    """Fraction of pixels whose denominator magnitude is trustworthy."""
    valid = np.abs(denominator) > DENOM_CONFIDENCE_THRESHOLD
    return float(np.count_nonzero(valid)) / valid.size


def compute_ndvi(nir, red):
    """Normalized Difference Vegetation Index: (NIR - RED) / (NIR + RED + eps)."""
    nir = _as_float32(nir)
    red = _as_float32(red)
    denominator = nir + red + EPSILON
    ndvi = (nir - red) / denominator
    return ndvi, _mean_confidence(denominator)


def compute_ndwi(nir, green):
    """Normalized Difference Water Index: (GREEN - NIR) / (GREEN + NIR + eps)."""
    nir = _as_float32(nir)
    green = _as_float32(green)
    denominator = green + nir + EPSILON
    ndwi = (green - nir) / denominator
    return ndwi, _mean_confidence(denominator)


def compute_evi(nir, red, blue, L=1.0):
    """Enhanced Vegetation Index.

    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + L + eps)
    """
    nir = _as_float32(nir)
    red = _as_float32(red)
    blue = _as_float32(blue)
    numerator = 2.5 * (nir - red)
    denominator = nir + 6.0 * red - 7.5 * blue + L + EPSILON
    evi = numerator / denominator
    return evi, _mean_confidence(denominator)
