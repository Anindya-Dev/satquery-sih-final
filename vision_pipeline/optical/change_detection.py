"""Deterministic bi-temporal change detection."""

import numpy as np

from vision_pipeline.optical.indices import compute_ndvi

CHANGE_NDVI_THRESHOLD = 0.2
CHANGE_CONFIDENCE_SCALE = 0.4


def detect_change_ndvi(nir_t1, red_t1, nir_t2, red_t2):
    """Flag pixels where NDVI changed between two dates.

    Returns ``(change_mask, confidence)``. Confidence rises with the magnitude
    of the NDVI difference and saturates at 1.0.
    """
    ndvi_t1, _ = compute_ndvi(nir_t1, red_t1)
    ndvi_t2, _ = compute_ndvi(nir_t2, red_t2)
    delta = np.abs(ndvi_t1 - ndvi_t2)
    change_mask = delta >= CHANGE_NDVI_THRESHOLD
    confidence = np.minimum(1.0, delta / CHANGE_CONFIDENCE_SCALE)
    return change_mask, confidence
