"""SAR backscatter conversion helpers."""

import numpy as np

EPSILON = 1e-6


def sar_linear_to_db(linear):
    """Convert linear backscatter intensity to dB: 10 * log10(linear + eps)."""
    linear = np.asarray(linear, dtype=np.float32)
    return 10.0 * np.log10(linear + EPSILON)


BUILT_UP_DB_THRESHOLD = -10.0
BUILT_UP_CONFIDENCE_SCALE = 5.0


def detect_built_up(vv_db, threshold=BUILT_UP_DB_THRESHOLD):
    """Flag built-up pixels with high backscatter.

    Returns ``(built_up_mask, confidence)``. Confidence rises with backscatter
    above the threshold and saturates at 1.0.
    """
    vv_db = np.asarray(vv_db, dtype=np.float32)
    built_up_mask = vv_db >= threshold
    confidence = np.clip(
        (vv_db - threshold) / BUILT_UP_CONFIDENCE_SCALE,
        0.0,
        1.0,
    )
    return built_up_mask, confidence
