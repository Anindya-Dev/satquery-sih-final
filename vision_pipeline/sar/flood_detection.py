"""Rule-based SAR flood detection."""

import numpy as np

FLOOD_DROP_DB_THRESHOLD = 3.0
FLOOD_CONFIDENCE_SCALE = 5.0


def flood_detection_rule_based(vv_pre_db, vv_post_db):
    """Detect flooded pixels from a drop in VV backscatter between two dates.

    A pixel is flooded when ``VV_pre - VV_post >= 3.0`` dB. Per-pixel confidence
    ramps linearly from 0 to 1 over a 0..5 dB drop and saturates at 1.0.
    Returns ``(flood_mask, confidence_map)``.
    """
    vv_pre_db = np.asarray(vv_pre_db, dtype=np.float32)
    vv_post_db = np.asarray(vv_post_db, dtype=np.float32)
    drop = vv_pre_db - vv_post_db
    flood_mask = drop >= FLOOD_DROP_DB_THRESHOLD
    confidence = np.minimum(1.0, drop / FLOOD_CONFIDENCE_SCALE)
    return flood_mask, confidence
