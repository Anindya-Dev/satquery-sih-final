"""Deterministic cloud masking for optical imagery."""

import numpy as np

CLOUD_BRIGHTNESS_THRESHOLD = 0.6
CLOUD_CONFIDENCE_SCALE = 0.2


def mask_clouds(red, green, blue):
    """Flag bright pixels as cloud using mean visible reflectance.

    Returns ``(cloud_mask, confidence)``. Confidence rises with distance from
    the brightness threshold and saturates at 1.0.
    """
    red = np.asarray(red, dtype=np.float32)
    green = np.asarray(green, dtype=np.float32)
    blue = np.asarray(blue, dtype=np.float32)
    brightness = (red + green + blue) / 3.0
    cloud_mask = brightness >= CLOUD_BRIGHTNESS_THRESHOLD
    confidence = np.minimum(
        1.0,
        np.abs(brightness - CLOUD_BRIGHTNESS_THRESHOLD) / CLOUD_CONFIDENCE_SCALE,
    )
    return cloud_mask, confidence
