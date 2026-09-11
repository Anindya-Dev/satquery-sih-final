"""Bounding-box localization from binary masks."""

import numpy as np
from scipy import ndimage


def localize_mask(mask):
    """Return bounding boxes for each connected True region in a mask.

    Each box is a dict with ``x_min``, ``y_min``, ``x_max``, ``y_max``
    (inclusive pixel coordinates) and ``area_pixels``.
    """
    mask = np.asarray(mask, dtype=bool)
    labeled, num_labels = ndimage.label(mask)
    boxes = []
    for label in range(1, num_labels + 1):
        ys, xs = np.nonzero(labeled == label)
        boxes.append({
            "x_min": int(xs.min()),
            "y_min": int(ys.min()),
            "x_max": int(xs.max()),
            "y_max": int(ys.max()),
            "area_pixels": int(xs.size),
        })
    return boxes
