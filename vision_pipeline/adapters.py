"""Adapters from Member 1's stacked tensors to Member 3's imagery dict."""

import numpy as np

BIGEARTHNET_OPTICAL_INDICES = {
    "NIR": 7,    # B08
    "RED": 3,    # B04
    "GREEN": 2,  # B03
    "BLUE": 1,   # B02
}


def bitemporal_array_to_imagery(data):
    """Convert a ``(6, H, W)`` float32 array into the imagery dict.

    Channel order (Member 1's written contract): NIR, RED, GREEN, BLUE,
    VV_pre, VV_post. SAR bands are expected in raw dB.

    The single ``VV`` key reuses ``VV_pre`` as a stand-in for the co-registered
    single-date SAR scene used by the fusion tool.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 6:
        raise ValueError("expected array of shape (6, H, W)")

    nir, red, green, blue, vv_pre, vv_post = data
    return {
        "optical": {"NIR": nir, "RED": red, "GREEN": green, "BLUE": blue},
        "sar": {"VV": vv_pre, "VV_pre": vv_pre, "VV_post": vv_post},
    }


def optical_sar_pair_to_imagery(data):
    """Convert a ``(5, H, W)`` float32 array into the imagery dict.

    Channel order (Member 1's written contract): NIR, RED, GREEN, BLUE, VV.
    SAR band is expected in raw dB.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 5:
        raise ValueError("expected array of shape (5, H, W)")

    nir, red, green, blue, vv = data
    return {
        "optical": {"NIR": nir, "RED": red, "GREEN": green, "BLUE": blue},
        "sar": {"VV": vv},
    }


def bigearthnet_to_imagery(data):
    """Convert Member 1's real BigEarthNet tensor to the imagery dict.

    Accepts a ``(12, H, W)`` optical tensor or a ``(14, H, W)`` fused tensor
    (12 Sentinel-2 bands + VV + VH). Maps B02/B03/B04/B08 to BLUE/GREEN/RED/
    NIR. For fused input, VV/VH are included as-is (they arrive normalized to
    [0, 1], not raw dB).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] not in (12, 14):
        raise ValueError("expected array of shape (12, H, W) or (14, H, W)")

    imagery = {
        "optical": {
            name: data[index]
            for name, index in BIGEARTHNET_OPTICAL_INDICES.items()
        }
    }
    if data.shape[0] == 14:
        imagery["sar"] = {"VV": data[12], "VH": data[13]}

    return imagery
bitemporal_optical_array_to_imagery = bitemporal_array_to_imagery
