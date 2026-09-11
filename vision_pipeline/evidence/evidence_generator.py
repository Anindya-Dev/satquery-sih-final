"""Deterministic evidence generation for the GIS/CV pipeline."""

import numpy as np

from vision_pipeline.optical.change_detection import (
    CHANGE_CONFIDENCE_SCALE,
    CHANGE_NDVI_THRESHOLD,
    detect_change_ndvi,
)
from vision_pipeline.optical.grounding import localize_mask
from vision_pipeline.optical.indices import compute_evi, compute_ndvi, compute_ndwi
from vision_pipeline.sar.backscatter import BUILT_UP_DB_THRESHOLD, detect_built_up
from vision_pipeline.sar.flood_detection import (
    FLOOD_CONFIDENCE_SCALE,
    FLOOD_DROP_DB_THRESHOLD,
    flood_detection_rule_based,
)

PIXEL_RESOLUTION_M = 10.0
PIXEL_AREA_KM2 = (PIXEL_RESOLUTION_M ** 2) / 1e6  # 100 m2 -> 0.0001 km2
VEGETATION_NDVI_THRESHOLD = 0.2

INDEX_FUNCS = {
    "NDVI": ("vegetation", compute_ndvi, ("NIR", "RED")),
    "NDWI": ("water", compute_ndwi, ("NIR", "GREEN")),
    "EVI": ("vegetation", compute_evi, ("NIR", "RED", "BLUE")),
}

INDEX_MATH = {
    "NDVI": "NDVI = (NIR - RED) / (NIR + RED + eps)",
    "NDWI": "NDWI = (GREEN - NIR) / (GREEN + NIR + eps)",
    "EVI": "EVI = 2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1+eps)",
}


def _band(imagery, sensor, name):
    return np.asarray(imagery.get(sensor, {})[name], dtype=np.float32)


def _handle_sar_flood(task_spec, imagery):
    vv_pre = _band(imagery, "sar", "VV_pre")
    vv_post = _band(imagery, "sar", "VV_post")
    flood_mask, confidence = flood_detection_rule_based(vv_pre, vv_post)

    requested = task_spec.get("evidence_requested", [])
    results = {
        "inundation_mask": flood_mask,
        "confidence_mean": float(np.mean(confidence)),
    }
    if "flooded_area_km2" in requested:
        flooded = int(np.count_nonzero(flood_mask))
        results["flooded_area_km2"] = flooded * PIXEL_AREA_KM2

    return [{
        "tool": "sar_flood_extractor",
        "target": "flood",
        "results": results,
        "math": (
            f"flood if (VV_pre - VV_post) >= {FLOOD_DROP_DB_THRESHOLD} dB; "
            f"confidence = min(1.0,(VV_pre-VV_post)/{FLOOD_CONFIDENCE_SCALE})"
        ),
    }]


def _handle_change(task_spec, imagery):
    optical = imagery.get("optical", {})
    t1 = optical.get("t1", {})
    t2 = optical.get("t2", {})

    change_mask, confidence = detect_change_ndvi(
        np.asarray(t1["NIR"], dtype=np.float32),
        np.asarray(t1["RED"], dtype=np.float32),
        np.asarray(t2["NIR"], dtype=np.float32),
        np.asarray(t2["RED"], dtype=np.float32),
    )

    requested = task_spec.get("evidence_requested", [])
    results = {
        "change_mask": change_mask,
        "confidence_mean": float(np.mean(confidence)),
    }
    if "changed_area_km2" in requested:
        changed = int(np.count_nonzero(change_mask))
        results["changed_area_km2"] = changed * PIXEL_AREA_KM2

    return [{
        "tool": "bitemporal_change_detector",
        "target": "change",
        "results": results,
        "math": (
            f"change if |NDVI(t1)-NDVI(t2)| >= {CHANGE_NDVI_THRESHOLD}; "
            f"confidence = min(1.0,|delta|/{CHANGE_CONFIDENCE_SCALE})"
        ),
    }]


def _handle_spectral_indices(task_spec, imagery):
    requested = task_spec.get("parameters", {}).get("indices_requested", [])
    entries = []
    for name in requested:
        spec = INDEX_FUNCS.get(name)
        if spec is None:
            continue
        target, func, bands = spec
        args = [_band(imagery, "optical", b) for b in bands]
        index, confidence = func(*args)
        entries.append({
            "tool": "spectral_indices_calculator",
            "target": target,
            "results": {
                f"{name}_map": index,
                f"mean_{name}": float(np.nanmean(index)),
                "confidence_mean": confidence,
            },
            "math": INDEX_MATH[name],
        })
    return entries


def _handle_fusion(task_spec, imagery):
    ndwi, water_conf = compute_ndwi(
        _band(imagery, "optical", "NIR"),
        _band(imagery, "optical", "GREEN"),
    )
    water_mask = ndwi > 0.0

    vv_db = _band(imagery, "sar", "VV")
    built_up_mask, built_up_conf = detect_built_up(vv_db)

    requested = task_spec.get("evidence_requested", [])
    results = {
        "water_mask": water_mask,
        "built_up_mask": built_up_mask,
        "confidence_mean": (
            float(water_conf) + float(np.mean(built_up_conf))
        ) / 2.0,
    }
    if "water_area_km2" in requested:
        results["water_area_km2"] = int(np.count_nonzero(water_mask)) * PIXEL_AREA_KM2
    if "built_up_area_km2" in requested:
        results["built_up_area_km2"] = int(np.count_nonzero(built_up_mask)) * PIXEL_AREA_KM2

    return [{
        "tool": "optical_sar_fusion_specialist",
        "target": "built_up_and_water",
        "results": results,
        "math": f"water if NDWI > 0; built-up if VV_dB >= {BUILT_UP_DB_THRESHOLD} dB",
    }]


def _handle_grounding(task_spec, imagery):
    params = task_spec.get("parameters", {})
    target = params.get("grounding_target", "water")
    if target == "water":
        ndwi, _ = compute_ndwi(
            _band(imagery, "optical", "NIR"),
            _band(imagery, "optical", "GREEN"),
        )
        mask = ndwi > 0.0
        rule = "NDWI > 0"
    elif target == "vegetation":
        ndvi, _ = compute_ndvi(
            _band(imagery, "optical", "NIR"),
            _band(imagery, "optical", "RED"),
        )
        mask = ndvi > VEGETATION_NDVI_THRESHOLD
        rule = f"NDVI > {VEGETATION_NDVI_THRESHOLD}"
    else:
        return []

    boxes = localize_mask(mask)
    return [{
        "tool": "grounding_rs_specialist",
        "target": target,
        "results": {
            "bounding_boxes": boxes,
            "num_regions": len(boxes),
        },
        "math": f"bounding boxes of connected regions where {rule}",
    }]


TOOL_HANDLERS = {
    "sar_flood_extractor": _handle_sar_flood,
    "bitemporal_change_detector": _handle_change,
    "spectral_indices_calculator": _handle_spectral_indices,
    "optical_sar_fusion_specialist": _handle_fusion,
    "grounding_rs_specialist": _handle_grounding,
}


def generate_evidence(task_spec, imagery):
    """Run the deterministic tool named in ``task_spec["primary_tool"]``.

    ``imagery`` holds aligned band tensors grouped by sensor (``optical`` and
    ``sar``). ``task_spec`` may carry ``parameters`` and ``evidence_requested``
    to control which outputs are computed. Unknown tools return an empty list.
    """
    tool = task_spec.get("primary_tool")
    handler = TOOL_HANDLERS.get(tool)
    if handler is None:
        return []
    return handler(task_spec, imagery)
