# Member 3 (GIS & Computer Vision Specialist) — Context / Handoff

## Role

Member 3 owns the **deterministic satellite pipeline**: pure-math GIS/CV
functions that turn aligned tensors into a strict Evidence JSON. No model
inference, no randomness — every result carries a confidence score and a
human-readable `math` string so the VLM layer never has to guess.

The pipeline is the middle step in the full flow:

```
user query -> Member 2 router (task_spec) -> Member 1 data (tensors)
           -> Member 3 (evidence JSON)    -> Member 4 VLM (answer)
```

## Current status (one line)

Deterministic library is functionally complete and tested (29 tests pass);
flood + fusion are verified end-to-end; change detection is blocked on Member 1
(bi-temporal optical missing); Member 4's orchestrator still needs the
corrected wiring applied.

## What's built

All code is under `vision_pipeline/`.

| Module | Purpose |
|---|---|
| `optical/indices.py` | NDVI, NDWI, EVI — each returns `(index, mean_confidence)` |
| `optical/cloud_mask.py` | Deterministic cloud mask (brightness threshold) |
| `optical/change_detection.py` | Bi-temporal change via `|NDVI(t1) - NDVI(t2)|` |
| `optical/grounding.py` | Bounding boxes from a binary mask (scipy `ndimage.label`) |
| `sar/backscatter.py` | Linear->dB conversion, built-up detection |
| `sar/flood_detection.py` | Flood rule: `(VV_pre - VV_post) >= 3.0 dB` |
| `models/unet_segmentor.py` | 3-level U-Net — **architecture only, not trained** |
| `evidence/evidence_generator.py` | The dispatcher: `generate_evidence(task_spec, imagery)` |
| `adapters.py` | Converts Member 1's tensors into the `imagery` dict |

Tests live in `vision_pipeline/tests/` (29 passing).

## The evidence contract

### Input

`task_spec` is a plain dict (from Member 2's `route_dict()`):

```python
{
    "primary_tool": "sar_flood_extractor",
    "parameters": {"indices_requested": ["NDVI"], "grounding_target": "water"},
    "evidence_requested": ["inundation_mask", "flooded_area_km2", "confidence_mean"],
}
```

`imagery` is a dict grouped by sensor:

```python
{
    "optical": {"NIR": ..., "RED": ..., "GREEN": ..., "BLUE": ...},
    "sar": {"VV": ..., "VV_pre": ..., "VV_post": ...},
}
```

For bi-temporal change, optical is nested: `{"optical": {"t1": {...}, "t2": {...}}}`.

### Output

A list of evidence entries:

```python
[{
    "tool": "sar_flood_extractor",
    "target": "flood",
    "results": {...},
    "math": "flood if (VV_pre - VV_post) >= 3.0 dB; ...",
}]
```

### Tool -> handler mapping

| `primary_tool` | What it computes | Data it needs |
|---|---|---|
| `sar_flood_extractor` | flood mask + `flooded_area_km2` | `sar.VV_pre`, `sar.VV_post` (raw dB) |
| `optical_sar_fusion_specialist` | water mask (NDWI) + built-up mask (VV) | `optical.NIR/GREEN`, `sar.VV` |
| `spectral_indices_calculator` | `{INDEX}_map` + `mean_{INDEX}` | `optical` bands |
| `grounding_rs_specialist` | `bounding_boxes` + `num_regions` | `optical` bands |
| `bitemporal_change_detector` | `change_mask` + `changed_area_km2` | `optical.t1`, `optical.t2` (**BLOCKED**) |

The other three `primary_tool` values (`optical_vqa_specialist`,
`captioning_rs_specialist`, `bitemporal_change_vqa_specialist`) are **Member 4's
VLM tools** — `generate_evidence` correctly returns `[]` for them.

### Key output keys (must match Member 2/4)

- `confidence_mean` — **top-level** in `results` (not nested under `summary`).
- `{INDEX}_map`, `mean_{INDEX}` — e.g. `NDVI_map`, `mean_NDVI` (case-preserving).
- `inundation_mask`, `flooded_area_km2`, `change_mask`, `changed_area_km2`
- `water_mask`, `built_up_mask`, `water_area_km2`, `built_up_area_km2`
- `bounding_boxes`, `num_regions`

## Integration state

### Member 1 (Data Engine) — branch `feat/data-engine`

Verified at commit `9ecbfe7`. Accessors exist and match our adapters:

- `get_member3_bitemporal_numpy(sar_mode="raw_db")` -> `(6,120,120)` float32,
  channels `[NIR, RED, GREEN, BLUE, VV_pre, VV_post]`.
- `get_member3_optical_sar_pair(sar_mode="raw_db")` -> `(5,120,120)` float32,
  channels `[NIR, RED, GREEN, BLUE, VV]`.

SAR in `raw_db` mode is reconstructed dB, clamped to `[-25, -5]`. `simulate_flood`
drops the center `[40:80, 40:80]` by `6 dB`, which fires our `3 dB` threshold.

**Still needed from Member 1:**
- A bi-temporal **optical** accessor (`t1`/`t2` NIR+RED) for the change detector.
- Fix `load_from_task_spec` to use `task_spec.get("modality")` (it currently uses
  `getattr`, so a plain dict always defaults to `CROSS_MODAL_PAIR`).

### Member 2 (Task Router) — branch `feat/member-2-task-router`

Verified at commits `3fd7124`/`d40e87b`. All aligned:

- `TaskRouterEngine` class, `route_dict()` returns a plain dict (solves the
  Pydantic-vs-dict boundary).
- NDBI dropped, few-shot exemplars harmonized, `threshold_method` defaults to
  `"fixed"`.

### Member 4 (Orchestrator) — branch `dev`

Their `orchestrator.py` had wrong imports (`TaskRouter` vs `TaskRouterEngine`,
`EvidenceGenerator` vs `generate_evidence`) and passed the wrong `imagery` shape.
I drafted a corrected `orchestrator.py` (see the relevant chat turn) that:

- uses `TaskRouterEngine` + module-level `generate_evidence`
- uses the right adapter per `primary_tool`
- uses `bigearthnet_to_imagery` for the optical/spectral/grounding path
- returns a clear "unsupported" note for `bitemporal_change_detector`

Flood + fusion are now correctly wired. The generic dispatch still needs the
corrected version applied by Member 4.

## Outstanding work / next steps

1. Member 4 applies the corrected `orchestrator.py` (drafted).
2. Member 1 adds the bi-temporal optical accessor (unblocks change detection).
3. Merge branches and run the real end-to-end test.
4. Train the U-Net and wire inference into evidence output.
5. Define the evidence -> VLM contract for Member 4's synthesizer.
6. `built_up` grounding via single-date SAR backscatter (now possible).

## Key constants & design decisions

- `EPSILON = 1e-6` (safe denominator), `DENOM_CONFIDENCE_THRESHOLD = 1e-3`.
- Flood: `FLOOD_DROP_DB_THRESHOLD = 3.0`, `FLOOD_CONFIDENCE_SCALE = 5.0`.
- Built-up: `BUILT_UP_DB_THRESHOLD = -10.0` (recalibrated for Member 1's
  `[-25, -5]` dB clamp — was `-5`, which sat at the clamp ceiling).
- Pixel area: `PIXEL_AREA_KM2 = 0.0001` (10 m grid).
- Vegetation grounding: NDVI `> 0.2`.

## Git / branches

- Working branch: `feat/gis-evidence-pipeline`.
- Member 1: `remotes/origin/feat/data-engine`.
- Member 2: `remotes/origin/feat/member-2-task-router`.
- Member 4: `remotes/origin/dev`.

Read teammates' code without merging via `git fetch` + `git show`:

```bash
git fetch origin <branch>
git show remotes/origin/<branch>:path/to/file.py
```

Do **not** merge the branches yet — the contracts are verified by reading, and
merging now would pull in mismatched versions.

## How to run

```bash
python -m pytest vision_pipeline/tests -q
```
