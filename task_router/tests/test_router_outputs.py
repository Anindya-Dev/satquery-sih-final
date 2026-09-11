"""
Unit tests for SatQuery AI Task Router.
Validates that representative user queries correctly map to strict TaskSpec
schemas with the appropriate specialist tools, sensor modalities, and parameters.
"""

import pytest
from task_router.schemas.task_spec_models import (
    TaskType,
    ModalityRequirement,
    SpecialistTool,
    TaskSpec
)
from task_router.inference.router_engine import TaskRouterEngine


@pytest.fixture
def router():
    return TaskRouterEngine()


def test_sar_flood_detection_routing(router):
    """SIH Case: Cloud cover requires SAR radar for flood mapping."""
    query = "Find flooded areas under clouds."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.SAR_FLOOD_DETECTION
    assert spec.modality == ModalityRequirement.SAR
    assert spec.primary_tool == SpecialistTool.SAR_BACKSCATTER_DETECTOR
    assert spec.parameters.cloud_penetration_needed is True
    assert "VV" in spec.parameters.bands_required or "VH" in spec.parameters.bands_required
    assert "inundation_mask" in spec.evidence_requested


def test_bitemporal_change_detection_routing(router):
    """SIH Case: Two acquisitions to detect land cover / structural change."""
    query = "Detect all structural changes between the before and after acquisitions."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.BITEMPORAL_CHANGE_DETECTION
    assert spec.modality == ModalityRequirement.BITEMPORAL_PAIR
    assert spec.primary_tool == SpecialistTool.BITEMPORAL_CHANGE_DETECTOR
    assert spec.parameters.temporal_comparison is True
    assert "change_mask" in spec.evidence_requested
    assert "changed_area_km2" in spec.evidence_requested


def test_bitemporal_change_vqa_routing(router):
    """SIH Case: Bi-temporal question answering (CDVQA benchmark)."""
    query = "Has the built-up area increased, decreased, or remained unchanged?"
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.BITEMPORAL_CHANGE_VQA
    assert spec.modality == ModalityRequirement.BITEMPORAL_PAIR
    # Routes to Member 3's supported bitemporal_change_detector
    assert spec.primary_tool == SpecialistTool.BITEMPORAL_CHANGE_DETECTOR
    assert spec.parameters.temporal_comparison is True
    assert spec.parameters.vqa_question is not None
    assert "change_mask" in spec.evidence_requested


def test_cross_modal_fusion_routing(router):
    """SIH Case: Joint Optical and SAR feature extraction."""
    query = "Use the optical and SAR images together to identify built-up and water-covered regions."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.CROSS_MODAL_FUSION
    assert spec.modality == ModalityRequirement.CROSS_MODAL_PAIR
    assert spec.primary_tool == SpecialistTool.OPTICAL_SAR_FUSION
    assert "water_mask" in spec.evidence_requested
    assert "built_up_mask" in spec.evidence_requested


def test_single_image_grounding_routing(router):
    """SIH Case: Text-guided region grounding / localization."""
    query = "Highlight the water body referred to in the query."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.SINGLE_IMAGE_GROUNDING
    assert spec.modality == ModalityRequirement.OPTICAL
    assert spec.primary_tool == SpecialistTool.REGION_GROUNDING
    assert spec.parameters.grounding_target == "water"
    assert "bounding_boxes" in spec.evidence_requested
    assert "num_regions" in spec.evidence_requested


def test_builtup_and_vegetation_grounding_routing(router):
    """SIH Case: Grounding built-up and vegetation regions."""
    builtup_query = "Identify built-up grounding regions."
    spec_builtup = router.route(builtup_query)
    assert spec_builtup.primary_tool == SpecialistTool.REGION_GROUNDING
    assert spec_builtup.parameters.grounding_target == "built_up"
    assert "bounding_boxes" in spec_builtup.evidence_requested

    veg_query = "Locate and ground vegetation zones."
    spec_veg = router.route(veg_query)
    assert spec_veg.primary_tool == SpecialistTool.REGION_GROUNDING
    assert spec_veg.parameters.grounding_target == "vegetation"
    assert "bounding_boxes" in spec_veg.evidence_requested


def test_single_image_captioning_routing(router):
    """SIH Case: Scene description / captioning."""
    query = "Describe the land-cover and major objects visible in this image."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.SINGLE_IMAGE_CAPTIONING
    assert spec.modality == ModalityRequirement.OPTICAL
    assert spec.primary_tool == SpecialistTool.SCENE_CAPTIONER
    assert "descriptive_caption" in spec.evidence_requested


def test_vegetation_health_routing(router):
    """SIH Case: Spectral index calculation (NDVI/NDWI)."""
    query = "Assess vegetation health and crop vigor across these farmlands using NDVI."
    spec = router.route(query)

    assert isinstance(spec, TaskSpec)
    assert spec.task_type == TaskType.VEGETATION_HEALTH_ANALYSIS
    assert spec.modality == ModalityRequirement.OPTICAL
    assert spec.primary_tool == SpecialistTool.INDICES_CALCULATOR
    assert "NDVI" in spec.parameters.indices_requested


def test_json_serialization_and_audit_trace(router):
    """Ensures output strictly serializes to JSON with required audit metadata."""
    query = "Find flooded areas under clouds."
    spec = router.route(query)
    json_str = spec.model_dump_json()

    assert isinstance(json_str, str)
    assert spec.audit_summary is not None
    assert len(spec.audit_summary) > 10
    assert 0.0 <= spec.confidence_threshold <= 1.0
