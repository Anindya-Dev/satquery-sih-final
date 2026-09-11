"""
Task Specification Models for SatQuery AI.
Defines strict Pydantic schemas for mapping natural language queries
to actionable, auditable execution specifications for Member 1 and Member 3.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "single_image_vqa"
    SINGLE_IMAGE_GROUNDING = "single_image_grounding"
    SINGLE_IMAGE_CAPTIONING = "single_image_captioning"
    BITEMPORAL_CHANGE_DETECTION = "bitemporal_change_detection"
    BITEMPORAL_CHANGE_VQA = "bitemporal_change_vqa"
    CROSS_MODAL_FUSION = "cross_modal_fusion"
    SAR_FLOOD_DETECTION = "sar_flood_detection"
    VEGETATION_HEALTH_ANALYSIS = "vegetation_health_analysis"
    URBAN_BUILTUP_MAPPING = "urban_builtup_mapping"


class ModalityRequirement(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    CROSS_MODAL_PAIR = "CROSS_MODAL_PAIR"
    BITEMPORAL_PAIR = "BITEMPORAL_PAIR"


class SensorType(str, Enum):
    SENTINEL_2_OPTICAL = "Sentinel-2"
    SENTINEL_1_SAR = "Sentinel-1"
    CARTOSAT_OPTICAL = "Cartosat-2S"
    RISAT_SAR = "RISAT"
    GENERIC = "Generic"


class SpecialistTool(str, Enum):
    OPTICAL_VQA = "optical_vqa_specialist"
    REGION_GROUNDING = "grounding_rs_specialist"
    SCENE_CAPTIONER = "captioning_rs_specialist"
    BITEMPORAL_CHANGE_DETECTOR = "bitemporal_change_detector"
    BITEMPORAL_CHANGE_VQA = "bitemporal_change_vqa_specialist"
    OPTICAL_SAR_FUSION = "optical_sar_fusion_specialist"
    SAR_BACKSCATTER_DETECTOR = "sar_flood_extractor"
    INDICES_CALCULATOR = "spectral_indices_calculator"


class TaskParameters(BaseModel):
    target_features: List[str] = Field(
        default_factory=list,
        description="Target objects or landcover types (e.g. ['water', 'built-up', 'vegetation', 'flood'])."
    )
    bands_required: List[str] = Field(
        default_factory=list,
        description="Specific spectral or SAR bands required by downstream dataloader (e.g. ['VV', 'VH'] or ['B04', 'B08'])."
    )
    indices_requested: List[str] = Field(
        default_factory=list,
        description="Remote sensing indices to compute (e.g. ['NDVI', 'NDWI', 'NDBI'])."
    )
    cloud_penetration_needed: bool = Field(
        default=False,
        description="Whether SAR is required due to cloud occlusion or night conditions."
    )
    temporal_comparison: bool = Field(
        default=False,
        description="True if query compares two points in time (bi-temporal change)."
    )
    threshold_method: Optional[str] = Field(
        default="otsu",
        description="Thresholding algorithm for masks (e.g. 'otsu', 'fixed', 'adaptive')."
    )
    grounding_target: Optional[str] = Field(
        default=None,
        description="Target feature to localize for grounding_rs_specialist (e.g. 'water', 'built_up', 'vegetation')."
    )
    grounding_prompt: Optional[str] = Field(
        default=None,
        description="Text prompt if task involves locating/bounding specific regions."
    )
    vqa_question: Optional[str] = Field(
        default=None,
        description="Precise visual question to be answered by the specialist VQA model."
    )


class TaskSpec(BaseModel):
    """
    Root task specification output by Member 2's Router.
    Guarantees strict type safety and zero-hallucination downstream parameterization.
    """
    task_type: TaskType = Field(
        ...,
        description="The primary remote-sensing task identified from the natural language query."
    )
    modality: ModalityRequirement = Field(
        ...,
        description="Required image input modality configuration."
    )
    sensors: List[SensorType] = Field(
        default_factory=lambda: [SensorType.GENERIC],
        description="Sensors expected or recommended for the task."
    )
    primary_tool: SpecialistTool = Field(
        ...,
        description="The primary specialist model or GIS tool from the registry to execute."
    )
    secondary_tools: List[SpecialistTool] = Field(
        default_factory=list,
        description="Optional auxiliary tools (e.g., indices calculator or cloud mask)."
    )
    parameters: TaskParameters = Field(
        default_factory=TaskParameters,
        description="Algorithm and model parameters."
    )
    evidence_requested: List[str] = Field(
        default_factory=lambda: ["summary_text", "confidence_score"],
        description="Visual and numerical evidence to extract (e.g. ['change_mask', 'confidence_score', 'area_km2'])."
    )
    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required for definitive assertions."
    )
    audit_summary: str = Field(
        ...,
        description="Auditable human-readable execution summary (task, selected tool, reasoning)."
    )

    @field_validator("modality")
    @classmethod
    def validate_modality_for_temporal_change(cls, v: ModalityRequirement, info):
        # Bi-temporal change detection requires BITEMPORAL_PAIR modality
        task = info.data.get("task_type")
        if task in (TaskType.BITEMPORAL_CHANGE_DETECTION, TaskType.BITEMPORAL_CHANGE_VQA):
            if v != ModalityRequirement.BITEMPORAL_PAIR:
                return ModalityRequirement.BITEMPORAL_PAIR
        return v
