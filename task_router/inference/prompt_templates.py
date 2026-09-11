"""
Prompt templates and few-shot exemplars for the SatQuery AI Task Router.
"""

SYSTEM_ROUTER_PROMPT = """You are SatQuery AI's Specialist Task Router.
Your objective is to interpret natural-language queries from non-expert users and determine the exact remote-sensing workflow.
Classify the task, identify required imagery modality (Optical, SAR, Cross-Modal Pair, or Bi-temporal Pair), select specialist models/tools from the registry, and configure permitted parameters.

Output ONLY valid JSON strictly matching the TaskSpec schema. Never output markdown code fences, conversational prose, or unverified parameters.
"""

FEW_SHOT_EXEMPLARS = [
    {
        "query": "Highlight the water body referred to in the query.",
        "output": {
            "task_type": "single_image_grounding",
            "modality": "OPTICAL",
            "sensors": ["Sentinel-2"],
            "primary_tool": "grounding_rs_specialist",
            "secondary_tools": ["spectral_indices_calculator"],
            "parameters": {
                "target_features": ["water"],
                "bands_required": ["B02", "B03", "B04", "B08"],
                "indices_requested": ["NDWI"],
                "cloud_penetration_needed": False,
                "temporal_comparison": False,
                "threshold_method": "fixed",
                "grounding_target": "water",
                "grounding_prompt": "Highlight the water body referred to in the query.",
                "vqa_question": None
            },
            "evidence_requested": ["bounding_boxes", "num_regions"],
            "confidence_threshold": 0.8,
            "audit_summary": "Routing query to grounding_rs_specialist for water localization."
        }
    },
    {
        "query": "Identify built-up grounding regions.",
        "output": {
            "task_type": "single_image_grounding",
            "modality": "OPTICAL",
            "sensors": ["Sentinel-2"],
            "primary_tool": "grounding_rs_specialist",
            "secondary_tools": [],
            "parameters": {
                "target_features": ["built_up"],
                "bands_required": ["B02", "B03", "B04", "B08"],
                "indices_requested": [],
                "cloud_penetration_needed": False,
                "temporal_comparison": False,
                "threshold_method": "fixed",
                "grounding_target": "built_up",
                "grounding_prompt": "Identify built-up grounding regions.",
                "vqa_question": None
            },
            "evidence_requested": ["bounding_boxes", "num_regions"],
            "confidence_threshold": 0.8,
            "audit_summary": "Routing query to grounding_rs_specialist for built-up localization."
        }
    },
    {
        "query": "Find flooded areas under clouds.",
        "output": {
            "task_type": "sar_flood_detection",
            "modality": "SAR",
            "sensors": ["Sentinel-1"],
            "primary_tool": "sar_flood_extractor",
            "secondary_tools": [],
            "parameters": {
                "target_features": ["standing_water", "inundation"],
                "bands_required": ["VV", "VH"],
                "indices_requested": [],
                "cloud_penetration_needed": True,
                "temporal_comparison": False,
                "threshold_method": "fixed",
                "grounding_target": None,
                "grounding_prompt": None,
                "vqa_question": None
            },
            "evidence_requested": ["inundation_mask", "flooded_area_km2", "confidence_mean"],
            "confidence_threshold": 0.85,
            "audit_summary": "Routing query to sar_flood_extractor using Sentinel-1 radar for cloud-penetrating flood detection."
        }
    },
    {
        "query": "What changed between these two dates, and where did the change occur?",
        "output": {
            "task_type": "bitemporal_change_detection",
            "modality": "BITEMPORAL_PAIR",
            "sensors": ["Sentinel-2"],
            "primary_tool": "bitemporal_change_detector",
            "secondary_tools": ["spectral_indices_calculator"],
            "parameters": {
                "target_features": ["change_area", "urban_expansion", "deforestation"],
                "bands_required": ["B02", "B03", "B04", "B08"],
                "indices_requested": ["NDVI"],
                "cloud_penetration_needed": False,
                "temporal_comparison": True,
                "threshold_method": "fixed",
                "grounding_target": None,
                "grounding_prompt": None,
                "vqa_question": None
            },
            "evidence_requested": ["change_mask", "changed_area_km2", "confidence_mean"],
            "confidence_threshold": 0.8,
            "audit_summary": "Routing bi-temporal query to bitemporal_change_detector to isolate land-use change between acquisitions."
        }
    },
    {
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "output": {
            "task_type": "cross_modal_fusion",
            "modality": "CROSS_MODAL_PAIR",
            "sensors": ["Sentinel-2", "Sentinel-1"],
            "primary_tool": "optical_sar_fusion_specialist",
            "secondary_tools": ["spectral_indices_calculator"],
            "parameters": {
                "target_features": ["built_up", "water"],
                "bands_required": ["B02", "B03", "B04", "B08", "VV", "VH"],
                "indices_requested": ["NDVI", "NDWI"],
                "cloud_penetration_needed": False,
                "temporal_comparison": False,
                "threshold_method": "fixed",
                "grounding_target": None,
                "grounding_prompt": None,
                "vqa_question": None
            },
            "evidence_requested": ["water_mask", "built_up_mask", "water_area_km2", "built_up_area_km2"],
            "confidence_threshold": 0.85,
            "audit_summary": "Routing query to optical_sar_fusion_specialist for cross-modal complementary feature extraction."
        }
    }
]


def format_router_prompt(query: str, input_metadata: dict = None) -> str:
    """Constructs prompt for the SLM router."""
    meta_info = ""
    if input_metadata:
        meta_info = f"\nInput Metadata Context: {input_metadata}\n"

    prompt = f"""{SYSTEM_ROUTER_PROMPT}

User Query: "{query}"{meta_info}

JSON Output:"""
    return prompt
