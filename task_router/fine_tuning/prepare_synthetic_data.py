"""
Synthetic Dataset Generator for SatQuery AI Task Router.
Generates diverse natural language remote sensing queries paired with
strict TaskSpec JSON targets according to the SIH Problem Statement.
Produces training and validation splits in both Alpaca and ChatML/ShareGPT formats.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any
from pathlib import Path

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task_router.schemas.task_spec_models import (
    TaskType,
    ModalityRequirement,
    SensorType,
    SpecialistTool,
    TaskParameters,
    TaskSpec
)

# Deterministic seed for reproducibility
random.seed(42)

QUERY_TEMPLATES = [
    # 1. Single Image VQA
    {
        "task_type": TaskType.SINGLE_IMAGE_VQA,
        "modality": ModalityRequirement.OPTICAL,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.OPTICAL_VQA,
        "secondary_tools": [],
        "evidence": ["textual_answer", "confidence_score", "attention_map"],
        "queries": [
            "What type of land cover dominates the northern section of this scene?",
            "Is there an airport or runway visible in this satellite patch?",
            "How many distinct agricultural parcels can be observed here?",
            "Does this area contain dense residential housing or commercial zones?",
            "What is the predominant crop pattern visible in this multispectral tile?",
            "Can you identify any major roads or transportation networks passing through?",
            "Are there solar panel installations in the southern quadrant?",
            "Is this coastal region showing signs of tidal flats or mangroves?",
            "What kind of forest canopy is visible in this optical tile?",
            "Is there a river or canal cutting across the landscape?"
        ],
        "params": lambda q: TaskParameters(
            target_features=["land_cover", "infrastructure"],
            bands_required=["B02", "B03", "B04", "B08"],
            vqa_question=q
        )
    },
    # 2. Single Image Grounding
    {
        "task_type": TaskType.SINGLE_IMAGE_GROUNDING,
        "modality": ModalityRequirement.OPTICAL,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.REGION_GROUNDING,
        "secondary_tools": [SpecialistTool.INDICES_CALCULATOR],
        "evidence": ["bounding_boxes", "num_regions"],
        "queries": [
            "Highlight the water body referred to in the query.",
            "Locate and draw bounding boxes around all industrial storage tanks.",
            "Detect and segment the dense forest boundaries in this image.",
            "Find and pinpoint the commercial seaport docks.",
            "Highlight all solar panel arrays in this region.",
            "Pinpoint the bridges crossing over the river.",
            "Highlight the boundary of the agricultural irrigation pivot.",
            "Segment the urban built-up cluster in the center of the tile."
        ],
        "params": lambda q: TaskParameters(
            target_features=["water" if "water" in q.lower() or "river" in q.lower() else "vegetation"],
            bands_required=["B02", "B03", "B04", "B08"],
            threshold_method="fixed",
            grounding_target="water" if "water" in q.lower() or "river" in q.lower() else "vegetation",
            grounding_prompt=q
        )
    },
    # 3. Single Image Captioning / Scene Description
    {
        "task_type": TaskType.SINGLE_IMAGE_CAPTIONING,
        "modality": ModalityRequirement.OPTICAL,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.SCENE_CAPTIONER,
        "secondary_tools": [],
        "evidence": ["descriptive_caption", "key_attributes", "confidence_score"],
        "queries": [
            "Describe the land-cover and major objects visible in this image.",
            "Generate a comprehensive scene summary of this remote sensing observation.",
            "Provide a detailed descriptive breakdown of the terrain and structures.",
            "What does this satellite capture show in terms of natural and human-made features?",
            "Summarize the dominant geographic and environmental features present."
        ],
        "params": lambda q: TaskParameters(
            target_features=["terrain", "land_cover", "urban_features"],
            bands_required=["B02", "B03", "B04", "B08"],
            threshold_method="fixed"
        )
    },
    # 4. Bi-temporal Change Detection
    {
        "task_type": TaskType.BITEMPORAL_CHANGE_DETECTION,
        "modality": ModalityRequirement.BITEMPORAL_PAIR,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.BITEMPORAL_CHANGE_DETECTOR,
        "secondary_tools": [SpecialistTool.INDICES_CALCULATOR],
        "evidence": ["change_mask", "changed_area_km2", "confidence_mean"],
        "queries": [
            "What changed between these two dates, and where did the change occur?",
            "Detect all structural changes between the before and after acquisitions.",
            "Generate a spatial change mask showing new constructions between T1 and T2.",
            "Identify areas of deforestation or tree loss between these two image dates.",
            "Map the changes in vegetation density from the baseline date to the current date.",
            "Did urban expansion consume agricultural land between the two acquisitions?",
            "Highlight all surface water body contractions or expansions between these two dates."
        ],
        "params": lambda q: TaskParameters(
            target_features=["change_area", "urban_expansion", "deforestation"],
            temporal_comparison=True,
            threshold_method="fixed"
        )
    },
    # 5. Bi-temporal Change VQA
    {
        "task_type": TaskType.BITEMPORAL_CHANGE_VQA,
        "modality": ModalityRequirement.BITEMPORAL_PAIR,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.BITEMPORAL_CHANGE_DETECTOR,
        "secondary_tools": [SpecialistTool.INDICES_CALCULATOR],
        "evidence": ["change_mask", "changed_area_km2", "confidence_mean"],
        "queries": [
            "Has the built-up area increased, decreased, or remained unchanged?",
            "Did the forest boundary recede between these two observation dates?",
            "How much new infrastructure appeared in the eastern section since last year?",
            "Has the reservoir water level dropped significantly compared to the prior image?",
            "Was the agricultural field harvested or left fallow between these two passes?"
        ],
        "params": lambda q: TaskParameters(
            target_features=["built_up", "forest", "water_level"],
            temporal_comparison=True,
            threshold_method="fixed",
            vqa_question=q
        )
    },
    # 6. Cross-Modal Optical + SAR Fusion
    {
        "task_type": TaskType.CROSS_MODAL_FUSION,
        "modality": ModalityRequirement.CROSS_MODAL_PAIR,
        "sensors": [SensorType.SENTINEL_2_OPTICAL, SensorType.SENTINEL_1_SAR],
        "primary_tool": SpecialistTool.OPTICAL_SAR_FUSION,
        "secondary_tools": [SpecialistTool.INDICES_CALCULATOR],
        "evidence": ["water_mask", "built_up_mask", "water_area_km2", "built_up_area_km2"],
        "queries": [
            "Use the optical and SAR images together to identify built-up and water-covered regions.",
            "Fuse Sentinel-2 optical and Sentinel-1 SAR observations to delineate complex urban boundaries.",
            "Combine multispectral bands and dual-pol SAR backscatter to classify challenging wetland terrain.",
            "Extract complementary structural and spectral information from this co-registered optical-SAR pair.",
            "Use optical color context alongside radar double-bounce to detect buildings obscured by shadows."
        ],
        "params": lambda q: TaskParameters(
            target_features=["built_up", "water", "wetland"],
            bands_required=["B02", "B03", "B04", "B08", "VV", "VH"],
            indices_requested=["NDVI", "NDWI"],
            threshold_method="fixed"
        )
    },
    # 7. SAR Flood Detection (Cloud Penetration)
    {
        "task_type": TaskType.SAR_FLOOD_DETECTION,
        "modality": ModalityRequirement.SAR,
        "sensors": [SensorType.SENTINEL_1_SAR],
        "primary_tool": SpecialistTool.SAR_BACKSCATTER_DETECTOR,
        "secondary_tools": [],
        "evidence": ["inundation_mask", "flooded_area_km2", "confidence_mean"],
        "queries": [
            "Find flooded areas under clouds.",
            "Map standing floodwater using SAR backscatter because optical imagery is cloud covered.",
            "Identify inundated zones through heavy monsoon cloud cover using Sentinel-1 radar.",
            "Detect surface water submergence in this cloudy region using VV/VH polarizations.",
            "Is there standing water underneath the overcast cloud layer in this area?"
        ],
        "params": lambda q: TaskParameters(
            target_features=["standing_water", "inundation"],
            bands_required=["VV", "VH"],
            cloud_penetration_needed=True,
            threshold_method="fixed"
        )
    },
    # 8. Vegetation Health & Indices
    {
        "task_type": TaskType.VEGETATION_HEALTH_ANALYSIS,
        "modality": ModalityRequirement.OPTICAL,
        "sensors": [SensorType.SENTINEL_2_OPTICAL],
        "primary_tool": SpecialistTool.INDICES_CALCULATOR,
        "secondary_tools": [],
        "evidence": ["NDVI_map", "mean_NDVI", "confidence_mean"],
        "queries": [
            "Assess vegetation health and crop vigor across these farmlands using NDVI.",
            "Calculate normalized difference vegetation index to detect crop water stress.",
            "Analyze canopy chlorophyll content and identify drought stressed crops.",
            "Is the farmland in this scene experiencing healthy growth or crop stress?"
        ],
        "params": lambda q: TaskParameters(
            target_features=["vegetation", "canopy", "crops"],
            bands_required=["B04", "B08"],
            indices_requested=["NDVI", "NDWI"],
            grounding_target="vegetation",
            threshold_method="fixed"
        )
    }
]

SYSTEM_PROMPT = """You are SatQuery AI's Specialist Task Router.
Your job is to translate non-expert natural language queries into a strictly formatted TaskSpec JSON schema for downstream remote-sensing specialist pipelines.
Never output markdown fences or conversational explanations. Output ONLY valid JSON adhering to the TaskSpec schema."""


def build_sample(template_dict: Dict[str, Any], query_text: str) -> Dict[str, Any]:
    task_type = template_dict["task_type"]
    modality = template_dict["modality"]
    sensors = template_dict["sensors"]
    primary_tool = template_dict["primary_tool"]
    secondary_tools = template_dict["secondary_tools"]
    evidence = template_dict["evidence"]
    params = template_dict["params"](query_text)

    spec = TaskSpec(
        task_type=task_type,
        modality=modality,
        sensors=sensors,
        primary_tool=primary_tool,
        secondary_tools=secondary_tools,
        parameters=params,
        evidence_requested=evidence,
        confidence_threshold=0.80,
        audit_summary=f"Routing query '{query_text}' to {primary_tool.value} requiring {modality.value} modality."
    )

    spec_json = spec.model_dump_json(indent=2)

    # Return formats
    alpaca_format = {
        "instruction": SYSTEM_PROMPT,
        "input": query_text,
        "output": spec_json
    }

    chatml_format = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query_text},
            {"role": "assistant", "content": spec_json}
        ]
    }

    return {
        "alpaca": alpaca_format,
        "chatml": chatml_format,
        "spec_dict": spec.model_dump()
    }


def generate_dataset(output_dir: str = "task_router/fine_tuning/data", val_ratio: float = 0.2):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Synthesize variations by applying paraphrasing prefixes and suffixes
    paraphrase_prefixes = [
        "",
        "Please ",
        "Can you ",
        "System, ",
        "SatQuery, ",
        "I need you to ",
        "Quickly "
    ]

    for template in QUERY_TEMPLATES:
        base_queries = template["queries"]
        for base_q in base_queries:
            for prefix in paraphrase_prefixes:
                # Combine prefix with query
                if prefix and base_q[0].isupper():
                    combined_query = prefix + base_q[0].lower() + base_q[1:]
                else:
                    combined_query = prefix + base_q
                
                sample = build_sample(template, combined_query)
                all_samples.append(sample)

    # Shuffle dataset
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * (1.0 - val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Save Alpaca format
    with open(out_path / "train_alpaca.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s["alpaca"]) + "\n")

    with open(out_path / "val_alpaca.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s["alpaca"]) + "\n")

    # Save ChatML format
    with open(out_path / "train_chatml.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s["chatml"]) + "\n")

    with open(out_path / "val_chatml.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s["chatml"]) + "\n")

    print(f"[Dataset Generation Complete]")
    print(f"  Total Synthetic Samples: {len(all_samples)}")
    print(f"  Training Samples:        {len(train_samples)}")
    print(f"  Validation Samples:      {len(val_samples)}")
    print(f"  Saved to:                {out_path.resolve()}")

    return len(train_samples), len(val_samples)


if __name__ == "__main__":
    generate_dataset()
