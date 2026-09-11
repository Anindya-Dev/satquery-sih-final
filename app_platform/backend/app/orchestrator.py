import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path for direct teammate imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app_platform.backend.app.vlm_synthesizer import VLMSynthesizer

# Member 1 Data Engine Loaders
from data_engine.loaders import (
    get_dataloader,
    get_member3_bitemporal_numpy,
    get_member3_optical_sar_pair,
)

# Member 2 Task Router Engine
from task_router.inference.router_engine import TaskRouterEngine

# Member 3 Vision Pipeline Adapters & Evidence Generator
from vision_pipeline.adapters import (
    bigearthnet_to_imagery,
    bitemporal_array_to_imagery,
    optical_sar_pair_to_imagery,
)
from vision_pipeline.evidence.evidence_generator import generate_evidence


class SystemOrchestrator:
    def __init__(self):
        self.router = TaskRouterEngine()
        self.synthesizer = VLMSynthesizer()

    async def run_pipeline(self, user_query: str, patch_id: str) -> Dict[str, Any]:
        # Step 1: Member 2 - Route user query to TaskSpec dictionary
        task_spec = self.router.route_dict(user_query)
        primary_tool = task_spec.get("primary_tool", "sar_flood_extractor")

        imagery = None
        unsupported = None

        # Step 2: Member 1 Data Engine -> Member 3 Adapters tool dispatch
        if primary_tool == "sar_flood_extractor":
            arr = get_member3_bitemporal_numpy(simulate_flood=True, sar_mode="raw_db")
            imagery = bitemporal_array_to_imagery(arr)

        elif primary_tool == "optical_sar_fusion_specialist":
            arr = get_member3_optical_sar_pair(sar_mode="raw_db")
            imagery = optical_sar_pair_to_imagery(arr)

        elif primary_tool in ("spectral_indices_calculator", "grounding_rs_specialist"):
            loader = get_dataloader(mode="optical", batch_size=1)
            batch = next(iter(loader))
            tensor = batch["image"][0]
            imagery = bigearthnet_to_imagery(tensor.detach().cpu().numpy())

        elif primary_tool == "bitemporal_change_detector":
            unsupported = (
                "bitemporal_change_detector needs t1/t2 optical bands, "
                "which Member 1's accessors don't provide yet"
            )

        else:
            unsupported = f"unsupported primary_tool: {primary_tool}"

        # Step 3: Member 3 - Vision Pipeline Evidence Generation
        if imagery is not None:
            evidence = generate_evidence(task_spec, imagery)
        else:
            evidence = {
                "target": task_spec.get("target", "unknown"),
                "status": "unsupported",
                "note": unsupported or "imagery unavailable",
            }

        # Step 4: Member 4 - VLM Grounded Synthesis
        synthesized_answer = await self.synthesizer.generate_grounded_answer(
            user_query=user_query,
            task_spec=task_spec,
            evidence=evidence,
        )

        return {
            "query": user_query,
            "patch_id": patch_id,
            "task_spec": task_spec,
            "evidence": evidence,
            "final_response": synthesized_answer,
        }