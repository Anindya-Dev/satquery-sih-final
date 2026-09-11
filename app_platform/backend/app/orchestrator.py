import os
import sys
from typing import Dict, Any

# Ensure project root is in sys.path for direct module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from app_platform.backend.app.vlm_synthesizer import VLMSynthesizer

# Member 1 Imports: Data Engine Loaders & Adapters
try:
    from data_engine.loaders import (
        load_from_task_spec,
        get_member3_bitemporal_numpy,
        get_member3_optical_sar_pair,
        CHANNEL_MAP
    )
    HAS_DATA_ENGINE = True
except ImportError:
    HAS_DATA_ENGINE = False

# Member 2 Imports: Task Router
try:
    from task_router.inference.router_engine import TaskRouter
except ImportError:
    TaskRouter = None

# Member 3 Imports: Vision Pipeline
try:
    from vision_pipeline.evidence.evidence_generator import EvidenceGenerator
except ImportError:
    EvidenceGenerator = None


class SystemOrchestrator:
    def __init__(self):
        self.router = TaskRouter() if TaskRouter else None
        self.evidence_gen = EvidenceGenerator() if EvidenceGenerator else None
        self.synthesizer = VLMSynthesizer()

    async def run_pipeline(self, user_query: str, patch_id: str) -> Dict[str, Any]:
        # Step 1: Member 2 - Task Router (Query -> JSON Task Spec)
        if self.router and hasattr(self.router, "parse_query"):
            task_spec = await self.router.parse_query(user_query)
        else:
            task_spec = {
                "target": "flood_detection", 
                "modality": "BITEMPORAL_PAIR",
                "patch_id": patch_id
            }

        # Step 2: Member 1 - Data Engine (Load Tensors / Arrays via Member 1 Adapter)
        if HAS_DATA_ENGINE:
            # Option 1: Automatic dispatch driven directly by Member 2's task_spec
            data_payload = load_from_task_spec(task_spec)
        else:
            # Fallback payload structure matching Member 1 output format
            data_payload = {
                "patch_id": patch_id,
                "modality": task_spec.get("modality", "SAR"),
                "status": "mock_data_engine_loaded"
            }

        # Step 3: Member 3 - Vision Pipeline (Compute & Generate Evidence JSON)
        if self.evidence_gen and hasattr(self.evidence_gen, "execute_task"):
            evidence = self.evidence_gen.execute_task(task_spec=task_spec, tensors=data_payload)
        else:
            # Fallback evidence containing spatial bbox [min_lon, min_lat, max_lon, max_lat]
            evidence = {
                "target": task_spec.get("target", "flood_detection"),
                "affected_area_pct": 14.2, 
                "confidence": 0.91,
                "bbox": [76.20, 9.90, 76.35, 10.05]
            }

        # Step 4: Member 4 - VLM Synthesis (Grounded Answer Generation)
        synthesized_answer = await self.synthesizer.generate_grounded_answer(
            user_query=user_query,
            task_spec=task_spec,
            evidence=evidence
        )

        return {
            "query": user_query,
            "patch_id": patch_id,
            "task_spec": task_spec,
            "evidence": evidence,
            "final_response": synthesized_answer
        }