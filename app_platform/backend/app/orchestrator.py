from typing import Dict, Any
from app_platform.backend.app.vlm_synthesizer import VLMSynthesizer

# Import teammates' entry points directly from root
try:
    from task_router.inference.router_engine import TaskRouter
except ImportError:
    TaskRouter = None

try:
    from data_engine.loaders.bigearthnet_dataset import BigEarthNetLoader
except ImportError:
    BigEarthNetLoader = None

try:
    from vision_pipeline.evidence.evidence_generator import EvidenceGenerator
except ImportError:
    EvidenceGenerator = None


class SystemOrchestrator:
    def __init__(self):
        self.router = TaskRouter() if TaskRouter else None
        self.data_loader = BigEarthNetLoader() if BigEarthNetLoader else None
        self.evidence_gen = EvidenceGenerator() if EvidenceGenerator else None
        self.synthesizer = VLMSynthesizer()

    async def run_pipeline(self, user_query: str, patch_id: str) -> Dict[str, Any]:
        # Step 1: Member 2 - Task Router (Query -> JSON Task Spec)
        if self.router and hasattr(self.router, "parse_query"):
            task_spec = await self.router.parse_query(user_query)
        else:
            task_spec = {"target": "flood_detection", "modality": "SAR"}

        # Step 2: Member 1 - Data Engine (Load Patch Tensors)
        if self.data_loader and hasattr(self.data_loader, "get_patch"):
            tensors = self.data_loader.get_patch(patch_id=patch_id, modality=task_spec.get("modality"))
        else:
            tensors = {"status": "mock_tensor_loaded", "patch_id": patch_id}

        # Step 3: Member 3 - Vision Pipeline (Compute & Generate Evidence JSON)
        if self.evidence_gen and hasattr(self.evidence_gen, "execute_task"):
            evidence = self.evidence_gen.execute_task(task_spec=task_spec, tensors=tensors)
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