import traceback
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from app_platform.backend.app.orchestrator import SystemOrchestrator

app = FastAPI(title="SatQuery SIH Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = SystemOrchestrator()

def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(i) for i in obj]
    return obj

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
@app.post("/api/v1/query")
async def process_query(payload: dict):
    try:
        user_query = payload.get("query") or payload.get("prompt") or ""
        patch_id = payload.get("patch_id") or payload.get("patch") or "Patch 001"
        result = await orchestrator.run_pipeline(user_query, patch_id)
        return jsonable_encoder(sanitize_for_json(result))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))