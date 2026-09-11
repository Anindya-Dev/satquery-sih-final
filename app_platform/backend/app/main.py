import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app_platform.backend.app.orchestrator import SystemOrchestrator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = SystemOrchestrator()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query")
async def process_query(payload: dict):
    try:
        user_query = payload.get("query") or payload.get("prompt") or ""
        patch_id = payload.get("patch_id") or payload.get("patch") or "Patch 001"
        result = await orchestrator.run_pipeline(user_query, patch_id)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail={"error_type": type(e).__name__, "message": str(e)}
        )