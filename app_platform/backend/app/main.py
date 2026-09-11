from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app_platform.backend.app.orchestrator import SystemOrchestrator

app = FastAPI(title="SatQuery Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = SystemOrchestrator()

class QueryPayload(BaseModel):
    query: str
    patch_id: str

@app.get("/health")
def health_check():
    return {"status": "online", "system": "SatQuery"}

@app.post("/api/v1/query")
async def handle_query(payload: QueryPayload):
    try:
        result = await orchestrator.run_pipeline(
            user_query=payload.query,
            patch_id=payload.patch_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))