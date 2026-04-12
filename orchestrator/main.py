import os
import uuid
from typing import Any
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title="BB84 Orchestrator")

sessions:dict[str,dict[str,Any]]={}

def _post(url:str, payload:dict, label:str)->dict:
    try:
        r=requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{label} error: {e}")


class RunReq(BaseModel):
    size:int=Field(..., ge=1, le=10_000)


@app.get("/health")
def health():
    return {"statut": "ok", "sessions": len(sessions)}


@app.post("/run")
def run(req: RunReq):
    alice=_post(
        "http://localhost:8103/alice",
        {"size":req.size},
        "alice_service",
    )

    bob=_post(
        "http://localhost:8104/bob",
        {
            "alice_bits":alice["bits"],
            "alice_basis":alice["basis"],
            "size":req.size,
        },
        "bob_service",
    )

    sift=_post(
        "http://localhost:8102/sift",
        {
            "alice_bits":alice["bits"],
            "alice_basis":alice["basis"],
            "bob_basis":bob["bob_basis"],
            "bob_res":bob["bob_res"],
        },
        "sifting_service",
    )

    session_id=str(uuid.uuid4())
    sessions[session_id]={
        "size":req.size,
        "alice_bits":alice["bits"],
        "alice_basis":alice["basis"],
        "bob_basis":bob["bob_basis"],
        "bob_res":bob["bob_res"],
        "sifted_key":sift["sifted_key"],
    }

    return {
        "session_id":session_id,
        "alice_bits":alice["bits"],
        "alice_basis":alice["basis"],
        "bob_basis":bob["bob_basis"],
        "bob_res":bob["bob_res"],
        "sifted_key":sift["sifted_key"],
    }


@app.get("/sessions")
def list_sessions():
    return {"count":len(sessions), "session_ids": list(sessions.keys())}


@app.get("/sessions/{session_id}")
def get_session(session_id:str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]
