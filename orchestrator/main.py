import uuid
from typing import Any
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app=FastAPI(title="BB84 Orchestrator")

sessions:dict[str,dict[str,Any]]={}

def _post(url:str, payload:dict, label:str, timeout:int=120)->dict:
    try:
        r=requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"{label} error: {e}")

class RunReq(BaseModel):
    size:int=Field(..., ge=1, le=500, description="Number of qubits(QuNetSim capacity)")

@app.get("/health")
def health():
    return {"statut": "ok", "sessions": len(sessions)}

@app.post("/run")
def run(req: RunReq):
    start=_post("http://localhost:8108/start", {}, "latency_service")
    latency_session=start["session_id"]
    alice=_post( "http://localhost:8103/alice", {"size":req.size}, "alice_service", timeout=120, )
    bob=_post( "http://localhost:8104/bob", { "alice_bits":alice["bits"], "alice_basis":alice["basis"], "size":req.size, }, "bob_service", timeout=120, )
    sift=_post( "http://localhost:8102/sift", { "alice_bits":alice["bits"], "alice_basis":alice["basis"], "bob_basis":bob["bob_basis"], "bob_res":bob["bob_res"], }, "sifting_service", )
    if len(sift["sifted_key"])==0:
        raise HTTPException(status_code=400, detail="no shared bases. Try re-running protocol")
    qber=_post("http://localhost:8106/qber", {"alice_sifted":sift["alice_sifted"], "bob_sifted":sift["bob_sifted"],}, "qber_service",)
    lat_end = _post( "http://localhost:8108/end", {"session_id": latency_session}, "latency_service", )
    latency = lat_end["latency"]
    key_len = len(sift["sifted_key"])
    keyrate = _post( "http://localhost:8107/keyrate", {"key_length": key_len, "time_seconds": latency if latency > 0 else 1e-6}, "keyrate_service", )
    session_id=str(uuid.uuid4())
    sessions[session_id]={ "size":req.size, "alice_bits":alice["bits"], "alice_basis":alice["basis"], "bob_basis":bob["bob_basis"], "bob_res":bob["bob_res"], "sifted_key":sift["sifted_key"], "qber": qber["qber"], "key_rate": keyrate["key_rate"], "latency": latency, }

    return {
        "session_id":session_id,
        "alice_bits":alice["bits"],
        "alice_basis":alice["basis"],
        "bob_basis":bob["bob_basis"],
        "bob_res":bob["bob_res"],
        "sifted_key":sift["sifted_key"],
        "qber": qber["qber"],
        "key_raste": keyrate["key_rate"],
        "latency": latency,
        }

@app.get("/sessions")
def list_sessions():
    return {"count":len(sessions), "session_ids": list(sessions.keys())}

@app.get("/sessions/{session_id}")
def get_session(session_id:str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]
