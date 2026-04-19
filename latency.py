import time
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional

app=FastAPI(title="Latency Service")

#in memory storage => session_id -> start timestamp
sessions={}

class StartReq(BaseModel):
    session_id:str
class StopReq(BaseModel):
    session_id:str

@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/start")
def start(req:StartReq):
    sessions[req.session_id]=time.monotonic()
    print(f"[latency_service]timer started for session{req.session_id}")
    return{"ok":True}


@app.post("/stop")
def stop(req:StopReq):
    start_time=sessions.pop(req.session_id,None)
    if start_time is None:
        return{"error":f",o timer found for session {req.session_id}"}
    elapsed=time.monotonic()-start_time
    print(f"[latency_service]session{req.session_id} elapsed:{elapsed:.4f}s")
    return{"session_id":req.session_id,"latency_seconds":round(elapsed, 4)}
