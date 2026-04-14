import time
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app=FastAPI(title="Latency Service")

_starts: dict[str, float]={}

class EndReq(BaseModel):
    session_id: str

@app.get("/health")
def health():
    return {"statut": "ok", "active_sessions": len(_starts)}


@app.post("/start")
def start():
    session_id=str(uuid.uuid4())
    _starts[session_id]=time.perf_counter()
    return {"session_id":session_id}


@app.post("/end")
def end(req: EndReq):
    t0=_starts.pop(req.session_id, None)
    if t0 is None:
        raise HTTPException(status_code=404, detail="Session not found")
    latency=round(time.perf_counter()-t0, 6)
    return {"latency":latency}
