import os
import logging
import threading
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse 
from models import (
    Basis,
    SiftReq, SiftResp,
    BobSessionState,
    QubitMeasurement,
    BB84Error, ErrorCode,
)
 
logger=logging.getLogger("bob")
logging.basicConfig(level=logging.INFO)
QNS_URL=os.getenv("QNS_URL", "http://localhost:8003")
_sessions: dict[str, BobSessionState]={}
_lock=threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Bob] Service started")
    yield
    logger.info("[Bob] Service stopped")
 
 
app=FastAPI(
    title="Bob Service",
    description="Receiver BB84",
    lifespan=lifespan,
)
@app.post("/session/register")
async def register_session(session_id: str):
    with _lock:
        if session_id in _sessions:
            return {"statut": "already_registered", "session_id": session_id}
        _sessions[session_id]=BobSessionState(session_id=session_id)
 
    logger.info(f"[Bob] Session {session_id} registered")
    return {"statut": "ready", "session_id": session_id}

@app.post("/sift", response_model=SiftResp)
async def sift(req: SiftReq):
    with _lock:
        session=_sessions.get(req.session_id)
 
    if not session:
        raise HTTPException(
            status_code=404,
            detail=BB84Error(
                code=ErrorCode.SESSION_NOT_FOUND,
                session_id=req.session_id,
                message="Session non registered",
            ).model_dump(),
        )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{QNS_URL}/measurements/{req.session_id}"
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=BB84Error(
                code=ErrorCode.NETWORK_UNAVAILABLE,
                message=f"Impossible to get the measurements QNS: {e}",
                session_id=req.session_id,
            ).model_dump(),
        )
 
    meas_data=resp.json()
    measurements=[QubitMeasurement(**m) for m in meas_data["measurements"]]
    alice_bases_map:dict[int, Basis]={qid: basis for qid, basis in req.alice_bases}
    bob_sifted_bits=[]
    bob_bases_for_alice: list[tuple[int, Basis]]=[]
 
    for m in measurements:
        bob_bases_for_alice.append((m.qubit_id, m.basis))
        if m.qubit_id in alice_bases_map and alice_bases_map[m.qubit_id]==m.basis:
            bob_sifted_bits.append(m.bit_res)
    with _lock:
        session.measurements=measurements
        session.sifted_bits=bob_sifted_bits
        session.statut="sifted"
    logger.info(
        f"[Bob] Session {req.session_id} sifted: "
        f"{len(bob_sifted_bits)} bits retained"
    )
    return SiftResp(
        session_id=req.session_id,
        bob_bases=bob_bases_for_alice,
        n_sifted=len(bob_sifted_bits),
    )

@app.get("/session/{session_id}/sifted-bits")
async def get_sifted_bits(session_id: str):
    with _lock:
        session=_sessions.get(session_id)
 
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable")
 
    if session.statut!="sifted":
        raise HTTPException(
            status_code=409,
            detail="Sifting hasn't been done yet",)
    return {
        "session_id":session_id,
        "sifted_bits":session.sifted_bits,
        "n_sifted":len(session.sifted_bits),}

@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    with _lock:
        session=_sessions.pop(session_id, None)
 
    if session:
        logger.info(f"[Bob] Session {session_id} cleaned")
        return {"statut": "cleaned", "session_id": session_id}
    return {"statut": "not_found"}

 
@app.get("/health")
async def health():
    with _lock:
        active=list(_sessions.keys())
    return {"statut": "ok", "active_sessions": active}
 
 
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0", port=8002, log_level="warning")