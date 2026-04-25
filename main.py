import asyncio
import logging
import random
import threading
import time
import os
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from celery import group, chain, chord
from celery.result import AsyncResult
from workers.qubit_tasks   import send_qubit_task
from workers.sifting_tasks import sifting_task, qber_key_task
from workers.celery_config import celery_app
from models import (
    Basis, new_session_id,
    SessionStartReq, SessionStartResp,
    NetworkInitReq,
    BB84Error, ErrorCode,)
from bb84_logic import (
    perform_sifting, compute_qber,
    QBER_THRESHOLD,)


logger=logging.getLogger("alice")
logging.basicConfig(level=logging.INFO)

QNS_URL=os.getenv("QNS_URL", "http://localhost:8003")
BOB_URL=os.getenv("BOB_URL", "http://localhost:8002")
HTTP_TIMEOUT=30.0

_sessions:dict[str,dict]={}
_lock=threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Alice] Service started")
    yield
    logger.info("[Alice] Service stopped")

 
app=FastAPI(
    title="Alice Service",
    description="Transmitter BB84",
    version="0.4.0",
    lifespan=lifespan,)

async def _http_post(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    try:
        resp=await client.post(url, **kwargs)
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Appel POST {url} -> {e.response.status_code}: "f"{e.response.text[:200]}",)


def _wait_for_result(task_id: str, timeout:float=180.0)-> dict:
    result=AsyncResult(task_id, app=celery_app)
    return result.get(timeout=timeout, propagate=True)


@app.post("/session/start", response_model=SessionStartResp)
async def start_session(req: SessionStartReq):
    session_id=new_session_id()
    n=req.n_qubits
    t_start=time.time()
 
    logger.info(f"[Alice] Starting session {session_id} ({n} qubits (Celery))")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        await _http_post(client, f"{QNS_URL}/network/init",
                    json=NetworkInitReq(session_id=session_id,
                        n_qubits=n,
                    ).model_dump())
        logger.info(f"[Alice] QNS initialized")

        await _http_post(client, f"{BOB_URL}/session/register",
                    params={"session_id": session_id})
        logger.info(f"[Alice] Bob registered")

        bits=[random.randint(0, 1) for _ in range(n)]
        bases=[random.choice(list(Basis)) for _ in range(n)]
        
        session_meta={"session_id": session_id, "n_qubits":n}
 
        qubit_group=group(
            send_qubit_task.s(
                session_id=session_id,
                qubit_id=i,
                bit=bits[i],
                basis=bases[i].value,
                ) for i in range(n)
            )
        pipeline=chord(qubit_group)(
            chain(
                sifting_task.s(session_meta=session_meta),
                qber_key_task.s(),
            )
        )
        logger.info(
            f"[Alice] Pipeline Celery started -- "
            f"task_id={pipeline.id}"
        )

    loop= asyncio.get_event_loop()
    try:
        final=await loop.run_in_executor(None, _wait_for_result, pipeline.id, 180.0)
    
    except Exception as e:
        logger.error(f"[Alice] Pipeline failed: {e}")
        # Cleanup best-effort
        await _stop(session_id)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed=time.time()-t_start
    logger.info(
        f"[Alice] Session terminated in {elapsed:.2f}s — "
        f"statut={final.get('statut')}"
    )
    
    await _stop(session_id)
    key = final.get("key_final", "")
    if isinstance(key, list):
        key= "".join(map(str, key))
    
    with _lock:
        _sessions[session_id]={**final, "elapsed":elapsed}
    statut = final.get("statut", "aborted")
    return SessionStartResp(
        session_id=session_id,
        statut=statut,
        n_qubits_sent=n,
        n_qubits_received=final.get("n_delivered", 0),
        n_sifted=final.get("n_sifted", 0),
        qber=final.get("qber", 1.0),
        key_final=key,
        latency=elapsed,
        error_message=final.get("error_message", ""),
    )
 

async def _stop(session_id: str):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{QNS_URL}/network/stop",
                json={"session_id": session_id},)
            await client.delete(
                f"{BOB_URL}/session/{session_id}",)
    except Exception as e:
        logger.warning(f"[Alice] stop partiel: {e}")

 
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    with _lock:
        session=_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, **session}
 
 
@app.get("/health")
async def health():
    #verif connexion Redis/Celery
    try:
        celery_app.control.inspect(timeout=1.0).active()
        celery_ok = True
    except Exception:
        celery_ok = False
 
    return {
        "statut": "ok",
        "celery": celery_ok,
        "qns_url": QNS_URL,
        "bob_url": BOB_URL,
    }

 


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
