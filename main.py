import asyncio
import logging
import random
import threading
import time
import os
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from models import (
    Basis, new_session_id,
    SessionStartReq, SessionStartResp,
    NetworkInitReq,
    SendQubitReq,
    SiftReq,
    BB84Error, ErrorCode,)
from bb84_logic import (
    perform_sifting, compute_qber,
    QBER_THRESHOLD,)


logger=logging.getLogger("alice")
logging.basicConfig(level=logging.INFO)

QNS_URL=os.getenv("QNS_URL", "http://localhost:8003")
BOB_URL=os.getenv("BOB_URL", "http://localhost:8002")
QUBIT_SEND_TIMEOUT=10.0
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
    lifespan=lifespan,)

async def _call(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    try:
        resp=await getattr(client, method)(url, **kwargs)
        resp.raise_for_status()
        return resp
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Appel {method.upper()} {url} -> {e.response.status_code}: "f"{e.response.text[:200]}",)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service indisponible ({url}): {e}",)
    
@app.post("/session/start", response_model=SessionStartResp)
async def start_session(req: SessionStartReq):
    session_id=new_session_id()
    n=req.n_qubits
    t_start=time.time()
 
    logger.info(f"[Alice] Starting session {session_id} ({n} qubits)")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        await _call(client, "post", f"{QNS_URL}/network/init",
                    json=NetworkInitReq(session_id=session_id,
                        n_qubits=n,
                    ).model_dump())
        logger.info(f"[Alice] QNS initialized")
        await _call(client, "post", f"{BOB_URL}/session/register",
                    params={"session_id": session_id})
        logger.info(f"[Alice] Bob registered")
        bits=[random.randint(0, 1) for _ in range(n)]
        bases=[random.choice(list(Basis)) for _ in range(n)]
        n_delivered=0
        t_send=time.time()
 
        for i in range(n):
            resp=await _call(
                client, "post", f"{QNS_URL}/qubit/send",
                json=SendQubitReq(session_id=session_id,
                    qubit_id=i,
                    bit=bits[i],
                    basis=bases[i],
                ).model_dump(),)
            if resp.json().get("delivered"):
                n_delivered+=1
 
        t_send_elapsed=time.time()-t_send
        logger.info(
            f"[Alice] {n_delivered}/{n} qubits delivered "f"en {t_send_elapsed:.2f}s") 
        
        alice_bases_payload=[(i, bases[i]) for i in range(n)]
 
        sift_resp=await _call(
            client, "post", f"{BOB_URL}/sift",
            json=SiftReq(session_id=session_id,
                alice_bases=alice_bases_payload,
            ).model_dump(),
        )
        sift_data=sift_resp.json()
        bob_bases=sift_data["bob_bases"]
 
        bob_bits_resp=await _call(
            client, "get",f"{BOB_URL}/session/{session_id}/sifted-bits",)
        bob_sifted=bob_bits_resp.json()["sifted_bits"]
 
        bob_bases_map={qid: Basis(b) for qid, b in bob_bases}
        alice_sifted=[bits[qid]
            for qid, bob_basis in sorted(bob_bases_map.items())
            if qid < len(bases) and bases[qid]==bob_basis]
        
        n_sifted=len(alice_sifted)
        logger.info(f"[Alice] Sifting : {n_sifted} bits retained")
    
        if n_sifted < 10:
            await _stop(client, session_id)
            return SessionStartResp(
                session_id=session_id,
                statut="aborted",
                n_qubits_sent=n,
                n_qubits_received=n_delivered,
                n_sifted=n_sifted,
                qber=1.0,
                key_final="",
                error_message=ErrorCode.INSUFFICIENT_BITS,)
        sample_seed = random.randint(0, 2**31)
        qber, alice_final, bob_final_expected=compute_qber(
            alice_sifted, bob_sifted, sample_seed=sample_seed
        )
        logger.info(f"[Alice] QBER={qber*100:.2f}%")
        key_str = "".join(map(str, alice_final))
        if qber > QBER_THRESHOLD:
            await _stop(client, session_id)
            return SessionStartResp(
                session_id=session_id,
                statut="aborted",
                n_qubits_sent=n,
                n_qubits_received=n_delivered,
                n_sifted=n_sifted,
                qber=qber,
                key_final=key_str,
                error_message=ErrorCode.QBER_TOO_HIGH,
            )
 
        elapsed=time.time()-t_start
 
        logger.info(f"[Alice] Session {session_id} finished in {elapsed:.2f}s "f"— key: {alice_final}...")
        with _lock:
            _sessions[session_id]={
                "key_hash": alice_final,
                "qber": qber,
                "n_sifted": n_sifted,
                "elapsed": elapsed,}

        await _stop(client, session_id)
  
        return SessionStartResp(
            session_id=session_id,
            statut="success",
            n_qubits_sent=n,
            n_qubits_received=n_delivered,
            n_sifted=n_sifted,
            qber=qber,
            key_final=key_str,
        )

async def _stop(client: httpx.AsyncClient, session_id: str):
    try:
        await client.post(
            f"{QNS_URL}/network/stop",
            json={"session_id": session_id},
            timeout=5.0,
        )
        await client.delete(
            f"{BOB_URL}/session/{session_id}",
            timeout=5.0,
        )
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
    return {"statut": "ok", "service": "alice",
            "qns_url": QNS_URL, "bob_url": BOB_URL}
 


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
