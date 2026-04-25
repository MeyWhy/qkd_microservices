from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from qunetsim.components import Host, Network
from qunetsim.backends import EQSNBackend
from qunetsim.objects import Qubit
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import random
import threading
import time
from models import(
    Basis,
    NetworkInitReq, NetworkInitResp,
    SendQubitReq, SendQubitResp,
    GetMeasurementsReq, GetMeasurementsResp,
    QubitMeasurement,
    NetworkStopReq,
    BB84Error, ErrorCode,
)
logging.basicConfig(level=logging.WARNING)
logger=logging.getLogger("qns")

class NetworkSession:
    def __init__(self, session_id: str, loss_rate:float=0.0):
        self.session_id= session_id
        self.loss_rate= loss_rate
        self.backend= EQSNBackend()
        self.network= Network.get_instance()
        self.alice_host:Optional[Host] = None
        self.bob_host:Optional[Host] = None
        self.measurements: list[QubitMeasurement] = []
        self.lock= threading.Lock()
        self.bob_ready= threading.Event()
        self._active= False
    
    def start(self):
        self.network.start(['Alice', 'Bob'], self.backend)
        self.alice_host=Host('Alice', self.backend)
        self.bob_host=Host('Bob', self.backend)
        self.alice_host.add_connection('Bob')
        self.bob_host.add_connection('Alice')
        
        if self.loss_rate>0:
            self.network.packet_drop_rate=self.loss_rate
        self.alice_host.start()
        self.bob_host.start()

        self.network.add_host(self.alice_host)
        self.network.add_host(self.bob_host)
        self._active=True
        logger.info(f"[QNS] Session {self.session_id} started")
     
    def stop(self):
        if self._active:
            try:
                self.network.stop(stop_hosts=True)
            except Exception as e:
                logger.warning(f"[QNS] Error stop: {e}")
            self._active=False
            logger.info(f"[QNS] Session {self.session_id} stopped")

    def is_active(self)->bool:
        return self._active

_sessions:dict[str, NetworkSession]={}
_sessions_lock=threading.Lock()

def _encode_and_send(session: NetworkSession, qubit_id: int, bit:int, basis: Basis)-> bool:
    if random.random() < session.loss_rate:
        return False

    q=Qubit(session.alice_host)
    if bit==1:
        q.X()
    if basis==Basis.DIAGONAL:
        q.H()
    session.alice_host.send_qubit('Bob', q, await_ack=False)
    return True

def _bob_receive_and_measure(session:NetworkSession, qubit_id: int, timeout: float=3.0)-> Optional[QubitMeasurement]:
    q=session.bob_host.get_qubit('Alice', wait=timeout)
    if q is None:
        return None
    bob_basis=random.choice(list(Basis))
    if bob_basis==Basis.DIAGONAL:
        q.H()
    bit_res=q.measure()
    return QubitMeasurement(qubit_id=qubit_id, basis=bob_basis, bit_res=bit_res)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[QNS] Service Started")
    yield
    with _sessions_lock:
        for session in _sessions.values():
            session.stop()
    logger.info("[QNS] Service stopped")
   
app=FastAPI(
    title="Quantum Network Service",
    description="QuNetSim wrapper for BB84",
    version="0.3.0",
    lifespan=lifespan,
) 
@app.post("/network/init", response_model=NetworkInitResp)
async def init_network(req: NetworkInitReq):
    with _sessions_lock:
        dead=[sid for sid, s in _sessions.items() if not s.is_active()]
        for sid in dead:
            del _sessions[sid]
        if req.session_id in _sessions:
            return NetworkInitResp( session_id=req.session_id, statut="ready", message="Already active session")
        if _sessions:
            raise HTTPException(
                status_code=409,
                detail=BB84Error(
                    code=ErrorCode.NETWORK_UNAVAILABLE,
                    message="a session is already active"
                            "wait for it to end or call /network/stop",
                    session_id=req.session_id,
                ).model_dump(),
            )
    loop=asyncio.get_event_loop()
    session=NetworkSession(req.session_id)

    try:
        await loop.run_in_executor(None, session.start)
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
 
    with _sessions_lock:
        _sessions[req.session_id] = session
 
    return NetworkInitResp(
        session_id=req.session_id,
        statut="ready",
        message=f"network ready for {req.n_qubits} qubits",
    )

@app.post("/qubit/send", response_model=SendQubitResp)
async def send_qubit(req: SendQubitReq):
    with _sessions_lock:
        session=_sessions.get(req.session_id)
 
    if not session:
        raise HTTPException(
            status_code=404,
            detail=BB84Error(
                code=ErrorCode.SESSION_NOT_FOUND,
                session_id=req.session_id,
                message="Session Not found",
            ).model_dump(),
        )
 
    loop=asyncio.get_event_loop()
    bob_future=loop.run_in_executor(
        None, _bob_receive_and_measure, session, req.qubit_id
    )
    await asyncio.sleep(0.02)

    delivered=await loop.run_in_executor(None, _encode_and_send, session, req.qubit_id, req.bit, req.basis)
    if delivered:
        measurement=await bob_future
        if measurement:
            with session.lock:
                session.measurements.append(measurement)
    else:
        bob_future.cancel()
    return SendQubitResp(
        qubit_id=req.qubit_id,
        delivered=delivered,
    )

@app.get("/measurements/{session_id}", response_model=GetMeasurementsResp)
async def get_measurements(session_id: str):
    with _sessions_lock:
        session=_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=BB84Error(
                code=ErrorCode.SESSION_NOT_FOUND,
                session_id=session_id,
                message="Session not found",
            ).model_dump(),
        )
    with session.lock:
        measurements=list(session.measurements)
    
    return GetMeasurementsResp(session_id=session_id, measurements=measurements)

@app.post("/network/stop")
async def stop_network(req: NetworkStopReq):
    with _sessions_lock:
        session=_sessions.pop(req.session_id, None)
    if not session:
        return{"statut": "not_found"}

    loop=asyncio.get_event_loop()
    await loop.run_in_executor(None, session.stop)
    return {"statut": "stopped", "session_id": req.session_id}    

 
@app.get("/health")
async def health():
    with _sessions_lock:
        active = list(_sessions.keys())
    return {"statut": "ok", "active_sessions": active}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")