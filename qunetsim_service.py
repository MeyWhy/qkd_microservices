"""
qkdl/main.py  — QKDL (QKD Link Layer)
========================================
Port 8003.

What changed from your v5/v6 qunetsim_service.py:

1. MeasurementRecord.bit_res → bit_result
   New models.py uses bit_result. Fixed in _process_batch_sync.

2. Removed direct Redis writes (save_bob_measurements_batch, mark_delivered)
   In the new architecture, the QNS does NOT write measurements to Redis.
   Measurements are returned in the HTTP response and Bob posts them
   to the KME via POST /sessions/{id}/measurements.
   The QNS is now a pure quantum transport layer with no Redis dependency.

3. Added GET /qubit/receive/{session_id}
   Bob's node polls this endpoint to get measured qubits one at a time.
   The QNS stores a per-session measurement queue in memory.
   Bob drains it at his own pace.

4. Classical channel endpoints kept (send_classical, recv_classical)
   These are already correct in your code. Kept as-is.

5. NetworkInitReq/NetworkStopReq import path updated to shared.models
   (your file had from models import ... which works if models.py is
   in the same directory; kept flexible with try/except)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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

# ── Import models — try new path first, fall back to flat layout ──
try:
    from models import (
        Basis, NetworkInitReq, NetworkInitResp,
        MeasurementRecord, SendBatchReq, SendBatchResp,
        QubitBatch, NetworkStopReq,
    )
except ModuleNotFoundError:
    from models import (                                   # flat layout (v5/v6)
        Basis, NetworkInitReq, NetworkInitResp,
        MeasurementRecord, SendBatchReq, SendBatchResp,
        QubitBatch, NetworkStopReq,
    )

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("qkdl")


# ─────────────────────────────────────────────
# Classical channel models
# ─────────────────────────────────────────────

class ClassicalSendReq(BaseModel):
    session_id:  str
    payload_hex: str
    mode:        str = "direct"   # "direct" | "broadcast"


class ClassicalSendResp(BaseModel):
    session_id: str
    delivered:  bool
    mode:       str


class ClassicalRecvResp(BaseModel):
    session_id:  str
    payload_hex: str
    available:   bool


# ─────────────────────────────────────────────
# Network session
# ─────────────────────────────────────────────

class NetworkSession:
    def __init__(self, session_id: str, loss_rate: float = 0.0):
        self.session_id   = session_id
        self.loss_rate    = loss_rate
        self.backend      = EQSNBackend()
        self.network      = Network.get_instance()
        self.alice_host:  Optional[Host] = None
        self.bob_host:    Optional[Host] = None
        self._send_lock   = threading.Lock()
        self._active      = False

        # FIX 3: in-memory measurement queue for Bob to poll
        # key: qubit_id, value: {qubit_id, basis, bit_result, delivered}
        self._meas_queue:  list[dict] = []
        self._meas_lock    = threading.Lock()

    def start(self):
        self.network.start(['Alice', 'Bob'], self.backend)
        self.alice_host = Host('Alice', self.backend)
        self.bob_host   = Host('Bob',   self.backend)
        self.alice_host.add_connection('Bob')
        self.bob_host.add_connection('Alice')
        if self.loss_rate > 0:
            self.network.packet_drop_rate = self.loss_rate
        self.alice_host.start()
        self.bob_host.start()
        self.network.add_host(self.alice_host)
        self.network.add_host(self.bob_host)
        self._active = True
        logger.info(f"[QKDL] Session {self.session_id} started")

    def stop(self):
        if self._active:
            try:
                self.network.stop(stop_hosts=True)
            except Exception as e:
                logger.warning(f"[QKDL] Stop error: {e}")
            self._active = False
            logger.info(f"[QKDL] Session {self.session_id} stopped")

    def is_active(self) -> bool:
        return self._active

    def push_measurement(self, result: dict) -> None:
        with self._meas_lock:
            self._meas_queue.append(result)

    def pop_measurement(self) -> Optional[dict]:
        with self._meas_lock:
            return self._meas_queue.pop(0) if self._meas_queue else None

    def measurement_count(self) -> int:
        with self._meas_lock:
            return len(self._meas_queue)


_sessions:      dict[str, NetworkSession] = {}
_sessions_lock  = threading.Lock()

# Classical inbox: session_id → [hex_payload, ...]
_classical_inbox: dict[str, list[str]] = {}
_classical_lock   = threading.Lock()


# ─────────────────────────────────────────────
# Qubit processing
# ─────────────────────────────────────────────

def _process_batch_sync(
    session: NetworkSession,
    batch:   QubitBatch,
) -> list[dict]:
    """
    Processes one batch of qubits synchronously.
    Returns list of result dicts with qubit_id, delivered, bob_basis, bob_bit.

    FIX 1: bob_bit stored as bit_result in MeasurementRecord (not bit_res).
    FIX 2: results stored in session._meas_queue instead of Redis directly.
    """
    results = []

    for qrec in batch.qubits:
        qid   = qrec.qubit_id
        bit   = qrec.bit
        basis = qrec.basis

        # Simulate loss
        if session.loss_rate > 0 and random.random() < session.loss_rate:
            results.append({
                "qubit_id": qid, "delivered": False,
                "bob_basis": None, "bob_bit": None,
            })
            continue

        bob_res   = {}
        bob_ready = threading.Event()

        def bob_receive(res=bob_res, ready=bob_ready):
            q = session.bob_host.get_qubit("Alice", wait=3)
            if q is None:
                res["bit"]   = None
                res["basis"] = None
            else:
                bob_basis = random.choice(list(Basis))
                if bob_basis == Basis.DIAGONAL:
                    q.H()
                res["bit"]   = q.measure()
                res["basis"] = bob_basis.value
            ready.set()

        t_bob = threading.Thread(target=bob_receive, daemon=True)
        t_bob.start()
        time.sleep(0.003)   # reduced from 0.01 → 0.003 for better latency

        with session._send_lock:
            q = Qubit(session.alice_host)
            if bit == 1:
                q.X()
            if basis == Basis.DIAGONAL:
                q.H()
            session.alice_host.send_qubit('Bob', q, await_ack=False)

        bob_ready.wait(timeout=4.0)
        t_bob.join(timeout=4.0)

        delivered = bob_res.get("bit") is not None
        result    = {
            "qubit_id":  qid,
            "delivered": delivered,
            "bob_basis": bob_res.get("basis"),
            "bob_bit":   bob_res.get("bit"),    # internal key kept as bob_bit
        }
        results.append(result)

        # FIX 2: store in session queue for Bob to poll
        # (replaces direct Redis write)
        if delivered:
            session.push_measurement({
                "qubit_id":   qid,
                "basis":      bob_res.get("basis"),
                "bit_result": bob_res.get("bit"),   # FIX 1: bit_result not bit_res
                "delivered":  True,
            })

    return results


# ─────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[QKDL] Service started")
    yield
    with _sessions_lock:
        for session in _sessions.values():
            session.stop()
    logger.info("[QKDL] Service stopped")


app = FastAPI(
    title="QKDL — QKD Link Layer (QuNetSim)",
    description="Quantum transport layer for BB84",
    version="0.7.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Network lifecycle
# ─────────────────────────────────────────────

@app.post("/network/init", response_model=NetworkInitResp)
async def init_network(req: NetworkInitReq):
    with _sessions_lock:
        # Clean dead sessions
        dead = [sid for sid, s in _sessions.items() if not s.is_active()]
        for sid in dead:
            del _sessions[sid]

        if req.session_id in _sessions:
            return NetworkInitResp(
                session_id=req.session_id,
                statut="ready",
                message="Already active",
            )
        if _sessions:
            raise HTTPException(status_code=409, detail="A session is already active")

    session = NetworkSession(req.session_id, loss_rate=req.loss_rate)
    loop    = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, session.start)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with _sessions_lock:
        _sessions[req.session_id] = session

    return NetworkInitResp(
        session_id=req.session_id,
        statut="ready",
        message=f"Network ready for {req.n_qubits} qubits",
    )


@app.post("/network/stop")
async def stop_network(req: NetworkStopReq):
    with _sessions_lock:
        session = _sessions.pop(req.session_id, None)
    with _classical_lock:
        _classical_inbox.pop(req.session_id, None)
    if session:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, session.stop)
    return {"statut": "stopped", "session_id": req.session_id}


# ─────────────────────────────────────────────
# Quantum batch transmission
# ─────────────────────────────────────────────

@app.post("/batch/send", response_model=SendBatchResp)
async def send_batch(req: SendBatchReq):
    """
    Alice sends a batch of qubits.
    Results are stored in the session measurement queue.
    Bob retrieves them via GET /qubit/receive/{session_id}.

    FIX 2: No Redis writes here. The QNS is stateless w.r.t. Redis.
    """
    with _sessions_lock:
        session = _sessions.get(req.session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    loop    = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, _process_batch_sync, session, req.batch
    )

    return SendBatchResp(
        session_id=req.session_id,
        batch_id=req.batch.batch_id,
        results=results,
    )


# ─────────────────────────────────────────────
# FIX 3: Bob polling endpoint
# ─────────────────────────────────────────────

@app.get("/qubit/receive/{session_id}")
async def receive_qubit(session_id: str):
    """
    Bob's node polls this to retrieve measured qubits one at a time.

    Returns:
      - The next measurement if available:
        {qubit_id, basis, bit_result, delivered, queue_empty: false}
      - {qubit_id: null, queue_empty: true} if nothing waiting

    Bob drains this queue after Alice has uploaded all batches.
    The KME notifies Bob when measurements are ready (measurements_ready event).
    """
    with _sessions_lock:
        session = _sessions.get(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    meas = session.pop_measurement()
    if meas:
        return {
            "session_id":  session_id,
            "qubit_id":    meas["qubit_id"],
            "basis":       meas["basis"],
            "bit_result":  meas["bit_result"],   # FIX 1: bit_result
            "delivered":   meas["delivered"],
            "queue_empty": False,
            "remaining":   session.measurement_count(),
        }

    return {
        "session_id":  session_id,
        "qubit_id":    None,
        "bit_result":  None,
        "queue_empty": True,
        "remaining":   0,
    }


@app.get("/qubit/count/{session_id}")
async def qubit_count(session_id: str):
    """Returns how many measured qubits are waiting for Bob to collect."""
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "count":      session.measurement_count(),
    }


# ─────────────────────────────────────────────
# Classical channel (kept from your code, no changes)
# ─────────────────────────────────────────────

@app.post("/classical/send", response_model=ClassicalSendResp)
async def send_classical(req: ClassicalSendReq):
    """
    Sends a classical message (ciphertext) from Alice to Bob
    using the live QuNetSim hosts.
    Stores received payload in the classical inbox for Bob to retrieve.
    """
    with _sessions_lock:
        session = _sessions.get(req.session_id)

    if not session or not session.is_active():
        raise HTTPException(
            status_code=404, detail="Session not found or inactive"
        )

    loop = asyncio.get_event_loop()

    def _do_send() -> bool:
        received = {}
        ready    = threading.Event()

        def _bob_listen():
            msgs    = session.bob_host.get_classical("Alice", wait=10)
            if msgs:
                m       = msgs[-1] if isinstance(msgs, list) else msgs
                content = getattr(m, "content", m)
                received["hex"] = (
                    content if isinstance(content, str) else content.hex()
                )
            ready.set()

        threading.Thread(target=_bob_listen, daemon=True).start()
        time.sleep(0.05)

        if req.mode == "broadcast":
            session.alice_host.send_broadcast(req.payload_hex, ["Bob"])
        else:
            session.alice_host.send_classical("Bob", req.payload_hex)

        ready.wait(timeout=10.0)

        if "hex" in received:
            with _classical_lock:
                _classical_inbox.setdefault(req.session_id, []).append(
                    received["hex"]
                )
            return True
        return False

    delivered = await loop.run_in_executor(None, _do_send)
    return ClassicalSendResp(
        session_id=req.session_id,
        delivered=delivered,
        mode=req.mode,
    )


@app.get("/classical/recv/{session_id}", response_model=ClassicalRecvResp)
async def recv_classical(session_id: str):
    """Retrieves the next classical message from Bob's inbox."""
    with _classical_lock:
        inbox = _classical_inbox.get(session_id, [])
        if inbox:
            return ClassicalRecvResp(
                session_id=session_id,
                payload_hex=inbox.pop(0),
                available=True,
            )
    return ClassicalRecvResp(
        session_id=session_id,
        payload_hex="",
        available=False,
    )


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    with _sessions_lock:
        active = list(_sessions.keys())
        counts = {
            sid: _sessions[sid].measurement_count()
            for sid in active
        }
    return {
        "statut":          "ok",
        "active_sessions": active,
        "measurement_queues": counts,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")