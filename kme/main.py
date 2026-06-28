import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    NodeRegistration, NodeInfo, NodeRole,
    SessionCreateReq, SessionJoinReq, SessionJoinResp,
    QubitUpload, MeasurementUpload,
    SiftUpload, KeyUpload,
    SessionStatusResponse, KeyStatus,
    NetworkInitReq, NetworkStopReq,
    new_session_id,
)
from kme.state_machine import SessionStatus
from collections import defaultdict
from kme.session_store import (
    get_redis, save_session, load_session, update_session,
    push_qubit_batch, pop_qubit_batch, qubit_batch_count,
    save_measurements, load_measurements,
    save_sift_upload, load_sift_upload,
    save_key_upload, load_key_upload,
    activate_key, consume_key, delete_session,
    list_open_sessions, list_active_sessions,
)
from kme.node_registry import (
    register_node, load_node, find_node_by_label,
    list_nodes,
)
from kme.event_bus import publish_event, publish_broadcast, register_pending_session


logger = logging.getLogger("kme")
logging.basicConfig(level=logging.INFO)


HTTP_TO = 30.0

#QKDL pool
_raw = os.getenv(
    "QKDL_URLS",
    os.getenv("QKDL_URL", "http://localhost:8003"),
)
QKDL_POOL: list[str] = [u.strip().rstrip("/") for u in _raw.split(",") if u.strip()]

METRICS = {
    "sessions_created":        0,
    "sessions_completed":      0,
    "sessions_aborted":        0,
    "sessions_intercepted":    0,   #Eve sessions
    "total_qubits":            0,
    "total_batches":           0,
    "registry_hits":           0,
    "stream_events":           0,
    "coordination_latency_ms": [],
    "session_latency_s":       [],
    "batch_latency_ms":        [],
    "active_nodes_peak":       0,
}

NODE_SESSION_COUNT: dict[str, int] = defaultdict(int)


#QKDL pool helpers

def _qkdl_lock_key(qkdl_url: str) -> str:
    safe = qkdl_url.replace("://", "_").replace("/", "_").replace(":", "_")
    return f"kme:qkdl_lock:{safe}"


def _acquire_qkdl_lock(r, qkdl_url: str, session_id: str) -> bool:
    return bool(r.set(_qkdl_lock_key(qkdl_url), session_id, nx=True, ex=600))


def _release_qkdl_lock(r, qkdl_url: str, session_id: str) -> None:
    key = _qkdl_lock_key(qkdl_url)
    if r.get(key) == session_id:
        r.delete(key)


def _get_qkdl_lock_holder(r, qkdl_url: str) -> str | None:
    return r.get(_qkdl_lock_key(qkdl_url))


def pick_free_qkdl(r) -> str | None:
    for url in QKDL_POOL:
        if not _get_qkdl_lock_holder(r, url):
            return url
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"[KME] Started - QKDL pool: {QKDL_POOL}")
    yield
    logger.info("[KME] Stopped")


app = FastAPI(
    title="KME - Key Management Entity",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def _get_session_or_404(r, session_id: str) -> dict:
    session = load_session(r, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def _emit(session_id: str, event: str, target_role: str, payload: dict | None = None) -> None:
    """
    Publish a lifecycle event to the session's Redis Stream.

    Replaces the old _notify(node_id, WebhookEvent) which did an HTTP POST
    to a single node's /webhook callback. target_role replaces node_id as
    the addressing mechanism: instead of "send to this specific node_id",
    events say "send to whoever is playing this role in this session" -
    which is what every call site actually meant (the node_id was always
    just "the current sender" or "the current receiver" for the session).
    Use target_role="broadcast" for events the old code fired at multiple
    node_ids in a row (e.g. key_available going to both Bob and Eve).
    """
    publish_event(session_id, event, target_role=target_role, payload=payload or {})
    METRICS["stream_events"] += 1


#Node registry

@app.post("/nodes/register", response_model=NodeInfo)
async def register(reg: NodeRegistration):
    r    = get_redis()
    info = register_node(r, reg)
    METRICS["registry_hits"] += 1
    count = len(list_nodes(r))
    if count > METRICS["active_nodes_peak"]:
        METRICS["active_nodes_peak"] = count
    NODE_SESSION_COUNT[info.node_id] = 0
    return info


@app.get("/nodes", response_model=list[NodeInfo])
async def get_nodes(role: str = None):
    r         = get_redis()
    role_enum = NodeRole(role) if role else None
    return list_nodes(r, role=role_enum)


@app.get("/nodes/{node_id}", response_model=NodeInfo)
async def get_node(node_id: str):
    r    = get_redis()
    node = load_node(r, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


#Session lifecycle

@app.post("/sessions", status_code=202)
@app.post("/keys",     status_code=202)
async def create_session(
    req: SessionCreateReq,
    background_tasks: BackgroundTasks,
):
    r          = get_redis()
    session_id = new_session_id()

    #Pick + lock a free QKDL (atomic, no race window)
    qkdl_url = pick_free_qkdl(r)
    if not qkdl_url:
        raise HTTPException(
            status_code=409,
            detail=f"All QKDL instances are busy. Pool: {QKDL_POOL}",
        )

    locked = _acquire_qkdl_lock(r, qkdl_url, session_id)
    if not locked:
        #Lost atomic race - try once more
        qkdl_url = pick_free_qkdl(r)
        if not qkdl_url:
            raise HTTPException(status_code=409,
                                detail="All QKDL instances are busy.")
        locked = _acquire_qkdl_lock(r, qkdl_url, session_id)
        if not locked:
            raise HTTPException(status_code=409,
                                detail="All QKDL instances are busy.")

    #Find receiver
    bob_node = find_node_by_label(r, req.receiver_label)
    if not bob_node:
        _release_qkdl_lock(r, qkdl_url, session_id)
        raise HTTPException(
            status_code=404,
            detail=f"Receiver node '{req.receiver_label}' not registered",
        )

    #Resolve Eve (optional) fail fast if label given but not found
    eve_node = None
    if req.interceptor_label:
        eve_node = find_node_by_label(r, req.interceptor_label)
        if not eve_node:
            _release_qkdl_lock(r, qkdl_url, session_id)
            raise HTTPException(
                status_code=404,
                detail=f"Interceptor node '{req.interceptor_label}' not registered",
            )

    #Persist session
    session = {
        "session_id":          session_id,
        "status":              SessionStatus.INITIALIZING.value,
        "sender_node_id":      req.sender_node_id,
        "receiver_node_id":    bob_node.node_id,
        "interceptor_node_id": eve_node.node_id if eve_node else None,
        "interceptor_label":   req.interceptor_label,
        "intercepted":         eve_node is not None,
        "n_qubits":            req.n_qubits,
        "batch_size":          req.batch_size,
        "loss_rate":           req.loss_rate,
        "distance_km":  req.distance_km, 
        "retry_enabled":       req.retry_enabled,
        "qkdl_url":            qkdl_url,
        "created_at":          time.time(),
        "started_at":          None,
        "sending_at":          None,
        "completed_at":        None,
        "key_status":          KeyStatus.NONE.value,
        "key_expires_at":      None,
        "n_delivered":         0,
        "n_sifted":            0,
        "qber":                0.0,
        "key_final":           "",
        "error_message":       "",
    }
    save_session(r, session)

    #QKDL init - awaited synchronously, NOT a background_task.
    #session_open must never be published before QKDL has confirmed the
    #network is ready, or a fast-discovering Bob (Streams' wake-up delivers
    #in single-digit milliseconds) can poll QKDL before /network/init has
    #even been sent, get a 404, and give up without retrying. See
    #_init_qkdl's docstring for the full race description.
    qkdl_ready = await _init_qkdl(
        session_id, req.n_qubits, req.loss_rate, req.distance_km, qkdl_url
    )
    if not qkdl_ready:
        #_init_qkdl already marked the session ABORTED and released the
        #QKDL lock on failure - surface this to Alice as a request failure
        #rather than returning 202 for a session that can never proceed.
        raise HTTPException(
            status_code=503,
            detail=f"QKDL failed to initialise network for session {session_id}",
        )

    #Register the session as pending pickup for Bob BEFORE publishing
    #session_open - Bob's poll loop discovers it via the pending-session
    #registry and calls begin_listening() to start its stream consumer.
    #Order matters here: if we published the event first and registered
    #pending second, Bob could in principle start listening (after some
    #other discovery path) before this line ran and have nothing wrong
    #happen - but registering first costs nothing and removes any doubt.
    register_pending_session(bob_node.node_id, session_id, r=r)

    #Notify Bob - published to the "receiver" role on this session's stream.
    _emit(
        session_id, "session_open", target_role="receiver",
        payload={
            "role":           "receiver",
            "sender_node_id": req.sender_node_id,
            "n_qubits":       req.n_qubits,
            "qkdl_url":       qkdl_url,
        },
    )

    #Notify Eve if present
    #Eve receives session_open with role="monitor" so she knows to register
    #as interceptor on the QKDL before Alice starts sending.
    if eve_node:
        register_pending_session(eve_node.node_id, session_id, r=r)
        _emit(
            session_id, "session_open", target_role="monitor",
            payload={
                "role":     "monitor",
                "n_qubits": req.n_qubits,
                "qkdl_url": qkdl_url,
            },
        )
        METRICS["sessions_intercepted"] += 1

    logger.info(
        f"[KME] Session {session_id} created "
        f"sender={req.sender_node_id[:8]} receiver={bob_node.label} "
        f"qkdl={qkdl_url}"
        + (f" interceptor={req.interceptor_label}" if req.interceptor_label else "")
    )

    METRICS["sessions_created"] += 1
    NODE_SESSION_COUNT[req.sender_node_id] += 1
    NODE_SESSION_COUNT[bob_node.node_id]   += 1

    return {
        "session_id":  session_id,
        "status":      "open",
        "qkdl_url":    qkdl_url,
        "intercepted": eve_node is not None,
    }


@app.post("/sessions/{session_id}/join", response_model=SessionJoinResp)
async def join_session(session_id: str, req: SessionJoinReq):
    r       = get_redis()
    session = _get_session_or_404(r, session_id)

    if session["status"] not in [
        SessionStatus.WAITING.value,
        SessionStatus.INITIALIZING.value,
    ]:
        raise HTTPException(
            status_code=409,
            detail=f"Session not open (status={session['status']})",
        )

    update_session(
        r, session_id,
        status=SessionStatus.SENDING.value,
        started_at=time.time(),
        sending_at=time.time(),
    )

    coord_ms = (time.time() - session["created_at"]) * 1000
    METRICS["coordination_latency_ms"].append(round(coord_ms, 2))

    #Notify Alice - published to "sender" role.
    _emit(
        session_id, "receiver_joined", target_role="sender",
        payload={"receiver_node_id": req.node_id},
    )

    logger.info(f"[KME] Bob {req.node_id[:8]} joined session {session_id}")

    return SessionJoinResp(
        session_id=session_id,
        role=NodeRole.RECEIVER,
        sender_node_id=session["sender_node_id"],
        n_qubits=session["n_qubits"],
        status="joined",
        qkdl_url=session.get("qkdl_url", ""),
    )


@app.post("/sessions/{session_id}/qubits", status_code=202)
async def upload_qubits(session_id: str, req: QubitUpload):
    r = get_redis()
    _get_session_or_404(r, session_id)
    push_qubit_batch(r, session_id, req.batch.model_dump())
    METRICS["total_batches"] += 1
    METRICS["total_qubits"]  += len(req.batch.qubits)
    return {"session_id": session_id, "batch_id": req.batch.batch_id, "queued": True}


@app.get("/sessions/{session_id}/qubits/next")
async def next_qubit_batch(session_id: str):
    r     = get_redis()
    batch = pop_qubit_batch(r, session_id)
    return {"session_id": session_id, "batch": batch,
            "remaining": qubit_batch_count(r, session_id)}


@app.post("/sessions/{session_id}/measurements", status_code=202)
async def upload_measurements(
    session_id: str,
    upload:     MeasurementUpload,
    background_tasks: BackgroundTasks,
):
    r = get_redis()
    _get_session_or_404(r, session_id)
    save_measurements(r, session_id, upload.model_dump())
    n = len(upload.measurements)
    update_session(r, session_id, n_delivered=n)

    #Notify Alice - published to "sender" role.
    _emit(
        session_id, "measurements_ready", target_role="sender",
        payload={"n_measurements": n},
    )
    logger.info(f"[KME] {n} measurements posted for session {session_id}")
    return {"session_id": session_id, "n_received": n}


@app.get("/sessions/{session_id}/measurements")
async def get_measurements(session_id: str):
    r    = get_redis()
    _get_session_or_404(r, session_id)
    meas = load_measurements(r, session_id)
    return {"session_id": session_id, "measurements": list(meas.values())}


@app.post("/sessions/{session_id}/sift", status_code=202)
async def upload_sift(
    session_id: str,
    upload:     SiftUpload,
    background_tasks: BackgroundTasks,
):
    r       = get_redis()
    session = _get_session_or_404(r, session_id)
    save_sift_upload(r, session_id, upload.model_dump())

    #Notify Bob - published to "receiver" role.
    _emit(
        session_id, "sift_ready", target_role="receiver",
        payload={"sample_seed": upload.sample_seed},
    )
    return {"session_id": session_id, "stored": True}


@app.get("/sessions/{session_id}/sift")
async def get_sift(session_id: str):
    r      = get_redis()
    _get_session_or_404(r, session_id)
    upload = load_sift_upload(r, session_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Sift data not yet available")
    return upload


@app.post("/sessions/{session_id}/key")
async def publish_key(
    session_id: str,
    upload:     KeyUpload,
    background_tasks: BackgroundTasks,
):
    r       = get_redis()
    session = _get_session_or_404(r, session_id)
    save_key_upload(r, session_id, upload.model_dump())

    if upload.status == "success":
        expires_at = activate_key(r, session_id)
        update_session(
            r, session_id,
            status=SessionStatus.DONE.value,
            completed_at=time.time(),
            key_final=upload.key_final,
            key_hash=upload.key_hash,
            qber=upload.qber,
            n_sifted=upload.n_sifted,
        )
        event   = "key_available"
        payload = {
            "key_status":     KeyStatus.ACTIVE.value,
            "key_expires_at": expires_at,
            "qber":           upload.qber,
        }
        elapsed = time.time() - session["created_at"]
        METRICS["sessions_completed"] += 1
        METRICS["session_latency_s"].append(round(elapsed, 3))
        logger.info(
            f"[KME] Key ACTIVE session={session_id} QBER={upload.qber*100:.2f}%"
        )
    else:
        update_session(
            r, session_id,
            status=SessionStatus.ABORTED.value,
            completed_at=time.time(),
            key_final="",
            qber=upload.qber if upload.qber else 0.0,
            error_message=upload.error_message,
        )
        event   = "session_aborted"
        payload = {"reason": upload.error_message, "qber": upload.qber}
        METRICS["sessions_aborted"] += 1
        logger.warning(
            f"[KME] Session {session_id} aborted: {upload.error_message} "
            + (f"QBER={upload.qber*100:.2f}%" if upload.qber else "")
        )

    #Notify Bob AND (if intercepted) Eve in one shot - this is exactly the
    #"same event, multiple node_ids" pattern the old webhook code handled
    #with two separate background_tasks.add_task(_notify, ...) calls.
    #target_role="broadcast" delivers to every role's consumer group;
    #only Bob and (if present) a registered Eve are actually listening on
    #this session's stream, so it's equivalent in effect to the old
    #explicit dual-notify, with one published event instead of two.
    _emit(session_id, event, target_role="broadcast", payload=payload)

    background_tasks.add_task(_stop_qkdl, session_id)
    return {"session_id": session_id, "status": upload.status}


@app.post("/sessions/{session_id}/consume-key")
@app.post("/keys/{session_id}/consume")
async def consume_session_key(session_id: str):
    r       = get_redis()
    ok, key = consume_key(r, session_id)
    if not ok:
        session = load_session(r, session_id)
        status  = session.get("key_status") if session else "unknown"
        raise HTTPException(
            status_code=409, detail=f"Key not available: status={status}"
        )
    return {"session_id": session_id, "key_final": key,
            "key_status": KeyStatus.CONSUMED.value}


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
@app.get("/keys/{session_id}",     response_model=SessionStatusResponse)
async def get_session(session_id: str):
    r       = get_redis()
    session = _get_session_or_404(r, session_id)

    created_at   = session.get("created_at", time.time())
    completed_at = session.get("completed_at")
    elapsed_s    = (
        round(completed_at - created_at, 3) if completed_at
        else round(time.time() - created_at, 3)
    )

    progress = {"open": 5, "waiting": 5, "initializing": 10,
                "sending": 50, "done": 100, "aborted": 0}
    labels   = {
        "open":         "Waiting for receiver",
        "waiting":      "Waiting for receiver",
        "initializing": "Initialising quantum network",
        "sending":      "Quantum transmission",
        "done":         "Key generated",
        "aborted":      "Session aborted",
    }

    valid_data = {k: session[k] for k in SessionStatusResponse.model_fields
                  if k in session}
    valid_data.update({
        "session_id":        session_id,
        "elapsed_s":         elapsed_s,
        "progress_pct":      progress.get(session["status"], 0),
        "phase_label":       labels.get(session["status"], ""),
        "qkdl_url":          session.get("qkdl_url", ""),
        "distance_km":       session.get("distance_km", 0.0),
        "intercepted":       session.get("intercepted", False),
        "interceptor_label": session.get("interceptor_label"),
    })
    return SessionStatusResponse(**valid_data)


@app.get("/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    """
    Full ordered event history for a session, read directly from its
    Redis Stream. New endpoint enabled by the Streams migration - webhooks
    had no equivalent (an HTTP POST leaves no trace once delivered). Useful
    for the BB84 timeline figure and for debugging stuck sessions.
    """
    from kme.event_bus import session_history
    r = get_redis()
    _get_session_or_404(r, session_id)
    return {"session_id": session_id, "events": session_history(session_id, r=r)}


@app.get("/sessions")
async def list_sessions(active_only: bool = True):
    r   = get_redis()
    ids = (list_active_sessions(r) if active_only
           else list_open_sessions(r) + list_active_sessions(r))
    return {"sessions": ids, "count": len(ids)}


@app.delete("/sessions/{session_id}")
async def cancel_session(session_id: str):
    r       = get_redis()
    session = _get_session_or_404(r, session_id)
    update_session(r, session_id, status=SessionStatus.ABORTED.value,
                   error_message="Cancelled by user")
    asyncio.create_task(_stop_qkdl(session_id))
    return {"status": "cancelled", "session_id": session_id}


#QKDL coordination

INIT_RETRIES = 5
INIT_RETRY_S = 1.2


async def _init_qkdl(
    session_id: str,
    n_qubits:   int,
    loss_rate:  float,
    distance_km: float,
    qkdl_url:   str,
) -> bool:
    """
    Returns True once QKDL has confirmed the network is ready, False if it
    never did after INIT_RETRIES attempts.

    Called synchronously (awaited) from create_session BEFORE session_open
    is published - not as a fire-and-forget background_task. This used to
    run purely in the background while the 202 response (and therefore
    session_open, sent right after, see the old race below) went out
    immediately; with Streams' event delivery now taking single-digit
    milliseconds, Bob could poll QKDL before this had even sent its first
    HTTP request, hit a 404 (no NetworkSession registered yet), and give up
    without retrying - a session that would never complete. Awaiting this
    here means session_open is only ever published once QKDL has actually
    confirmed the network exists, which removes the race at its source
    instead of asking every consumer to tolerate "not ready yet" responses.
    """
    r = get_redis()
    last_err = None

    for attempt in range(1, INIT_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TO) as client:
                resp = await client.post(
                    f"{qkdl_url}/network/init",
                    json=NetworkInitReq(
                        session_id=session_id,
                        n_qubits=n_qubits,
                        loss_rate=loss_rate,
                        distance_km=distance_km,
                    ).model_dump(),
                )

                if resp.status_code == 503:
                    wait = INIT_RETRY_S
                    try:
                        wait = float(
                            resp.json().get("detail", "").split("Retry in ")[-1]
                            .replace("s.", "").strip()
                        ) + 0.1
                    except Exception:
                        pass
                    logger.info(
                        f"[KME] QKDL cooling down, retry {attempt}/{INIT_RETRIES} "
                        f"in {wait:.1f}s session={session_id[:8]}"
                    )
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                logger.info(
                    f"[KME] QKDL initialised session={session_id[:8]} "
                    f"url={qkdl_url} (attempt {attempt})"
                )
                return True

        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code == 409:
                break
            logger.warning(
                f"[KME] QKDL init attempt {attempt} failed "
                f"session={session_id[:8]}: {e}"
            )
            await asyncio.sleep(INIT_RETRY_S)

        except Exception as e:
            last_err = e
            logger.warning(
                f"[KME] QKDL init attempt {attempt} error "
                f"session={session_id[:8]}: {e}"
            )
            await asyncio.sleep(INIT_RETRY_S)

    logger.error(
        f"[KME] QKDL init failed after {INIT_RETRIES} attempts "
        f"session={session_id[:8]}: {last_err}"
    )
    update_session(r, session_id,
                   status=SessionStatus.ABORTED.value,
                   error_message=f"QKDL init failed: {last_err}")
    _release_qkdl_lock(r, qkdl_url, session_id)
    return False


async def _stop_qkdl(session_id: str) -> None:
    r        = get_redis()
    session  = load_session(r, session_id)
    qkdl_url = session.get("qkdl_url", QKDL_POOL[0]) if session else QKDL_POOL[0]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{qkdl_url}/network/stop",
                json={"session_id": session_id},
            )
    except Exception as e:
        logger.warning(f"[KME] QKDL teardown partial session={session_id}: {e}")
    finally:
        _release_qkdl_lock(r, qkdl_url, session_id)


#Metrics + health

@app.get("/metrics")
async def metrics():
    def avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    r           = get_redis()
    qkdl_status = {
        url: (_get_qkdl_lock_holder(r, url) or "free")
        for url in QKDL_POOL
    }
    return {
        "sessions_created":              METRICS["sessions_created"],
        "sessions_completed":            METRICS["sessions_completed"],
        "sessions_aborted":              METRICS["sessions_aborted"],
        "sessions_intercepted":          METRICS["sessions_intercepted"],
        "total_qubits":                  METRICS["total_qubits"],
        "total_batches":                 METRICS["total_batches"],
        "throughput_qubits_per_session": round(
            METRICS["total_qubits"] / max(METRICS["sessions_completed"], 1), 2
        ),
        "avg_session_latency_s":         avg(METRICS["session_latency_s"]),
        "avg_coordination_latency_ms":   avg(METRICS["coordination_latency_ms"]),
        "avg_batch_latency_ms":          avg(METRICS["batch_latency_ms"]),
        "registry_hits":                 METRICS["registry_hits"],
        "stream_events":                 METRICS["stream_events"],
        "active_nodes_peak":             METRICS["active_nodes_peak"],
        "active_sessions":               len(list_active_sessions(get_redis())),
        "qkdl_pool":                     qkdl_status,
    }


@app.get("/health")
async def health():
    r = get_redis()
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status":           "ok",
        "redis":            redis_ok,
        "active_sessions":  len(list_active_sessions(r)),
        "registered_nodes": len(list_nodes(r)),
        "qkdl_pool":        QKDL_POOL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")