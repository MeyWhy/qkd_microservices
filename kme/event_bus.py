from __future__ import annotations
 
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional
 
import redis
 
logger = logging.getLogger("kme.event_bus")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
 
#Roles map 1:1 to consumer groups. "broadcast" is not a group of its own -
#broadcast events are read by every role's group via the target_role filter.
ROLES = ("sender", "receiver", "monitor")
 
#How long an entry can sit unacked in a consumer's PEL before another
#consumer in the same group is allowed to steal it via XAUTOCLAIM. Generous
#relative to BB84 session timescales (QKDL_FIXED_OVERHEAD=30s + per-qubit
#cost), since a node mid-restart should get a real chance to come back.
CLAIM_MIN_IDLE_MS = 15_000
 
#Stream entries are trimmed after this many, per session - sessions in this
#architecture exchange a handful of lifecycle events (session_open,
#receiver_joined, measurements_ready, sift_ready, key_available/aborted),
#so this is a generous ceiling, not a tight budget.
STREAM_MAXLEN = 200
 
#Streams (and their consumer groups) expire with the session - mirrors
#kme/session_store.py's SESSION_TTL so event history doesn't outlive the
#session data it describes.
STREAM_TTL_S = 7200
 
 
def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)
 
 
def _stream_key(session_id: str) -> str:
    return f"qubit_events_{session_id}"
 
 
def _group_name(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"Unknown role '{role}', expected one of {ROLES}")
    return f"{role}-group"
 
 
def _ensure_group(r: redis.Redis, stream_key: str, group: str) -> None:
    """
    Create the consumer group at the start of the stream ('0') if it
    doesn't already exist, and create the stream itself if missing
    (mkstream=True) so a group can be created before any XADD has
    happened yet (e.g. Bob may subscribe before session_open is published
    in some race-y startup orderings).
    """
    try:
        r.xgroup_create(stream_key, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise  #some other failure - surface it
 
 
@dataclass
class StreamEvent:
    event: str
    session_id: str
    payload: dict
    target_role: str
    entry_id: str  #Redis stream entry ID, needed for XACK
 
 
#Publishing
 
def publish_event(
    session_id: str,
    event: str,
    target_role: str,
    payload: dict | None = None,
    r: redis.Redis | None = None,
) -> str:
    """
    Append one event to the session's stream. Returns the new entry ID.
 
    target_role: "sender" | "receiver" | "monitor" | "broadcast".
    "broadcast" is delivered to every role's consumer group (e.g.
    key_available / session_aborted, which both Bob and an interceting Eve
    need to see) - equivalent to the old code's pattern of firing the same
    WebhookEvent at multiple node_ids.
    """
    if target_role not in (*ROLES, "broadcast"):
        raise ValueError(f"Unknown target_role '{target_role}'")
 
    r = r or get_redis()
    stream_key = _stream_key(session_id)
 
    #Make sure every role's group exists before the first XADD, so no
    #group is created *after* events have already been appended (which
    #would make it start reading from "$" / now and miss earlier history
    #if it weren't for our id="0" default in _ensure_group).
    for role in ROLES:
        _ensure_group(r, stream_key, _group_name(role))
 
    entry_id = r.xadd(
        stream_key,
        {
            "event": event,
            "session_id": session_id,
            "target_role": target_role,
            "payload": json.dumps(payload or {}),
        },
        maxlen=STREAM_MAXLEN,
        approximate=True,
    )
    r.expire(stream_key, STREAM_TTL_S)
    logger.debug(
        f"[EventBus] XADD {stream_key} event={event} "
        f"target={target_role} id={entry_id}"
    )
    return entry_id
 
 
def publish_broadcast(session_id: str, event: str, payload: dict | None = None) -> str:
    return publish_event(session_id, event, target_role="broadcast", payload=payload)
 
 
#Consuming
 
def read_role_events(
    r: redis.Redis,
    session_id: str,
    role: str,
    consumer_name: str,
    block_ms: int = 2000,
    count: int = 10,
) -> list[StreamEvent]:
    """
    Read new events for this role from the session stream via its
    consumer group, filtering client-side on target_role (Streams have no
    server-side field filter). Does NOT auto-ack - callers ack after
    successfully handling each event (see ack_event), so a crash between
    read and handling leaves the entry claimable by XAUTOCLAIM instead of
    silently lost.
    """
    stream_key = _stream_key(session_id)
    group = _group_name(role)
    _ensure_group(r, stream_key, group)
 
    try:
        resp = r.xreadgroup(
            group, consumer_name,
            {stream_key: ">"},  #">" = only new, never-delivered entries
            count=count, block=block_ms,
        )
    except redis.ResponseError as e:
        #Stream or group vanished (session TTL'd out) - treat as no events
        #rather than crashing the node's poll loop.
        logger.debug(f"[EventBus] read_role_events {stream_key}/{group}: {e}")
        return []
 
    events: list[StreamEvent] = []
    for _stream, entries in resp or []:
        for entry_id, fields in entries:
            if fields.get("target_role") not in (role, "broadcast"):
                #Not for us - ack immediately so it doesn't linger in our
                #PEL forever (we'll never "handle" something addressed to
                #a different role).
                r.xack(stream_key, group, entry_id)
                continue
            try:
                payload = json.loads(fields.get("payload", "{}"))
            except json.JSONDecodeError:
                payload = {}
            events.append(StreamEvent(
                event=fields.get("event", ""),
                session_id=fields.get("session_id", session_id),
                payload=payload,
                target_role=fields.get("target_role", role),
                entry_id=entry_id,
            ))
    return events
 
 
def ack_event(r: redis.Redis, session_id: str, role: str, entry_id: str) -> None:
    r.xack(_stream_key(session_id), _group_name(role), entry_id)
 
 
def reclaim_stale_entries(
    r: redis.Redis, session_id: str, role: str, consumer_name: str,
    min_idle_ms: int = CLAIM_MIN_IDLE_MS,
) -> list[StreamEvent]:
    """
    Steal entries that were read by a (possibly dead) consumer in this
    group but never acked, after min_idle_ms of inactivity. Call this
    periodically (e.g. once per poll-loop tick) so a node that crashed
    mid-handling doesn't permanently strand events in its PEL.
    """
    stream_key = _stream_key(session_id)
    group = _group_name(role)
    try:
        _next_cursor, claimed, _deleted = r.xautoclaim(
            stream_key, group, consumer_name,
            min_idle_time=min_idle_ms, start_id="0-0", count=20,
        )
    except redis.ResponseError:
        return []
 
    events: list[StreamEvent] = []
    for entry_id, fields in claimed:
        if fields.get("target_role") not in (role, "broadcast"):
            r.xack(stream_key, group, entry_id)
            continue
        try:
            payload = json.loads(fields.get("payload", "{}"))
        except json.JSONDecodeError:
            payload = {}
        events.append(StreamEvent(
            event=fields.get("event", ""),
            session_id=fields.get("session_id", session_id),
            payload=payload,
            target_role=fields.get("target_role", role),
            entry_id=entry_id,
        ))
    return events
 
 
#Pending-session registry - solves the pull-model chicken-and-egg problem:
#Bob/Eve don't know a session_id exists until they read the session_open
#event for it, but they can't start reading a session's stream (and
#therefore can't see session_open) without already knowing the
#session_id. Instead of a global discovery stream (which has the same
#race, just one level up - a node could start listening on it *after* an
#announcement was published and miss it), the KME registers each new
#session directly under the recipient node_id's pending-set at creation
#time (create_session already knows bob_node.node_id / eve_node.node_id
#at that point). Bob/Eve poll their own pending-set - a plain Redis SET
#read, not a stream - via their existing _agent_loop tick.
 
_PENDING_TTL_S = 600  #generous vs. typical session pickup latency (~POLL_INTERVAL)
 
 
def _pending_key(node_id: str) -> str:
    return f"kme:pending_sessions:{node_id}"
 
 
def register_pending_session(node_id: str, session_id: str, r: redis.Redis | None = None) -> None:
    r = r or get_redis()
    key = _pending_key(node_id)
    r.sadd(key, session_id)
    r.expire(key, _PENDING_TTL_S)
    #Best-effort low-latency wake-up: Pub/Sub has no delivery guarantee
    #(exactly the reason it was rejected for the events themselves), but
    #that's fine here - the pending-session SET above is the durable
    #source of truth, and this PUBLISH is purely an optimization to avoid
    #waiting for the next poll tick. A missed PUBLISH (no listener at that
    #instant) just means the node falls back to discovering the session on
    #its next periodic drain - never a lost session, only added latency.
    try:
        r.publish(f"kme:wakeup:{node_id}", session_id)
    except Exception:
        pass  #wake-up is an optimization, never allowed to break registration
 
 
def pop_pending_sessions(node_id: str, r: redis.Redis | None = None) -> list[str]:
    """
    Atomically drain and return all session_ids pending pickup for this
    node_id. Each session is returned at most once across all callers -
    safe even if a node restarts mid-poll, since a session it already
    popped is gone from the set either way (it will already have started
    listening on that session's stream by then).
    """
    r = r or get_redis()
    key = _pending_key(node_id)
    ids: list[str] = []
    while True:
        sid = r.spop(key)
        if sid is None:
            break
        ids.append(sid)
    return ids
 
 
class WakeupListener:
    """
    Background Pub/Sub subscriber for the low-latency session-discovery
    wake-up signal (see register_pending_session). On receiving a wake-up,
    calls back into the node so it can immediately drain its pending-
    session registry instead of waiting for the next poll tick.
 
    This is the only Pub/Sub usage left in the bus - deliberately, since
    it's the one place where "fire and forget, occasionally missed" is
    actually the right tradeoff: the durable pending-session SET (Streams-
    adjacent registry) is always the source of truth, this is purely a
    latency optimization on top of it.
    """
 
    def __init__(self, node_id: str, on_wakeup: Callable[[], None]):
        self.node_id = node_id
        self.on_wakeup = on_wakeup
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
 
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._listen, daemon=True,
            name=f"wakeup-{self.node_id[:8]}",
        )
        self._thread.start()
 
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
 
    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                r = get_redis()
                pubsub = r.pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(f"kme:wakeup:{self.node_id}")
                for raw in pubsub.listen():
                    if self._stop.is_set():
                        break
                    if raw["type"] != "message":
                        continue
                    try:
                        self.on_wakeup()
                    except Exception as e:
                        logger.warning(f"[EventBus] Wake-up handler error: {e}")
            except Exception as e:
                if not self._stop.is_set():
                    logger.debug(f"[EventBus] Wake-up listener reconnecting: {e}")
                    time.sleep(1.0)
 
 
def session_history(session_id: str, r: redis.Redis | None = None) -> list[dict]:
    """
    Full ordered event history for a session - used for the BB84 timeline
    figure (article/thesis) and for debugging. Does not require a
    consumer group; reads the raw stream directly.
    """
    r = r or get_redis()
    stream_key = _stream_key(session_id)
    try:
        raw = r.xrange(stream_key, min="-", max="+")
    except redis.ResponseError:
        return []
    out = []
    for entry_id, fields in raw:
        try:
            payload = json.loads(fields.get("payload", "{}"))
        except json.JSONDecodeError:
            payload = {}
        ts_ms = int(entry_id.split("-")[0])
        out.append({
            "entry_id": entry_id,
            "timestamp_s": ts_ms / 1000.0,
            "event": fields.get("event", ""),
            "target_role": fields.get("target_role", ""),
            "payload": payload,
        })
    return out
 
 
class NodeStreamConsumer:
    """
    Background polling loop that reads this node's events from a session
    stream and dispatches them to a handler - the Streams equivalent of
    the old BaseNode's inbound /webhook POST endpoint.
 
    One instance is created per *session* the node is participating in
    (mirroring the old per-node webhook dispatch, just pulled instead of
    pushed), since a node's role and node_id stay fixed but the session_id
    changes per BB84 run.
    """
 
    def __init__(
        self,
        session_id: str,
        role: str,
        consumer_name: str,
        handler: Callable[[str, str, dict], None],
        poll_interval_s: float = 0.3,
    ):
        self.session_id = session_id
        self.role = role
        self.consumer_name = consumer_name
        self.handler = handler
        self.poll_interval_s = poll_interval_s
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
 
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name=f"stream-{self.role}-{self.session_id[:8]}",
        )
        self._thread.start()
 
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
 
    def _run(self) -> None:
        r = get_redis()
        last_reclaim = 0.0
        while not self._stop.is_set():
            try:
                events = read_role_events(
                    r, self.session_id, self.role, self.consumer_name,
                    block_ms=int(self.poll_interval_s * 1000),
                )
                now = time.time()
                if now - last_reclaim > 5.0:
                    events += reclaim_stale_entries(
                        r, self.session_id, self.role, self.consumer_name,
                    )
                    last_reclaim = now
 
                for ev in events:
                    try:
                        self.handler(ev.event, ev.session_id, ev.payload)
                    except Exception as e:
                        logger.warning(
                            f"[EventBus] Handler error session={self.session_id[:8]} "
                            f"event={ev.event}: {e}"
                        )
                    finally:
                        #Ack regardless of handler success - matches the old
                        #webhook semantics where handle_webhook() always
                        #returned 200 once received, and retries were not
                        #the receiver's responsibility. A handler that needs
                        #at-least-once retry semantics should ack only after
                        #success instead; not needed for this architecture's
                        #idempotent on_* handlers.
                        ack_event(r, self.session_id, self.role, ev.entry_id)
            except Exception as e:
                if not self._stop.is_set():
                    logger.warning(f"[EventBus] Consumer loop error: {e}")
                    time.sleep(1.0)
 