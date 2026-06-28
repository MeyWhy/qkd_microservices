import asyncio
import logging
import os
import threading
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI

from models import (
    NodeRole, NodeRegistration, NodeInfo,
    SessionStatusResponse,
)
from kme.event_bus import get_redis as get_event_redis, NodeStreamConsumer, WakeupListener

logger = logging.getLogger("node.base")

KME_URL              = os.getenv("KME_URL",  "http://localhost:8000")
POLL_INTERVAL        = float(os.getenv("NODE_POLL_INTERVAL", "2.0"))
REGISTER_RETRY_DELAY = 3.0

# NodeRole values match the event_bus role strings 1:1 except for "sender"/
# "receiver"/"monitor" naming, which already lines up — RELAY has no event
# bus role today (no relay node type exists yet), so it isn't given a
# consumer.
_ROLE_TO_BUS_ROLE = {
    NodeRole.SENDER:   "sender",
    NodeRole.RECEIVER: "receiver",
    NodeRole.MONITOR:  "monitor",
}


class BaseNode(ABC):

    def __init__(
        self,
        role:         NodeRole,
        label:        str,
        callback_url: str = "",
        metadata:     dict = {},
    ):
        self.role         = role
        self.label        = label
        #callback_url is kept only for NodeRegistration backward
        #compatibility (some deployments may still record it for human
        #debugging / dashboards) — it is no longer dialed by the KME, since
        #event delivery is now pull-based via Redis Streams.
        self.callback_url = callback_url
        self.metadata     = metadata

        self.node_id: Optional[str] = None
        self._sessions: dict[str, dict] = {}
        self._client  = httpx.AsyncClient(timeout=30.0)
        self._running = False
        #Captured explicitly in start() — _dispatch_event runs on background
        #consumer/wake-up threads, where asyncio.get_event_loop() does not
        #reliably return the FastAPI app's running loop (no loop is set as
        #"current" on a non-main thread by default in modern asyncio).
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

        #session_id -> NodeStreamConsumer. One consumer thread per active
        #session this node is participating in (a node's role is fixed,
        #but it can be involved in multiple sessions over its lifetime —
        #mirrors the old model where each session_open webhook spawned new
        #per-session state in self._sessions).
        self._stream_consumers: dict[str, NodeStreamConsumer] = {}
        #_stream_consumers is touched from both the asyncio loop (poll tick)
        #and the WakeupListener's background thread (instant wake-up) —
        #guard it explicitly rather than relying on dict op atomicity, since
        #begin_listening() does a check-then-act (membership test + insert).
        self._stream_consumers_lock = threading.Lock()
        #Low-latency wake-up (receiver/monitor only — see _drain_pending_sessions).
        self._wakeup_listener: Optional[WakeupListener] = None

    #Lifecycle

    async def start(self) -> None:
        self._main_loop = asyncio.get_running_loop()
        self.node_id = await self._register()
        self._running = True
        if self.role in (NodeRole.RECEIVER, NodeRole.MONITOR):
            self._wakeup_listener = WakeupListener(
                self.node_id, on_wakeup=self._drain_pending_sessions
            )
            self._wakeup_listener.start()
        logger.info(
            f"[{self.label}] Started  node_id={self.node_id[:8]} "
            f"role={self.role.value}"
        )
        asyncio.create_task(self._agent_loop())

    async def stop(self) -> None:
        self._running = False
        if self._wakeup_listener:
            self._wakeup_listener.stop()
        for consumer in self._stream_consumers.values():
            consumer.stop()
        self._stream_consumers.clear()
        await self._client.aclose()
        logger.info(f"[{self.label}] Stopped")

    async def _register(self) -> str:
        while True:
            try:
                resp = await self._client.post(
                    f"{KME_URL}/nodes/register",
                    json=NodeRegistration(
                        role=self.role,
                        callback_url=self.callback_url,
                        label=self.label,
                        metadata=self.metadata,
                    ).model_dump(),
                )
                resp.raise_for_status()
                info = NodeInfo.model_validate(resp.json())
                logger.info(
                    f"[{self.label}] Registered → node_id={info.node_id[:8]}"
                )
                return info.node_id
            except (httpx.ConnectError, httpx.HTTPError) as e:
                logger.warning(
                    f"[{self.label}] KME unreachable, retrying in "
                    f"{REGISTER_RETRY_DELAY}s: {e}"
                )
                await asyncio.sleep(REGISTER_RETRY_DELAY)

    #Stream event dispatch

    def begin_listening(self, session_id: str) -> None:
        """
        Start pulling this node's events for `session_id` from its Redis
        Stream. Replaces the old model where the KME pushed a webhook to
        this node's /webhook endpoint — now the node pulls, via a
        consumer-group reader scoped to its role.

        Call this as soon as the node learns about a session (mirrors when
        the old code would have started receiving webhook POSTs for it —
        i.e. immediately on session creation for the sender, on
        session_open delivery for the receiver/monitor). Idempotent: a
        second call for an already-listening session is a no-op.
        """
        if session_id in self._stream_consumers:
            return
        bus_role = _ROLE_TO_BUS_ROLE.get(self.role)
        if bus_role is None:
            logger.warning(
                f"[{self.label}] No event-bus role mapping for {self.role.value}; "
                f"not listening on session {session_id[:8]}"
            )
            return
        with self._stream_consumers_lock:
            if session_id in self._stream_consumers:
                return  # lost the race to another thread — already started
            consumer = NodeStreamConsumer(
                session_id=session_id,
                role=bus_role,
                consumer_name=self.node_id or self.label,
                handler=self._dispatch_event,
            )
            consumer.start()
            self._stream_consumers[session_id] = consumer
        logger.debug(
            f"[{self.label}] Listening on session={session_id[:8]} "
            f"(role={bus_role})"
        )

    def stop_listening(self, session_id: str) -> None:
        with self._stream_consumers_lock:
            consumer = self._stream_consumers.pop(session_id, None)
        if consumer:
            consumer.stop()

    def _dispatch_event(self, event: str, session_id: str, payload: dict) -> None:
        """
        Handler passed to NodeStreamConsumer — runs on the consumer's
        background thread, so handlers that need the node's asyncio loop
        are scheduled onto it via run_coroutine_threadsafe rather than
        awaited directly here.
        """
        logger.debug(
            f"[{self.label}] Stream event: {event} session={session_id[:8]}"
        )

        if session_id not in self._sessions:
            self._sessions[session_id] = {"session_id": session_id}
        self._sessions[session_id]["last_event"] = event

        handler = {
            "session_open":          self.on_session_open,
            "receiver_joined":       self.on_receiver_joined,
            "measurements_ready":    self.on_measurements_ready,
            "transmission_complete": self.on_transmission_complete,
            "sift_ready":            self.on_sift_ready,
            "key_available":         self.on_key_available,
            "session_aborted":       self.on_session_aborted,
        }.get(event)

        if handler is None:
            logger.warning(f"[{self.label}] Unknown event: {event}")
            return

        try:
            loop = self._main_loop
        except AttributeError:
            loop = None

        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                handler(session_id, payload), loop
            )
        else:
            # Fallback for contexts without a captured running loop (e.g.
            # tests calling _dispatch_event directly) — run synchronously.
            asyncio.run(handler(session_id, payload))

    #Default handlers (override in subclasses)

    async def on_session_open(self, session_id: str, payload: dict) -> None:
        if self.role == NodeRole.RECEIVER:
            await self.join_session(session_id)

    async def on_receiver_joined(self, session_id: str, payload: dict) -> None:
        pass

    async def on_measurements_ready(self, session_id: str, payload: dict) -> None:
        pass

    async def on_transmission_complete(self, session_id: str, payload: dict) -> None:
        """Fired by the Celery chord callback when all qubit batches are done."""
        pass

    async def on_sift_ready(self, session_id: str, payload: dict) -> None:
        pass

    async def on_key_available(self, session_id: str, payload: dict) -> None:
        logger.info(
            f"[{self.label}] Key available session={session_id[:8]} "
            f"QBER={payload.get('qber', 0)*100:.2f}%"
        )
        self.stop_listening(session_id)

    async def on_session_aborted(self, session_id: str, payload: dict) -> None:
        logger.warning(
            f"[{self.label}] Session {session_id[:8]} aborted: "
            f"{payload.get('reason', '')}"
        )
        self._sessions.pop(session_id, None)
        self.stop_listening(session_id)

    #KME helpers

    async def join_session(self, session_id: str) -> dict:
        resp = await self._client.post(
            f"{KME_URL}/sessions/{session_id}/join",
            json={"node_id": self.node_id, "session_id": session_id},
        )
        resp.raise_for_status()
        data = resp.json()
        self._sessions[session_id] = data
        logger.info(
            f"[{self.label}] Joined session={session_id[:8]} "
            f"qkdl={data.get('qkdl_url', 'unknown')}"
        )
        return data

    async def get_session(self, session_id: str) -> dict:
        resp = await self._client.get(f"{KME_URL}/sessions/{session_id}")
        resp.raise_for_status()
        return resp.json()

    async def kme_post(self, path: str, payload: dict) -> dict:
        resp = await self._client.post(f"{KME_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def kme_get(self, path: str) -> dict:
        resp = await self._client.get(f"{KME_URL}{path}")
        resp.raise_for_status()
        return resp.json()

    #Agent loop

    async def _agent_loop(self) -> None:
        while self._running:
            try:
                self._drain_pending_sessions()
                await self._poll_tick()
            except Exception as e:
                logger.debug(f"[{self.label}] Poll tick error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    def _drain_pending_sessions(self) -> None:
        """
        For receiver/monitor roles: pick up any sessions the KME has
        registered as pending for this node_id, and start listening on
        each one's stream immediately (before session_open could possibly
        be missed). Senders don't need this — Alice always learns the
        session_id synchronously from the create_session response and
        calls begin_listening() directly at that point.
        """
        if self.role not in (NodeRole.RECEIVER, NodeRole.MONITOR):
            return
        if not self.node_id:
            return
        try:
            r = get_event_redis()
            from kme.event_bus import pop_pending_sessions
            for session_id in pop_pending_sessions(self.node_id, r=r):
                self.begin_listening(session_id)
        except Exception as e:
            logger.debug(f"[{self.label}] Pending-session drain error: {e}")

    async def _poll_tick(self) -> None:
        pass

    #FastAPI app factory

    def build_app(self, title: str, port: int) -> FastAPI:
        node = self

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await node.start()
            yield
            await node.stop()

        app = FastAPI(title=title, version="1.0.0", lifespan=lifespan)

        #NOTE: the old @app.post("/webhook") endpoint is gone. Events now
        #arrive by this node pulling its session's Redis Stream via
        #begin_listening()/NodeStreamConsumer, started explicitly wherever
        #the node first learns about a session_id (see AliceNode.
        #start_bb84_session, and on_session_open's call to join_session for
        #the receiver/monitor side, which now also calls begin_listening).

        @app.get("/health")
        async def health():
            with node._stream_consumers_lock:
                listening_on = list(node._stream_consumers.keys())
            return {
                "status":   "ok",
                "node_id":  node.node_id,
                "label":    node.label,
                "role":     node.role.value,
                "sessions": list(node._sessions.keys()),
                "listening_on": listening_on,
            }

        return app