"""
kme/message_bus.py
===================
Redis pub/sub wrapper for real-time event delivery between the KME and nodes.

This is the second delivery mechanism alongside webhooks:

  Webhooks (HTTP POST to node callback_url)
    + Low latency when delivered
    + No persistent connection needed
    - Fire-and-forget: no guarantee of delivery
    - Node must have a reachable HTTP endpoint

  Message bus (Redis pub/sub, this file)
    + Node subscribes and receives events in real time
    + Works even when the node has no public HTTP endpoint
    + Survives transient network glitches (subscriber reconnects)
    - Requires a persistent TCP connection to Redis
    - Messages are lost if no subscriber is listening at publish time

How they work together:
  KME always publishes to the bus AND sends a webhook.
  The node receives whichever arrives first.
  The polling fallback in BaseNode catches anything that slips through both.

Channel naming:
  kme:events:{node_id}      → events for a specific node
  kme:events:broadcast      → events for all nodes (e.g. topology changes)
  kme:session:{session_id}  → events scoped to a session (any participant)
"""

import json
import logging
import os
import threading
from typing import Callable, Optional

import redis

logger   = logging.getLogger("kme.bus")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ─────────────────────────────────────────────
# Publisher (used by KME)
# ─────────────────────────────────────────────

class MessageBus:
    """
    Publishes events to Redis channels.
    Used by the KME to notify nodes of session state changes.
    """

    def __init__(self):
        self._r = redis.from_url(REDIS_URL, decode_responses=True)

    def publish(
        self,
        channel: str,
        event:   str,
        session_id: str,
        payload: dict = {},
    ) -> int:
        """
        Publishes one event to a Redis channel.
        Returns the number of subscribers that received it.
        0 means no active subscriber — webhooks are the fallback.
        """
        message = json.dumps({
            "event":      event,
            "session_id": session_id,
            "payload":    payload,
        })
        n = self._r.publish(channel, message)
        logger.debug(f"[Bus] Published '{event}' → {channel} ({n} receivers)")
        return n

    def publish_to_node(
        self,
        node_id:    str,
        event:      str,
        session_id: str,
        payload:    dict = {},
    ) -> int:
        """Publishes an event to a specific node's private channel."""
        return self.publish(
            channel=f"kme:events:{node_id}",
            event=event,
            session_id=session_id,
            payload=payload,
        )

    def publish_to_session(
        self,
        session_id: str,
        event:      str,
        payload:    dict = {},
    ) -> int:
        """
        Publishes to the session channel.
        All nodes participating in the session receive this.
        """
        return self.publish(
            channel=f"kme:session:{session_id}",
            event=event,
            session_id=session_id,
            payload=payload,
        )

    def broadcast(self, event: str, payload: dict = {}) -> int:
        """Publishes to all nodes (topology changes, KME restarts, etc.)."""
        return self.publish(
            channel="kme:events:broadcast",
            event=event,
            session_id="",
            payload=payload,
        )


# ─────────────────────────────────────────────
# Subscriber (used by nodes)
# ─────────────────────────────────────────────

class NodeSubscriber:
    """
    Subscribes to Redis channels and calls a handler for each message.
    Runs in a background thread — non-blocking for the node's event loop.

    Usage in a node:
        sub = NodeSubscriber(node_id="abc-123", handler=my_callback)
        sub.start()
        # ... node runs ...
        sub.stop()

    The handler receives (event: str, session_id: str, payload: dict).
    """

    def __init__(
        self,
        node_id: str,
        handler: Callable[[str, str, dict], None],
    ):
        self.node_id  = node_id
        self.handler  = handler
        self._thread: Optional[threading.Thread] = None
        self._stop    = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._listen,
            daemon=True,
            name=f"bus-sub-{self.node_id[:8]}",
        )
        self._thread.start()
        logger.info(f"[Bus] Subscriber started for node {self.node_id[:8]}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _listen(self) -> None:
        """
        Blocking Redis pub/sub loop.
        Reconnects automatically on connection errors.
        """
        while not self._stop.is_set():
            try:
                r      = redis.from_url(REDIS_URL, decode_responses=True)
                pubsub = r.pubsub(ignore_subscribe_messages=True)

                # Subscribe to node-specific, broadcast, and session channels
                pubsub.subscribe(
                    f"kme:events:{self.node_id}",
                    "kme:events:broadcast",
                )

                for raw in pubsub.listen():
                    if self._stop.is_set():
                        break
                    if raw["type"] != "message":
                        continue
                    try:
                        msg = json.loads(raw["data"])
                        self.handler(
                            msg.get("event", ""),
                            msg.get("session_id", ""),
                            msg.get("payload", {}),
                        )
                    except Exception as e:
                        logger.warning(f"[Bus] Handler error: {e}")

            except Exception as e:
                if not self._stop.is_set():
                    logger.warning(f"[Bus] Redis disconnect, reconnecting: {e}")
                    import time
                    time.sleep(1.0)

    def subscribe_session(self, session_id: str) -> None:
        """
        Dynamically add a session channel subscription.
        Called when the node joins a session.
        Note: requires a new pubsub connection or psubscribe.
        Simple implementation: just re-listen (restart thread).
        """
        # In a full implementation, use psubscribe("kme:session:*")
        # For now, the node-specific channel carries all session events.
        pass