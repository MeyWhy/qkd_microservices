import json
import logging
import os
import time
from typing import Optional

import redis

from models import NodeInfo, NodeRole, NodeRegistration

logger = logging.getLogger("kme.registry")
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379/0")
NODE_TTL    = 86400   #24h nodes re-register on startup


def get_redis() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _kn(node_id: str) -> str: return f"kme:node:{node_id}"



def register_node(r: redis.Redis, reg: NodeRegistration) -> NodeInfo:
    import uuid
    node_id = str(uuid.uuid4())
    info = NodeInfo(
        node_id=node_id,
        role=reg.role,
        callback_url=reg.callback_url,
        label=reg.label or f"{reg.role.value}-{node_id[:8]}",
        metadata=reg.metadata,
        registered_at=time.time(),
    )
    r.set(_kn(node_id), info.model_dump_json(), ex=NODE_TTL)
    r.sadd("kme:nodes:all",          node_id)
    r.sadd(f"kme:nodes:{reg.role.value}", node_id)
    logger.info(f"[Registry] Node registered: {info.label} ({node_id[:8]})")
    return info


def load_node(r: redis.Redis, node_id: str) -> Optional[NodeInfo]:
    raw = r.get(_kn(node_id))
    return NodeInfo.model_validate_json(raw) if raw else None


def find_node_by_label(r: redis.Redis, label: str) -> Optional[NodeInfo]:
    all_ids = r.smembers("kme:nodes:all")
    for nid in all_ids:
        node = load_node(r, nid)
        if node and node.label == label:
            return node
    return None


def list_nodes(r: redis.Redis,
               role: Optional[NodeRole] = None) -> list[NodeInfo]:
    if role:
        ids = r.smembers(f"kme:nodes:{role.value}")
    else:
        ids = r.smembers("kme:nodes:all")
    nodes = []
    for nid in ids:
        n = load_node(r, nid)
        if n:
            nodes.append(n)
    return nodes


#NOTE: notify_node() has been removed. It performed an httpx POST to a
#node's callback_url ("/webhook") — the push-based delivery path that this
#migration replaces entirely with kme/event_bus.py's Redis Streams
#publish_event()/NodeStreamConsumer pull-based model (see kme/event_bus.py
#module docstring for the full rationale). callback_url is still accepted
#on NodeRegistration and stored for backward-compatible logging/dashboards,
#but the KME no longer dials it for anything.
#
#kme/message_bus.py (the old, never-wired-in Pub/Sub MessageBus/
#NodeSubscriber) is likewise superseded by kme/event_bus.py and should be
#considered dead code — left in place only so a `git log` / diff shows the
#exact migration path rather than a silent deletion. Safe to delete once
#this migration is confirmed working end-to-end.