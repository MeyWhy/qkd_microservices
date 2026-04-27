import json
import os
import redis
from typing import Optional
from state_machine import OrchestratorSession, SessionStatus

REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL=7200  

def get_redis()->redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _key(session_id: str) -> str:
    return f"orch:session:{session_id}"

def save_orch_session(r: redis.Redis, session: OrchestratorSession) -> None:
    r.set(_key(session.session_id), session.model_dump_json(), ex=SESSION_TTL)
    r.sadd("orch:sessions:all", session.session_id)
    if not session.is_terminal:
        r.sadd("orch:sessions:active", session.session_id)
    else:
        r.srem("orch:sessions:active", session.session_id)


def load_orch_session(
    r: redis.Redis, session_id: str
) -> Optional[OrchestratorSession]:
    raw=r.get(_key(session_id))
    if not raw:
        return None
    return OrchestratorSession.model_validate_json(raw)


def update_orch_session(r: redis.Redis,session: OrchestratorSession,) -> None:
    pipe=r.pipeline()
    pipe.set(_key(session.session_id), session.model_dump_json(), ex=SESSION_TTL)
    if session.is_terminal:
        pipe.srem("orch:sessions:active", session.session_id)
    pipe.execute()

def list_active_sessions(r: redis.Redis) -> list[str]:
    return list(r.smembers("orch:sessions:active"))

def list_all_sessions(r: redis.Redis) -> list[str]:
    return list(r.smembers("orch:sessions:all"))