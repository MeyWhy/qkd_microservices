import json
import os
import redis
from typing import Optional 
from models import SessionMeta, QubitMeasurement

REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0")
SESSION_TTL=3600  

def get_redis()->redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=True)


def _key_meta(sid: str)      -> str: return f"session:{sid}:meta"
def _key_bob_meas(sid: str)  -> str: return f"session:{sid}:bob_meas"
def _key_delivered(sid: str) -> str: return f"session:{sid}:delivered"
def _key_result(sid: str)    -> str: return f"session:{sid}:result"

def save_session_meta(r: redis.Redis, meta: SessionMeta)->None:
    r.set(_key_meta(meta.session_id), meta.model_dump_json(), ex=SESSION_TTL)

def load_session_meta(r: redis.Redis, session_id: str) -> Optional[SessionMeta]:
    raw=r.get(_key_meta(session_id))
    if not raw:
        return None
    return SessionMeta.model_validate_json(raw)
 

def update_session_status(r: redis.Redis, session_id: str, statut: str) -> None:
    meta=load_session_meta(r, session_id)
    if meta:
        meta.statut=statut
        save_session_meta(r, meta)


def save_bob_measurement(
    r:redis.Redis,
    session_id: str,
    measurement:QubitMeasurement,)-> None:
    r.hset(
        _key_bob_meas(session_id),
        str(measurement.qubit_id),
        measurement.model_dump_json(),
    )
    r.expire(_key_bob_meas(session_id), SESSION_TTL)
 

def save_bob_measurements_batch(
    r: redis.Redis,
    session_id: str,
    measurements: list[QubitMeasurement],
) -> None:
    if not measurements:
        return
    mapping={
        str(m.qubit_id): m.model_dump_json()
        for m in measurements
    }
    r.hset(_key_bob_meas(session_id), mapping=mapping)
    r.expire(_key_bob_meas(session_id), SESSION_TTL)


def load_all_bob_measurements(
    r: redis.Redis,
    session_id: str,
) -> dict[int, QubitMeasurement]:
    raw=r.hgetall(_key_bob_meas(session_id))
    return {
        int(qid): QubitMeasurement.model_validate_json(val)
        for qid, val in raw.items()
    }



def mark_delivered(r: redis.Redis, session_id: str, qubit_ids: list[int]) -> None:
    if qubit_ids:
        r.sadd(_key_delivered(session_id), *[str(q) for q in qubit_ids])
        r.expire(_key_delivered(session_id), SESSION_TTL)
 
def get_delivered_ids(r: redis.Redis, session_id: str) -> set[int]:
    return {int(x) for x in r.smembers(_key_delivered(session_id))}

def save_session_result(r: redis.Redis, session_id: str, result: dict) -> None:
    r.set(_key_result(session_id), json.dumps(result), ex=SESSION_TTL)

def load_session_result(r: redis.Redis, session_id: str) -> Optional[dict]:
    raw=r.get(_key_result(session_id))
    return json.loads(raw) if raw else None
 
def delete_session(r: redis.Redis, session_id: str) -> None:
    keys=[
        _key_meta(session_id),
        _key_bob_meas(session_id),
        _key_delivered(session_id),
        _key_result(session_id),
    ]
    r.delete(*keys)