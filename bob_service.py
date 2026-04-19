from fastapi import FastAPI
import threading
import time
import queue
from collections import defaultdict
from contextlib import asynccontextmanager
import redis

from quantum_core.redis_config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    STREAM_NAME, CONSUMER_GROUP, CONSUMER_NAME,
    READ_BLOCK_MS, READ_COUNT,
)
from quantum_core.qunetsim_service import transmit_qubit_event

app=FastAPI(title="Bob service v3")

#shared state== une db en memoire (volatile): perspec store it for comparison-TODO-  
_results: dict[str, dict[int, int]]=defaultdict(dict)
#lock it car shared var by threads
_results_lock: threading.Lock=threading.Lock()
#synch event driven archi so session has an event
_session_events:dict[str, threading.Event]=defaultdict(threading.Event)

#consumer: 
_work_queue: queue.Queue=queue.Queue(maxsize=512)
#producer: 
_redis=redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

#pour avoir redis stream grp
def ensure_consumer_grp()-> None:
    try:
        _redis.xgroup_create(STREAM_NAME, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            print(f"[bob_service] consumer group {CONSUMER_GROUP} already exists")
        else:
            raise


def _consumer_loop()-> None:
    ensure_consumer_grp()
    print("[bob_service] consumer loop started")

    while True:
        try:#read new msg from redis stream 
            messages=_redis.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={STREAM_NAME:">"},
                count=READ_COUNT,
                block=READ_BLOCK_MS,
            )

            if not messages:
                continue

            for stream, entries in messages:
                for msg_id, fields in entries:
                    event={
                        "session_id":fields["session_id"],
                        "index":int(fields["index"]),
                        "bit":int(fields["bit"]),
                        "alice_basis":int(fields["alice_basis"]),
                        "bob_basis":int(fields["bob_basis"]),}
                    try:
                        _work_queue.put(event, block=True, timeout=5)
                    except queue.Full:
                        print("[bob_service] work queue full, dropping event")
                        continue
                    #ack avant process pour dim latence ici => au lieu de ack apres avoir process msg
                    _redis.xack(STREAM_NAME, CONSUMER_GROUP, msg_id)

        except Exception as exc:
            print(f"[bob_service] consumer loop error: {exc}")
            time.sleep(1)

#kinda process msg: qubit
def _quantum_worker()-> None:
    print("[bob_service] quantum worker started")

    while True:
        try:#tq ya pas d'arrives
            event=_work_queue.get(timeout=1)
        except queue.Empty:
            continue

        try:#appel vers qunetsim service
            measured_bit=transmit_qubit_event(event)
        except Exception as exc:
            print(f"[bob_service] qubit failed: {exc}")
            _work_queue.task_done()
            continue

        session_id=event["session_id"]
        index=event["index"]

        with _results_lock:
            _results[session_id][index]=measured_bit
        #calls for /wait here
        _session_events[session_id].set()
        _work_queue.task_done() 

        print(f"[bob_service] qubit {index}, session={session_id} -> {measured_bit}")

#consumer_t put into queue
#et quantum worker recup de file and call qunetsim pour traiter
@asynccontextmanager
async def lifespan(app:FastAPI):
    t1=threading.Thread(target=_consumer_loop, daemon=True, name="bob-consumer")
    t2=threading.Thread(target=_quantum_worker, daemon=True, name="bob-quantum")

    t1.start()
    t2.start()

    print("[bob_service] threads started (consumer + quantum worker)")
    yield


app.router.lifespan_context=lifespan


@app.get("/health")
def health():
    return {"statut":"ok","queue_size":_work_queue.qsize()}


@app.get("/results/{session_id}")
def get_results(session_id:str):
    with _results_lock:
        data=dict(_results.get(session_id,{}))
    return {str(k):v for k,v in data.items()}

#soluce pour vider buffer
@app.delete("/results/{session_id}")
def clear_results(session_id:str):
    with _results_lock:
        removed=_results.pop(session_id,{})
    _session_events.pop(session_id,None)
    return {"session_id":session_id,"cleared":len(removed)}

#permet de stop when enough qubit(tq ya pas de new, ou ya timeout)
@app.get("/results/{session_id}/wait")
def wait_for_results(session_id:str,size:int,timeout:float=30.0):
    deadline=time.monotonic()+timeout
    ev=_session_events[session_id]

    while True:
        with _results_lock:
            current=dict(_results.get(session_id,{}))

        if len(current)>=size:
            break

        remaining=deadline-time.monotonic()
        if remaining<=0:
            break

        ev.clear()
        ev.wait(timeout=min(remaining,1.0))

    with _results_lock:
        final=dict(_results.get(session_id,{}))

    return {
        "ready":len(final)>=size,
        "count":len(final),
        "results":{str(k):v for k,v in final.items()},
    }
