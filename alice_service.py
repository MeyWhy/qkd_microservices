from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import redis
from quantum_core.redis_config import (REDIS_HOST, REDIS_PORT, REDIS_DB,STREAM_NAME, CONSUMER_GROUP, CONSUMER_NAME,READ_BLOCK_MS, READ_COUNT,)

app=FastAPI(title="Alice Service v2")
_redis=redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

class AliceReq(BaseModel):
    session_id:str
    alice_bits:List[int]
    alice_basis:List[int]
    bob_basis:List[int]

#publish msg to redis stream
#xadd distribuer task submitted
def publish_qubit(session_id: str, index: int,bit: int, alice_basis: int, bob_basis: int) ->str:
    msg_id=_redis.xadd(STREAM_NAME, 
        {"session_id":session_id,
        "index":str(index),
        "bit":str(bit),
        "alice_basis":str(alice_basis),
        "bob_basis":str(bob_basis),}
    )
    return msg_id

@app.get("/health")
def health():
    return {"statut":"ok"}

@app.post("/transmit")
def transmit(req:AliceReq):
    size=len(req.alice_bits)
    if not (size==len(req.alice_basis)==len(req.bob_basis)):
        raise HTTPException(status_code=422, detail="input must have equal length",)
    print(f"[alice_service] publishing session={req.session_id} sending {size} qubits")
    msg_ids=[]#gives us distrib async tasks
    for i in range(size):
        mid=publish_qubit(session_id= req.session_id, index=i, bit=req.alice_bits[i], alice_basis=req.alice_basis[i], bob_basis=req.bob_basis[i],)
        msg_ids.append(mid)
    print(f"[alice_service] all donee")
    return {"session_id":req.session_id,"published":size, "msg_ids": msg_ids}

