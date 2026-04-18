import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from quantum_core.celery_app import celery_app
from quantum_core.tasks import transmit_qubit

app=FastAPI(title="Alice Service")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
class AliceReq(BaseModel):
    alice_bits:List[int]
    alice_basis:List[int]
    bob_basis:List[int]

@app.get("/health")
def health():
    return {"statut":"ok"}

@app.post("/transmit")
def transmit(req:AliceReq):
    if not (len(req.alice_bits)==len(req.alice_basis)==len(req.bob_basis)):
        return {"error":"all the input variables must have equal length"}
    n= len(req.alice_bits)
    print(f"[alice_service] sending {n} qubits...")
    task_ids=[]

    for i in range(n):
        task=transmit_qubit.delay(bit=req.alice_bits[i],alice_basis=req.alice_basis[i],bob_basis=req.bob_basis[i],index=i,)
        task_ids.append(task.id)

    print(f"[alice_service] all {n} tasks dispatched")
    return {"task_ids":task_ids,"count":n}
