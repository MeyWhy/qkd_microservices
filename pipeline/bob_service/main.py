import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from celery.result import AsyncResult
from quantum_core.celery_app import celery_app

app=FastAPI(title="Bob service")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BobReq(BaseModel):
    task_ids:List[str]
    timeout: float=30.0   #time in s to wait per each task

@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/collect")
def collect_results(req: BobReq):
    print(f"[bob_service] collecting {len(req.task_ids)} results...")
    bob_res=[]
    for task_id in req.task_ids:
        res=AsyncResult(task_id, app=celery_app)
        try:
            val=res.get(timeout=req.timeout)   #here it blocks until its done
            bob_res.append(val)
        except Exception as e:
            print(f"[bob_service] Task {task_id} failed: {e}")
            bob_res.append(None)   #mark it as lost qubit
    print(f"[bob_service] collected : {bob_res}")
    return {"bob_res": bob_res}
