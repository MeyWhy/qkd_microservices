import uuid
import requests
from fastapi import FastAPI, Query
app=FastAPI(title="Orchestrator")

BIT_SERVICE_URL="http://localhost:8100"
BASIS_SERVICE_URL="http://localhost:8101"
ALICE_SERVICE_URL="http://localhost:8103"
BOB_SERVICE_URL="http://localhost:8104"
SIFTING_SERVICE_URL="http://localhost:8102"
QBER_SERVICE_URL="http://localhost:8106"
KEYRATE_SERVICE_URL="http://localhost:8107"
LATENCY_SERVICE_URL="http://localhost:8108"


def get(url, params=None):
    resp=requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def post(url, data):
    resp=requests.post(url, json=data, timeout=120)
    resp.raise_for_status()
    return resp.json()

@app.get("/health")
def health():
    return {"statut": "ok"}

@app.post("/run")
def run_bb84(size: int = Query(default=16, ge=1, le=256)):
    session_id = str(uuid.uuid4())

    print(f"\n[orchestrator] session {session_id} | n={size} ")
    
    print("[orchestrator] Step 1: Generating bits and bases...")
    alice_bits=get(f"{BIT_SERVICE_URL}/bits",{"n": size})["bits"]
    alice_basis=get(f"{BASIS_SERVICE_URL}/basis", {"n": size})["basis"]
    bob_basis=get(f"{BASIS_SERVICE_URL}/basis", {"n": size})["basis"]

    post(f"{LATENCY_SERVICE_URL}/start", {"session_id":session_id})
    
    print("[orchestrator] Step 3: Transmitting qubits through Alice service...")
    alice_resp=post(f"{ALICE_SERVICE_URL}/transmit",{"alice_bits":  alice_bits,"alice_basis": alice_basis,"bob_basis":   bob_basis,},)
    task_ids=alice_resp["task_ids"]
    
    print("[orchestrator] Step 4:Collecting Bob's measurements...")
    collect_resp=post(f"{BOB_SERVICE_URL}/collect",{"task_ids": task_ids, "timeout": 60.0},)
    bob_res=collect_resp["bob_res"]
    latency_resp=post(f"{LATENCY_SERVICE_URL}/stop", {"session_id": session_id})
    latency_seconds=latency_resp.get("latency_seconds", None)
    
    print("[orchestrator] Step 6: Sifting key...")
    sift_resp=post(f"{SIFTING_SERVICE_URL}/sift",{"alice_bits":  alice_bits,"alice_basis": alice_basis,"bob_basis":   bob_basis,"bob_res": bob_res,},)
    alice_sifted=sift_resp["alice_sifted"]
    bob_sifted=sift_resp["bob_sifted"]

    print("[orchestrator] Step 7: Computing QBER...")
    q=post(f"{QBER_SERVICE_URL}/qber",{"alice_sifted": alice_sifted, "bob_sifted": bob_sifted},)
    qber=q.get("qber")

    print("[orchestrator] Step 8: Computing key rate...")
    kr=post(f"{KEYRATE_SERVICE_URL}/keyrate" ,{"total_bits":size,"sifted_length":  sift_resp["sifted_length"],"qber":qber if qber is not None else 0.0,},)
    key_rate=kr.get("key_rate")
    sifted_key="".join(str(b) for b in alice_sifted)
   
    print(f"[orchestrator] session {session_id} DONEEE\n")
    return {
        "session_id":session_id,
        "n_qubits":size,
        "alice_bits":alice_bits,
        "alice_basis":alice_basis,
        "bob_basis":bob_basis,
        "bob_res":bob_res,
        "sifted_key":sifted_key,
        "sifted_length":sift_resp["sifted_length"],
        "qber":qber,
        "key_rate":key_rate,
        "final_key_bits":kr.get("final_key_bits"),
        "latency_seconds":latency_seconds,
    }
