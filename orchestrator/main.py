import uuid
import logging
import requests
from fastapi import FastAPI, Query

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

app=FastAPI(title="orchestrator v3")

BIT_SERVICE_URL="http://localhost:8109"
BASIS_SERVICE_URL="http://localhost:8101"
ALICE_SERVICE_URL="http://localhost:8103"
BOB_SERVICE_URL="http://localhost:8104"
SIFTING_SERVICE_URL="http://localhost:8102"
QBER_SERVICE_URL="http://localhost:8106"
KEYRATE_SERVICE_URL="http://localhost:8107"
LATENCY_SERVICE_URL="http://localhost:8108"

#FIX FOR READTIMEOUTERRORRRR
BOB_WAIT_TIMEOUT=90

#xrappers
def get(url, params=None, timeout=10):
    resp=requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def post(url, data, timeout=30):
    resp=requests.post(url, json=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/run")
def run_bb84(size: int = Query(default=16, ge=1, le=512)):
    session_id=str(uuid.uuid4())
    logger.info("[orchestrator] session=%s n=%d START", session_id, size)

    alice_bits=get(f"{BIT_SERVICE_URL}/bits",{"n":size})["bits"]
    alice_basis=get(f"{BASIS_SERVICE_URL}/basis",{"n":size})["basis"]
    bob_basis=get(f"{BASIS_SERVICE_URL}/basis",{"n":size})["basis"]

    post(f"{LATENCY_SERVICE_URL}/start",{"session_id":session_id})

    #alice sends qubits ===>> async publish event to redis
    logger.info("[orchestrator] sending qubit stream …")
    post(
        f"{ALICE_SERVICE_URL}/transmit",
        {
            "session_id":session_id,
            "alice_bits":alice_bits,
            "alice_basis":alice_basis,
            "bob_basis":bob_basis,
        },
        timeout=30,
    )

    #wait for bob results
    logger.info("[orchestrator] waiting for Bob results …")

    bob_wait=get(
        f"{BOB_SERVICE_URL}/results/{session_id}/wait",
        params={"size":size,"timeout":BOB_WAIT_TIMEOUT},
        timeout=BOB_WAIT_TIMEOUT+30,
    )

    if not bob_wait.get("ready"):
        logger.warning(
            "[orchestrator] partial results: %d/%d",
            bob_wait.get("count",0), size
        )

    bob_dict=bob_wait.get("results",{})
    bob_list=[bob_dict.get(str(i)) for i in range(size)]

    latency_resp=post(f"{LATENCY_SERVICE_URL}/stop",{"session_id":session_id})
    latency_seconds=latency_resp.get("latency_seconds")

    sift_resp=post(
        f"{SIFTING_SERVICE_URL}/sift",
        {
            "alice_bits":alice_bits,
            "alice_basis":alice_basis,
            "bob_basis":bob_basis,
            "bob_res":bob_list,
        },
    )

    alice_sifted=sift_resp["alice_sifted"]
    bob_sifted=sift_resp["bob_sifted"]

    qber_resp=post(
        f"{QBER_SERVICE_URL}/qber",
        {"alice_sifted":alice_sifted,"bob_sifted":bob_sifted},
    )
    qber=qber_resp.get("qber")

    kr=post(
        f"{KEYRATE_SERVICE_URL}/keyrate",
        {
            "total_bits":size,
            "sifted_length":sift_resp["sifted_length"],
            "qber":qber if qber is not None else 0.0,
        },
    )

    sifted_key="".join(str(b) for b in alice_sifted)

    #pour eviter blem read from the buffer => bob delete session 
    try:
        requests.delete(
            f"{BOB_SERVICE_URL}/results/{session_id}",
            timeout=5
        )
    except Exception:
        pass

    logger.info(
        "[orchestrator] session=%s DONE key=%s qber=%.4f latency=%.2fs",
        session_id, sifted_key, qber or 0, latency_seconds or 0
    )

    return {
        "session_id":session_id,
        "n_qubits":size,

        "alice_bits":alice_bits,
        "alice_basis":alice_basis,
        "bob_basis":bob_basis,

        "bob_res":bob_list,

        "sifted_key":sifted_key,
        "sifted_length":sift_resp["sifted_length"],

        "qber":qber,
        "key_rate":kr.get("key_rate"),
        "final_key_bits":kr.get("final_key_bits"),

        "latency_seconds":latency_seconds,
    }
