#tache de callback du chord apres l'appel du qubit_task worker
import os 
import httpx
import random
import logging
from celery import Task
from workers.celery_config import celery_app
from models import SiftReq, Basis
from bb84_logic import compute_qber, QBER_THRESHOLD

logger=logging.getLogger("worker.sifting")
 
BOB_URL=os.getenv("BOB_URL", "http://localhost:8002")
QNS_URL= os.getenv("QNS_URL", "http://localhost:8003")
 
@celery_app.task(
    name="workers.sifting_tasks.sifting_task",
    queue="sifting",
)
def sifting_task(qubit_results:list[dict], session_meta:dict)-> dict:
    session_id=session_meta["session_id"]
    n_qubits=session_meta["n_qubits"]
    logger.info(
        f"[sifting] Session {session_id} — "
        f"{len(qubit_results)} received results"
    )

    #check which qubits are delivered
    delivered={r["qubit_id"]:r for r in qubit_results if r and r.get("delivered")}
    n_delivered=len(delivered)
    logger.info(f"[sifting] {n_delivered}/{n_qubits} qubits delivered")

    #construction des bases d'alice pour sifting
    alice_bases_payload=[(qid, r["basis"]) for qid, r in sorted(delivered.items())]

    #alice sends bases to bob via classical canal
    try:
        with httpx.Client(timeout=30.0) as client:
            resp=client.post(f"{BOB_URL}/sift", json=SiftReq(session_id=session_id, alice_bases=alice_bases_payload).model_dump(),)
            resp.raise_for_status()
            sift_data=resp.json()    
            
            bob_bits_resp=client.get(f"{BOB_URL}/session/{session_id}/sifted-bits")
            bob_bits_resp.raise_for_status()
            bob_sifted=bob_bits_resp.json()["sifted_bits"]
    
    except httpx.HTTPError as e:
        logger.error(f"[sifting] Error HTTP Bob: {e}")
        return {
            "session_id": session_id,
            "error": f"Bob unreachable: {e}",
            "alice_sifted": [],
            "bob_sifted": [],
        }

    #sifting du cote d'alice
    bob_bases_map={qid:Basis(b) for qid, b in sift_data["bob_bases"]}

    alice_sifted=[
        delivered[qid]["bit"] for qid in sorted(bob_bases_map.keys())
        if qid in delivered and Basis(delivered[qid]["basis"])==bob_bases_map[qid]
    ]

    n_sifted=len(alice_sifted)
    logger.info(
        f"[sifting] {n_sifted} bits sifted "
        f"({n_sifted/n_qubits*100:.1f}%)"
    )
    return {
        "session_id":session_id,
        "alice_sifted": alice_sifted,
        "bob_sifted": bob_sifted,
        "n_delivered":  n_delivered,
        "n_sifted": n_sifted,
        "n_qubits":n_qubits,
        "error":None,
    }
 
@celery_app.task(
    name="workers.sifting_tasks.qber_key_task",
    queue="sifting",
)
def qber_key_task(sifting_result: dict) -> dict:
    session_id = sifting_result["session_id"]
    if sifting_result.get("error"):
        return {
            "session_id": session_id,
            "statut": "aborted",
            "error_message": sifting_result["error"],
            "qber": 1.0,
            "key_final": "",
            "n_sifted": 0,
        }
    alice_sifted = sifting_result["alice_sifted"]
    bob_sifted   = sifting_result["bob_sifted"]
    n_sifted     = sifting_result["n_sifted"]

    if n_sifted < 10:
        return {
            "session_id":    session_id,
            "statut":        "aborted",
            "error_message": "INSUFFICIENT_BITS",
            "qber":          1.0,
            "key_final":      "",
            "n_sifted":      n_sifted,
            "n_delivered":   sifting_result["n_delivered"],
            "n_qubits":      sifting_result["n_qubits"],
        }
    
    #qber calculation en attendant d'en faire un service 

    #also sample_seed here is generated (deterministe) mais en vrai elle doit etre negociée via canal classic
    seed=hash(session_id) & 0x7FFFFFFF
    qber, alice_final, _=compute_qber(
        alice_sifted, bob_sifted, sample_seed=seed
    )
    logger.info(f"[qber_key] Session {session_id} — QBER={qber*100:.2f}%")
    if qber > QBER_THRESHOLD:
        return {
            "session_id":    session_id,
            "statut":        "aborted",
            "error_message": "QBER_TOO_HIGH",
            "qber":          qber,
            "key_final":      "",
            "n_sifted":      n_sifted,
            "n_delivered":   sifting_result["n_delivered"],
            "n_qubits":      sifting_result["n_qubits"],
        }
    key_final = alice_final
    logger.info(f"[qber_key] Final key : {key_final[:16]}...")
 
    return {
        "session_id":    session_id,
        "statut":        "success",
        "error_message": "",
        "qber":          qber,
        "key_final":      key_final,
        "n_sifted":      n_sifted,
        "n_delivered":   sifting_result["n_delivered"],
        "n_qubits":      sifting_result["n_qubits"],
    }