#tache de callback du chord apres l'appel du qubit_task worker
import os 
import httpx
import hashlib
import logging
from celery import Task
from workers.celery_config import celery_app
from models import SiftReq, Basis
from bb84_logic import compute_qber, QBER_THRESHOLD
from redis_store import (get_redis, save_session_result, load_all_bob_measurements)

logger=logging.getLogger("worker.sifting")
 
BOB_URL=os.getenv("BOB_URL", "http://localhost:8002")
ORCH_URL=os.getenv("ORCH_URL", "http://localhost:8000")


#first task to be called here => gets res of all the batches
@celery_app.task(
    name="workers.sifting_tasks.assemble_and_sift_task",
    queue="sifting",
)
def assemble_and_sift_task(batch_results: list[dict], session_meta: dict) -> dict:
    import random

    session_id=session_meta["session_id"]
    n_qubits =session_meta["n_qubits"]
    alice_bits= session_meta["alice_bits"]
    alice_bases= session_meta["alice_bases"]

    #1st step: cocher batch as delivered
    delivered_ids: set[int] = set()
    for batch_res in batch_results:
        if batch_res:
            delivered_ids.update(batch_res.get("delivered", []))

    n_delivered= len(delivered_ids)

    logger.info(
        f"[sifting] Session {session_id} — "
        f"delivered={n_delivered}/{n_qubits} qubits"
    )

    #get alice's bases for the deliv qubs
    alice_bases_payload=[
        (qid, alice_bases[qid])
        for qid in sorted(delivered_ids)
        if qid < len(alice_bases)
    ]

    #sample seed transmise à bob pour eviter error qber too high
    sample_seed=random.randint(0, 2**31)

    logger.info(f"SIFT REQ seed type={type(sample_seed)} value={sample_seed}")
    
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{BOB_URL}/sift",
                json=SiftReq(
                    session_id=session_id,
                    alice_bases=alice_bases_payload,
                    sample_seed=sample_seed,
                ).model_dump(),
            )
            resp.raise_for_status()
            sift_data = resp.json()

            #sifted bits de bob
            bob_bits_resp = client.get(
                f"{BOB_URL}/session/{session_id}/sifted-bits"
            )
            bob_bits_resp.raise_for_status()
            bob_sifted= bob_bits_resp.json()["sifted_bits"]

    except Exception as e:
        logger.error(f"[sifting] Error HTTP Bob: {e}")
        return {
            "session_id": session_id,
            "error": str(e),
            "alice_sifted": [],
            "bob_sifted": [],
            "n_delivered": n_delivered,
            "n_qubits": n_qubits,
            "sample_seed": sample_seed,
        }

    #sifting cote d'alice by qub_id
    bob_bases_map: dict[int, str]={
        qid: basis_str
        for qid, basis_str in sift_data["bob_bases"]
    }
 
    alice_sifted=[
        alice_bits[qid]
        for qid in sorted(bob_bases_map.keys())
        if qid in delivered_ids
        and qid<len(alice_bases)
        and alice_bases[qid]==bob_bases_map[qid]
    ]
    n_sifted = len(alice_sifted)
    logger.info(
        f"[sifting] {n_sifted} bits sifted "
        f"({n_sifted/n_qubits*100:.1f}%)"
    )

    return {
        "session_id": session_id,
        "error": None,
        "alice_sifted": alice_sifted,
        "bob_sifted": bob_sifted,
        "n_delivered": n_delivered,
        "n_sifted": n_sifted,
        "n_qubits": n_qubits,
        "sample_seed": sample_seed,
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
            "status": "aborted",
            "error_message": sifting_result["error"],
            "qber": 1.0,
            "key_final": "",
            "n_sifted": 0,
            "n_delivered": sifting_result.get("n_delivered", 0),
            "n_qubits": sifting_result.get("n_qubits", 0),
        }

    alice_sifted = sifting_result["alice_sifted"]
    bob_sifted = sifting_result["bob_sifted"]
    sample_seed = sifting_result["sample_seed"]
    n_sifted=sifting_result["n_sifted"]

    if n_sifted < 10:
        return {
            "session_id": session_id,
            "status": "aborted",
            "error_message": "INSUFFICIENT_BITS",
            "qber": 1.0,
            "key_final": "",
            "n_sifted": n_sifted,
            "n_delivered": sifting_result["n_delivered"],
            "n_qubits": sifting_result["n_qubits"],
        }

    print(len(alice_sifted), len(bob_sifted))
    print(list(zip(alice_sifted[:10], bob_sifted[:10])))

    qber, alice_final, _ = compute_qber(
        alice_sifted, bob_sifted, sample_seed=sample_seed
    )

    logger.info(f"[qber_key] Session {session_id} == QBER={qber*100:.2f}%")

    if qber > QBER_THRESHOLD:
        return{
            "session_id": session_id,
            "status": "aborted",
            "error_message": "QBER_TOO_HIGH",
            "qber": qber,
            "key_final": "",
            "n_sifted": n_sifted,
            "n_delivered": sifting_result["n_delivered"],
            "n_qubits": sifting_result["n_qubits"],
        }

    
    key_final="".join(map(str, alice_final))
    return{
        "session_id": session_id,
        "status": "success",
        "error_message": "",
        "qber": qber,
        "key_final": key_final,
        "n_sifted": n_sifted,
        "n_delivered": sifting_result["n_delivered"],
        "n_qubits": sifting_result["n_qubits"],
    }

#notif orch que le pipeline est done
@celery_app.task(
    bind=True,
    name="workers.notify_tasks.notify_orchestrator_task",
    queue="orchestrator",
    max_retries=5,
    default_retry_delay=2.0,
)
def notify_orchestrator_task(self, pipeline_result:dict) -> None:
    session_id=pipeline_result["session_id"]
    
    try:
        with httpx.Client(timeout=10.0) as client:
            resp=client.post(
                f"{ORCH_URL}/session/{session_id}/complete",
                json=pipeline_result,
            )
            resp.raise_for_status()
        logger.info(f"[notify] Session {session_id} notified ---> orchestrateur")
    
    except httpx.HTTPError as e:
        logger.warning(f"[notify] Retry notification {session_id}: {e}")
        raise self.retry(exc=e)