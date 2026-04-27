import os 
import httpx
import logging
from workers.celery_config import celery_app
from models import QubitBatch

logger = logging.getLogger("worker.qubit")
QNS_URL = os.getenv("QNS_URL", "http://localhost:8003")

@celery_app.task(
    bind=True,
    name="workers.qubit_tasks.send_batch_task",
    max_retries=2,
    default_retry_delay=1.0,
    queue="qubit_send",
)
def send_batch_task(self, session_id: str, batch_payload: dict) -> dict:
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{QNS_URL}/batch/send",
                json={"session_id": session_id, "batch": batch_payload},
            )
            resp.raise_for_status()
            data = resp.json()

        results=data.get("results") or []

        delivered = [r["qubit_id"] for r in results if r.get("delivered")]
        failed    = [r["qubit_id"] for r in results if not r.get("delivered")]

        logger.debug(
            f"[batch {data['batch_id']}] "
            f"delivered={len(delivered)} failed={len(failed)}"
        )

        return {
            "session_id": session_id,
            "batch_id":   data["batch_id"],
            "delivered":  delivered,
            "failed":     failed,
        }

    except httpx.HTTPStatusError as e:
        logger.warning(
            f"[batch] HTTP {e.response.status_code} "
            f"session={session_id} — retry {self.request.retries}"
        )
        raise self.retry(exc=e)

    except httpx.RequestError as e:
        logger.warning(f"[batch] QNS indisponible: {e}")
        raise self.retry(exc=e)

    except Exception as e:
        logger.error(f"[batch] Erreur inattendue: {e}")
        batch = QubitBatch.model_validate(batch_payload)
        return {
            "session_id": session_id,
            "batch_id":   batch.batch_id,
            "delivered":  [],
            "failed":     [q.qubit_id for q in batch.qubits],
        }