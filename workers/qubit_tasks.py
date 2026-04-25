import os 
import httpx
import logging
from celery import Task
 
from workers.celery_config import celery_app
from models import SendQubitReq, Basis

logger=logging.getLogger("worker.qubit")
QNS_URL=os.getenv("QNS_URL", "http://localhost:8003")

class QubitTask(Task):
    abstract=True
    _client=None

    @property
    def client(self)-> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client=httpx.Client(timeout=10.0)
        return self._client
    
@celery_app.task(
    bind=True,
    base=QubitTask,
    name="workers.qubit_tasks.send_qubit_task",
    max_retries=2,
    default_retry_delay=0.5,
    queue="qubit_send",
)
def send_qubit_task(self, session_id:str, qubit_id:int, bit:int, basis:str)-> dict:
    try:
        payload=SendQubitReq(session_id=session_id, qubit_id=qubit_id, bit=bit, basis=Basis(basis),).model_dump()
        payload["basis"]=payload["basis"]
        resp=self.client.post(f"{QNS_URL}/qubit/send", json=payload,)
        resp.raise_for_status()
        data=resp.json()

        logger.debug(f"[qubit {qubit_id}] delivered={data.get('delivered')}")

        return{
            "qubit_id":qubit_id,
            "delivered":data.get("delivered", False),
            "bit":bit,
            "basis":basis,
            "session_id": session_id,
        }
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"[qubit {qubit_id}] HTTP {e.response.status_code}-  "
            f"retry {self.request.retries}/{self.max_retries}"
        )
        raise self.retry(exc=e)
    
    except httpx.RequestError as e:
        logger.warning(f"[qubit {qubit_id}] QNS not available: {exc}")
        raise self.retry(exc=e)

    except Exception as exc:

        logger.error(f"[qubit {qubit_id}] Unexpected error: {exc}")
        return {
            "qubit_id":   qubit_id,
            "delivered":  False,
            "bit":        bit,
            "basis":      basis,
            "session_id": session_id,
            "error":      str(exc),
        }

