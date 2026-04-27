import os
from celery import Celery

REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app=Celery(
    "bb84",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    result_backend_transport_options={
        "retry_policy":{"timeout":5.0}
    },
    task_routes={
        "workers.qubit_tasks.send_batch_task":  {"queue": "qubit_send"},
        "workers.sifting_tasks.assemble_and_sift_task":   {"queue": "sifting"},
        "workers.sifting_tasks.qber_key_task":  {"queue": "sifting"},
        "workers.notify_tasks.notify_orchestrator_task":{"queue": "orchestrator"},
    }, 
    #prefetch=1: chaque worker prend 1 tache à a fois =>evite qu'un worker monopolise toutes les taches qubit pendant qu'un autre est idle
    worker_prefetch_multiplier=1,
    task_acks_late=True, #ack post execution

    result_chord_join_timeout=180,

)

celery_app.autodiscover_tasks(["workers"])
celery_app.conf.imports = (
    "workers.qubit_tasks",
    "workers.sifting_tasks",
)