from celery import Celery

#here redis used as broker & result backend
REDIS_URL="redis://localhost:6379/0"

celery_app=Celery(
    "quantum_core",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["quantum_core.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=600,
    worker_prefetch_multiplier=1,
)
