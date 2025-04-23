import os
from celery import Celery
from kombu import Queue

CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

# this will be linked to task_queue/tasks.py
app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
app.conf.task_queues = (
    Queue("grid", routing_key="grid.#"),
    Queue("supply", routing_key="supply.#"),
)
