"""
This module in a FastAPI application uses Celery to handle asynchronous tasks:

1. `task_grid_opt`: Optimizes grid layouts for users and projects, with retry capabilities.
2. `task_supply_opt`: Optimizes energy supply systems for specific users and projects.
3. `task_remove_anonymous_users`: Deletes anonymous user accounts asynchronously.

Additionally, it includes functions to check the status of these tasks, identifying if they have completed, failed,
or been revoked. This setup enables efficient, asynchronous processing of complex tasks and user management.
"""



import os
import time
import traceback
import json
from copy import deepcopy
from celery import Celery
from celery.utils.log import get_task_logger

from supply_optimizer import optimize_energy_system
from grid_optimizer import optimize_grid


logger = get_task_logger(__name__)
CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

CELERY_TASK_NAME = os.environ.get("CELERY_TASK_NAME", "grid")

app = Celery(CELERY_TASK_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@app.task(name=f"supply.run_simulation")
def task_supply_opt(simulation_input: dict,) -> dict:
    logger.info("Start new simulation")
    try:
        simulation_output = optimize_energy_system(simulation_input)
        logger.info("Simulation finished")
        simulation_output["SERVER"] = CELERY_TASK_NAME
        if "message" in simulation_output:
            simulation_output["ERROR"] = simulation_output["message"]
            simulation_output["INPUT_JSON"] = simulation_input
        simulation_output = json.dumps(simulation_output)
    except Exception as e:
        logger.error(
            "An exception occured in the simulation task: {}".format(
                traceback.format_exc()
            )
        )
        simulation_output = json.dumps(dict(
            SERVER=CELERY_TASK_NAME,
            ERROR="{}".format(traceback.format_exc()),
            INPUT_JSON=simulation_input,
        ))
    return simulation_output


@app.task(name=f"grid.run_simulation")
def task_grid_opt(simulation_input: dict,) -> dict:
    logger.info("Start new simulation")
    try:
        simulation_output = optimize_grid(simulation_input)
        logger.info("Simulation finished")
        simulation_output["SERVER"] = CELERY_TASK_NAME
    except Exception as e:
        logger.error(
            "An exception occured in the simulation task: {}".format(
                traceback.format_exc()
            )
        )
        simulation_output = dict(
            SERVER=CELERY_TASK_NAME,
            ERROR="{}".format(traceback.format_exc()),
            INPUT_JSON=simulation_input,
        )
    return json.dumps(simulation_output)
