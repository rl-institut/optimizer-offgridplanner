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
from jsonschema import validate

from task_queue.supply_optimizer import optimize_energy_system
from task_queue.grid_optimizer import optimize_grid


logger = get_task_logger(__name__)
CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

CELERY_TASK_NAME = os.environ.get("CELERY_TASK_NAME", "grid")

app = Celery(CELERY_TASK_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# TODO decide where the schemas should be stored/if they should be accessible from the API
SUPPLY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "sequences": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date-time"},
                        "n_days": {"type": "integer", "minimum": 1},
                        "freq": {"type": "string", "enum": ["h"]},
                    },
                    "required": ["start_date", "n_days", "freq"],
                },
                "demand": {"type": "array", "items": {"type": "number"}},
                "solar_potential": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["index", "demand", "solar_potential"],
        },
        "energy_system_design": {
            "type": "object",
            "properties": {
                "battery": {"$ref": "#/definitions/component"},
                "diesel_genset": {"$ref": "#/definitions/component"},
                "inverter": {"$ref": "#/definitions/component"},
                "pv": {"$ref": "#/definitions/component"},
                "rectifier": {"$ref": "#/definitions/component"},
                "shortage": {
                    "type": "object",
                    "properties": {
                        "settings": {
                            "type": "object",
                            "properties": {"is_selected": {"type": "boolean"}},
                            "required": ["is_selected"],
                        },
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "max_shortage_total": {"type": "number"},
                                "max_shortage_timestep": {"type": "number"},
                                "shortage_penalty_cost": {"type": "number"},
                            },
                            "required": [
                                "max_shortage_total",
                                "max_shortage_timestep",
                                "shortage_penalty_cost",
                            ],
                        },
                    },
                    "required": ["settings", "parameters"],
                },
            },
            "required": [
                "battery",
                "diesel_genset",
                "inverter",
                "pv",
                "rectifier",
                "shortage",
            ],
        },
    },
    "required": ["sequences", "energy_system_design"],
    "definitions": {
        "component": {
            "type": "object",
            "properties": {
                "settings": {
                    "type": "object",
                    "properties": {
                        "is_selected": {"type": "boolean"},
                        "design": {"type": "boolean"},
                    },
                    "required": ["is_selected", "design"],
                },
                "parameters": {
                    "type": "object",
                    "properties": {
                        "nominal_capacity": {"type": ["number", "null"]},
                        "soc_min": {"type": "number"},
                        "soc_max": {"type": "number"},
                        "c_rate_in": {"type": "number"},
                        "c_rate_out": {"type": "number"},
                        "efficiency": {"type": "number"},
                        "epc": {"type": "number"},
                        "variable_cost": {"type": "number"},
                        "fuel_cost": {"type": "number"},
                        "fuel_lhv": {"type": "number"},
                        "min_load": {"type": "number"},
                        "max_load": {"type": "number"},
                        "min_efficiency": {"type": "number"},
                        "max_efficiency": {"type": "number"},
                    },
                    "additionalProperties": True,
                },
            },
            "required": ["settings", "parameters"],
        }
    },
}
GRID_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["nodes", "grid_design", "yearly_demand"],
    "properties": {
        "nodes": {
            "type": "object",
            "required": [
                "latitude",
                "longitude",
                "how_added",
                "node_type",
                "consumer_type",
                "custom_specification",
                "shs_options",
                "consumer_detail",
                "is_connected",
            ],
            "properties": {
                "latitude": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "longitude": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "how_added": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "node_type": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "consumer_type": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "custom_specification": {
                    "type": "array",
                    "items": {"type": ["string", "null"]},
                },
                "shs_options": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "consumer_detail": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "is_connected": {
                    "type": "array",
                    "items": {"type": "boolean"},
                },
                "distance_to_load_center": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "distribution_cost": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "parent": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
        "grid_design": {
            "type": "object",
            "required": ["distribution_cable", "connection_cable", "pole", "mg", "shs"],
            "properties": {
                "distribution_cable": {
                    "type": "object",
                    "required": ["capex", "max_length", "epc"],
                    "properties": {
                        "capex": {"type": "number"},
                        "max_length": {"type": "number"},
                        "epc": {"type": "number"},
                    },
                },
                "connection_cable": {
                    "type": "object",
                    "required": ["capex", "max_length", "epc"],
                    "properties": {
                        "capex": {"type": "number"},
                        "max_length": {"type": "number"},
                        "epc": {"type": "number"},
                    },
                },
                "pole": {
                    "type": "object",
                    "required": ["capex", "max_n_connections", "epc"],
                    "properties": {
                        "capex": {"type": "number"},
                        "max_n_connections": {"type": "integer"},
                        "epc": {"type": "number"},
                    },
                },
                "mg": {
                    "type": "object",
                    "required": ["connection_cost", "epc"],
                    "properties": {
                        "connection_cost": {"type": "number"},
                        "epc": {"type": "number"},
                    },
                },
                "shs": {
                    "type": "object",
                    "required": ["include", "max_grid_cost"],
                    "properties": {
                        "include": {"type": "boolean"},
                        "max_grid_cost": {"type": "number"},
                    },
                },
            },
        },
        "yearly_demand": {"type": "number"},
    },
    "additionalProperties": False,
}

@app.task(name=f"supply.run_simulation")
def task_supply_opt(simulation_input: dict,) -> dict:
    logger.info("Start new simulation")
    try:
        validate(instance=simulation_input, schema=SUPPLY_SCHEMA)
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
        validate(instance=simulation_input, schema=GRID_SCHEMA)
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
