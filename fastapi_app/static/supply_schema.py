supply_schema = {
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