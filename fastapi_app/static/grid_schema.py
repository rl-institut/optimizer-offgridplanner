grid_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["node_fields", "nodes", "grid_design", "yearly_demand"],
    "properties": {
        "node_fields": {"type": "array", "items": {"type": "string"}},
        "nodes": {
            "type": "array",
            "items": {
                "type": "array",
                "items": [
                    {"type": "integer"},  # id
                    {"type": "string", "enum": ["automatic", "k-means"]},  # how_added
                    {
                        "type": "string",
                        "enum": ["consumer", "power-house"],
                    },  # node_type
                    {
                        "type": "string",
                        "enum": ["enterprise", "household", "n.a.", "public_service"],
                    },  # consumer_type
                    {"type": "string"},  # custom_specification
                    {"type": "integer", "enum": [0]},  # shs_options
                    {
                        "type": "string",
                        "enum": [
                            "Education_School",
                            "Food_Bar",
                            "Food_Drinks",
                            "Health_CHPS",
                            "Retail_Other",
                            "Trades_Beauty or Hair",
                            "Trades_Car or Motorbike Repair",
                            "default",
                            "n.a.",
                        ],
                    },  # consumer_detail
                    {"type": "boolean"},  # is_connected
                    {
                        "type": "array",  # coordinates
                        "items": [
                            {"type": "number"},  # latitude
                            {"type": "number"},  # longitude
                        ],
                        "minItems": 2,
                        "maxItems": 2,
                    },
                ],
                "minItems": 9,
                "maxItems": 9,
            },
        },
        "grid_design": {
            "type": "object",
            "properties": {
                "distribution_cable": {
                    "type": "object",
                    "required": ["lifetime", "capex", "max_length", "epc"],
                    "properties": {
                        "lifetime": {"type": "integer"},
                        "capex": {"type": "number"},
                        "max_length": {"type": "number"},
                        "epc": {"type": "number"},
                    },
                },
                "connection_cable": {
                    "type": "object",
                    "required": ["lifetime", "capex", "max_length", "epc"],
                    "properties": {
                        "lifetime": {"type": "integer"},
                        "capex": {"type": "number"},
                        "max_length": {"type": "number"},
                        "epc": {"type": "number"},
                    },
                },
                "pole": {
                    "type": "object",
                    "required": ["lifetime", "capex", "max_n_connections", "epc"],
                    "properties": {
                        "lifetime": {"type": "integer"},
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
            "required": ["distribution_cable", "connection_cable", "pole", "mg", "shs"],
        },
        "yearly_demand": {"type": "number"},
    },
}