grid_schema_input = {
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
                    "items": {"type": ["integer", "null"]},
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
                    "items": {"type": ["number", "null"]},
                },
                "distribution_cost": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
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
            "required": ["distribution_cable", "connection_cable", "pole", "mg"],
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
        },
        "yearly_demand": {"type": "number"},
    },
}

grid_schema_output = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "nodes": {
      "type": "object",
        "required": [
            "latitude", "longitude", "how_added", "node_type", "consumer_type",
            "custom_specification", "shs_options", "consumer_detail", "is_connected",
            "distance_to_load_center", "parent", "distribution_cost"
        ],
        "properties": {
        "latitude": {
          "type": "array",
          "items": { "type": "string" }
        },
        "longitude": {
          "type": "array",
          "items": { "type": "string" }
        },
        "how_added": {
          "type": "array",
          "items": { "type": "string" }
        },
        "node_type": {
          "type": "array",
          "items": { "type": "string" }
        },
        "consumer_type": {
          "type": "array",
          "items": { "type": "string" }
        },
        "custom_specification": {
          "type": "array",
          "items": { "type": ["string", "null"] }
        },
        "shs_options": {
          "type": "array",
          "items": { "type": ["integer", "null"] }
        },
        "consumer_detail": {
          "type": "array",
          "items": { "type": "string" }
        },
        "is_connected": {
          "type": "array",
          "items": { "type": "boolean" }
        },
        "distance_to_load_center": {
          "type": "array",
          "items": { "type": ["number", "null"] }
        },
        "parent": {
          "type": "array",
          "items": { "type": "string" }
        },
        "distribution_cost": {
          "type": "array",
          "items": { "type": ["number", "null"] }
        }
      }
    },
    "links": {
      "type": "object",
      "properties": {
        "lat_from": {
          "type": "array",
          "items": { "type": "string" }
        },
        "lon_from": {
          "type": "array",
          "items": { "type": "string" }
        },
        "lat_to": {
          "type": "array",
          "items": { "type": "string" }
        },
        "lon_to": {
          "type": "array",
          "items": { "type": "string" }
        },
        "link_type": {
          "type": "array",
          "items": { "type": "string" }
        },
        "length": {
          "type": "array",
          "items": { "type": "number" }
        }
      },
      "required": [
        "lat_from", "lon_from", "lat_to", "lon_to", "link_type", "length"
      ]
    }
  },
  "required": ["nodes", "links"]
}