"""Tests for supply_optimizer.py.

These tests intentionally keep most coverage at the constructor / configuration level,
because EnergySystemOptimizer.optimize() delegates to oemof/pyomo and an external CBC
solver. The solver-backed workflow test is marked as integration and skipped when CBC
is not installed.
"""

from __future__ import annotations

import copy
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest

from task_queue.supply_optimizer import EnergySystemOptimizer, optimize_energy_system


@pytest.fixture
def energy_system_design() -> dict[str, Any]:
    return {
        "pv": {
            "settings": {"is_selected": True, "design": False},
            "parameters": {"nominal_capacity": 8.0, "epc": 900.0},
        },
        "diesel_genset": {
            "settings": {"is_selected": True, "design": False, "offset": True},
            "parameters": {
                "nominal_capacity": 10.0,
                "epc": 500.0,
                "variable_cost": 0.02,
                "fuel_cost": 1.2,
                "fuel_lhv": 11.9,
                "max_efficiency": 0.33,
                "min_load": 0.3,
            },
        },
        "battery": {
            "settings": {"is_selected": True, "design": False},
            "parameters": {
                "nominal_capacity": 20.0,
                "epc": 300.0,
                "soc_min": 0.1,
                "soc_max": 1.0,
                "efficiency": 0.95,
                "c_rate_in": 0.5,
                "c_rate_out": 0.5,
            },
        },
        "inverter": {
            "settings": {"is_selected": True, "design": False},
            "parameters": {
                "nominal_capacity": 10.0,
                "epc": 120.0,
                "efficiency": 0.95,
            },
        },
        "rectifier": {
            "settings": {"is_selected": True, "design": False},
            "parameters": {
                "nominal_capacity": 10.0,
                "epc": 120.0,
                "efficiency": 0.93,
            },
        },
        "shortage": {
            "settings": {"is_selected": True},
            "parameters": {
                "shortage_penalty_cost": 10_000.0,
                "max_shortage_total": 0.2,
                "max_shortage_timestep": 0.5,
            },
        },
    }


@pytest.fixture
def supply_opt_json(energy_system_design: dict[str, Any]) -> dict[str, Any]:
    # Four hourly periods keeps construction and optional solver-backed tests small.
    return {
        "energy_system_design": copy.deepcopy(energy_system_design),
        "sequences": {
            "index": {
                "start_date": "2024-01-01T00:00:00",
                "freq": "h",
                "n_days": 1,
            },
            "demand": [3.0] * 24,
            "solar_potential": [0.0] * 6 + [0.2, 0.5, 0.8, 1.0, 0.7, 0.4] + [0.0] * 12,
        },
    }


def test_constructor_reconstructs_datetime_index_and_sequences(
    supply_opt_json: dict[str, Any],
) -> None:
    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert len(optimizer.dt_index) == 24
    assert optimizer.dt_index[0] == pd.Timestamp("2024-01-01T00:00:00")
    assert optimizer.dt_index[-1] == pd.Timestamp("2024-01-01T23:00:00")

    assert optimizer.demand.index.equals(optimizer.dt_index)
    assert optimizer.solar_potential.index.equals(optimizer.dt_index)
    assert optimizer.demand.tolist() == supply_opt_json["sequences"]["demand"]
    assert optimizer.solar_potential.tolist() == supply_opt_json["sequences"]["solar_potential"]
    assert optimizer.demand_peak == 3.0
    assert optimizer.solar_potential_peak == 1.0


def test_constructor_selects_inverter_when_pv_is_selected(
    supply_opt_json: dict[str, Any],
) -> None:
    supply_opt_json = copy.deepcopy(supply_opt_json)
    supply_opt_json["energy_system_design"]["pv"]["settings"]["is_selected"] = True
    supply_opt_json["energy_system_design"]["battery"]["settings"]["is_selected"] = False
    supply_opt_json["energy_system_design"]["inverter"]["settings"]["is_selected"] = False

    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.inverter["settings"]["is_selected"] is True


def test_constructor_selects_inverter_when_battery_is_selected(
    supply_opt_json: dict[str, Any],
) -> None:
    supply_opt_json = copy.deepcopy(supply_opt_json)
    supply_opt_json["energy_system_design"]["pv"]["settings"]["is_selected"] = False
    supply_opt_json["energy_system_design"]["battery"]["settings"]["is_selected"] = True
    supply_opt_json["energy_system_design"]["inverter"]["settings"]["is_selected"] = False

    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.inverter["settings"]["is_selected"] is True


def test_constructor_forces_renewable_components_when_diesel_is_not_selected(
    supply_opt_json: dict[str, Any],
) -> None:
    supply_opt_json = copy.deepcopy(supply_opt_json)
    design = supply_opt_json["energy_system_design"]
    design["diesel_genset"]["settings"]["is_selected"] = False
    design["pv"]["settings"]["is_selected"] = False
    design["battery"]["settings"]["is_selected"] = False
    design["inverter"]["settings"]["is_selected"] = False

    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.pv["settings"]["is_selected"] is True
    assert optimizer.battery["settings"]["is_selected"] is True
    assert optimizer.inverter["settings"]["is_selected"] is True


def test_constructor_disables_diesel_offset_for_cbc_solver(
    supply_opt_json: dict[str, Any],
) -> None:
    supply_opt_json = copy.deepcopy(supply_opt_json)
    supply_opt_json["energy_system_design"]["diesel_genset"]["settings"]["offset"] = True

    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.solver == "cbc"
    assert optimizer.diesel_genset["settings"]["offset"] is False


def test_constructor_detects_h2_system(supply_opt_json: dict[str, Any]) -> None:
    supply_opt_json = copy.deepcopy(supply_opt_json)
    design = supply_opt_json["energy_system_design"]
    design["fuel_cell"] = {
        "settings": {"is_selected": True, "design": False},
        "parameters": {"nominal_capacity": 1.0, "epc": 100.0, "efficiency": 0.5},
    }
    design["electrolyzer"] = {
        "settings": {"is_selected": True, "design": False},
        "parameters": {"nominal_capacity": 1.0, "epc": 100.0, "efficiency": 0.7},
    }
    design["h2_storage"] = {
        "settings": {"is_selected": True, "design": False},
        "parameters": {
            "nominal_capacity": 1.0,
            "epc": 100.0,
            "soc_min": 0.0,
            "soc_max": 1.0,
            "efficiency": 0.99,
            "c_rate_in": 0.1,
            "c_rate_out": 0.1,
        },
    }

    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.H2_system is True
    assert optimizer.fuel_cell is design["fuel_cell"]
    assert optimizer.electrolyzer is design["electrolyzer"]
    assert optimizer.h2_storage is design["h2_storage"]


def test_constructor_sets_initial_state(supply_opt_json: dict[str, Any]) -> None:
    optimizer = EnergySystemOptimizer(supply_opt_json)

    assert optimizer.infeasible is False
    assert optimizer.fuel_density_diesel == pytest.approx(0.846)
    assert optimizer.energy_system_design is supply_opt_json["energy_system_design"]


def test_optimize_energy_system_serializes_oemof_result_shape(
    monkeypatch: pytest.MonkeyPatch,
    supply_opt_json: dict[str, Any],
) -> None:
    class FakeOptimizer:
        def __init__(self, supply_opt_json: dict[str, Any]) -> None:
            self.supply_opt_json = supply_opt_json

        def optimize(self) -> dict[tuple[str, str], dict[str, Any]]:
            return {
                ("pv", "electricity_dc"): {
                    "scalars": pd.Series({"invest": 1.5, "capacity": 2.0}),
                    "sequences": pd.DataFrame({"flow": [0.0, 1.0, np.nan]}),
                },
            }

    monkeypatch.setattr("task_queue.supply_optimizer.EnergySystemOptimizer", FakeOptimizer)

    result = optimize_energy_system(supply_opt_json)

    assert list(result) == ["pv__electricity_dc"]
    assert isinstance(result["pv__electricity_dc"]["scalars"], str)
    assert result["pv__electricity_dc"]["sequences"] == [0.0, 1.0]


@pytest.mark.integration
def test_optimizer_runs_end_to_end_with_cbc_solver(supply_opt_json: dict[str, Any]) -> None:
    if shutil.which("cbc") is None:
        pytest.skip("CBC solver is not installed in this environment")

    optimizer = EnergySystemOptimizer(supply_opt_json)
    result = optimizer.optimize()

    assert isinstance(result, dict)
    assert result
    assert "message" not in result

    for value in result.values():
        assert "scalars" in value
        assert "sequences" in value
