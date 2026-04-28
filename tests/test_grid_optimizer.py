"""Unit tests for grid_optimizer.py.

These tests focus on deterministic behavior and avoid the full optimization flow,
which is better covered by integration tests because it depends on constrained
k-means clustering, UTM projection, and SciPy's MST implementation.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Keep these tests importable in lightweight unit-test environments where the
# optional/geospatial optimization dependencies are not installed. The tested
# functions below do not execute those dependencies.
def _install_import_stubs_if_needed() -> None:
    if importlib.util.find_spec("k_means_constrained") is None:
        module = types.ModuleType("k_means_constrained")

        class KMeansConstrained:  # pragma: no cover - should not be used in these unit tests
            def __init__(self, *args, **kwargs):
                raise RuntimeError("KMeansConstrained was called in a unit test")

        module.KMeansConstrained = KMeansConstrained
        sys.modules["k_means_constrained"] = module

    if importlib.util.find_spec("utm") is None:
        module = types.ModuleType("utm")
        module.from_latlon = lambda latitude, longitude: (0, 0, 32, "U")
        sys.modules["utm"] = module

    if importlib.util.find_spec("pyproj") is None:
        module = types.ModuleType("pyproj")

        class Proj:  # pragma: no cover - should not be used in these unit tests
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, x, y, inverse=False):
                return x, y

        module.Proj = Proj
        sys.modules["pyproj"] = module

    if importlib.util.find_spec("scipy") is None:
        scipy = types.ModuleType("scipy")
        sparse = types.ModuleType("scipy.sparse")
        csgraph = types.ModuleType("scipy.sparse.csgraph")
        csgraph.minimum_spanning_tree = lambda graph_matrix: graph_matrix
        sparse.csgraph = csgraph
        scipy.sparse = sparse
        sys.modules["scipy"] = scipy
        sys.modules["scipy.sparse"] = sparse
        sys.modules["scipy.sparse.csgraph"] = csgraph


_install_import_stubs_if_needed()

# Adjust this import if your module lives in a package, e.g.:
# from app.services.grid_optimizer import GridOptimizer, optimize_grid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_queue.grid_optimizer import GridOptimizer, optimize_grid  # noqa: E402


@pytest.fixture
def grid_design() -> dict:
    return {
        "pole": {"max_n_connections": 2, "epc": 100.0},
        "mg": {"epc": 50.0},
        "connection_cable": {"max_length": 40.0, "epc": 2.0},
        "distribution_cable": {"max_length": 100.0, "epc": 5.0},
        "shs": {"include": True, "max_grid_cost": 1_500.0},
    }


@pytest.fixture
def grid_opt_json(grid_design: dict) -> dict:
    return {
        "nodes": [
            {
                "latitude": 0.0,
                "longitude": 0.0,
                "x": 0.0,
                "y": 0.0,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "a",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
            {
                "latitude": 0.0,
                "longitude": 0.0001,
                "x": 3.0,
                "y": 4.0,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "b",
                "how_added": "manual",
                "shs_options": 2,
                "custom_specification": "",
            },
            {
                "latitude": 0.0,
                "longitude": 0.0002,
                "x": 10.0,
                "y": 0.0,
                "node_type": "power-house",
                "consumer_type": "n.a.",
                "consumer_detail": "n.a.",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
        ],
        "grid_design": grid_design,
        "yearly_demand": 1_200.0,
    }


@pytest.fixture
def optimizer(grid_opt_json: dict) -> GridOptimizer:
    opt = GridOptimizer(grid_opt_json)
    # Add columns that are normally introduced during optimization, so the
    # processing and cost tests can exercise those methods in isolation.
    defaults = {
        "cluster_label": 0,
        "type_fixed": False,
        "n_connection_links": 0,
        "n_distribution_links": 0,
        "parent": "unknown",
        "distribution_cost": 0.0,
        "cost_per_pole": 0.0,
        "branch": None,
        "parent_branch": None,
        "total_grid_cost_per_consumer_per_a": np.nan,
        "connection_cost_per_consumer": 0.0,
        "cost_per_branch": 0.0,
        "distribution_cost_per_branch": 0.0,
        "yearly_consumption": 0.0,
    }
    for column, value in defaults.items():
        if column not in opt.nodes.columns:
            opt.nodes[column] = value
    return opt


def test_optimize_grid_delegates_to_grid_optimizer(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_input = {"payload": "value"}
    sentinel_result = {"nodes": [], "links": []}

    class DummyGridOptimizer:
        def __init__(self, payload):
            self.payload = payload

        def optimize(self):
            assert self.payload is sentinel_input
            return sentinel_result

    import task_queue.grid_optimizer as module

    monkeypatch.setattr(module, "GridOptimizer", DummyGridOptimizer)
    assert optimize_grid(sentinel_input) is sentinel_result


def test_init_queries_nodes_and_sets_shs_threshold(grid_opt_json: dict) -> None:
    opt = GridOptimizer(grid_opt_json)

    assert opt.nodes.index.tolist() == ["0", "1", "2"]
    assert opt.nodes.loc["0", "is_connected"] == True
    assert opt.nodes.loc["1", "is_connected"] == False
    assert opt.power_house.index.tolist() == ["2"]
    assert opt.max_levelized_grid_cost == pytest.approx(1.5)
    assert opt.connection_cable_max_length == 40.0
    assert opt.distribution_cable_max_length == 100.0


def test_query_nodes_drops_non_manual_power_house(grid_opt_json: dict) -> None:
    grid_opt_json["nodes"][2]["how_added"] = "automatic"

    opt = GridOptimizer(grid_opt_json)

    assert "2" not in opt.nodes_df.index
    assert opt.power_house is None


def test_haversine_distance_for_known_one_degree_longitude_at_equator() -> None:
    distance = GridOptimizer.haversine_distance(0.0, 0.0, 0.0, 1.0)
    assert distance == pytest.approx(111_195, rel=1e-3)


def test_add_node_uses_defaults_and_overrides(optimizer: GridOptimizer) -> None:
    optimizer._add_node("p-0", node_type="pole", x=100.0, y=200.0)

    assert optimizer.nodes.loc["p-0", "node_type"] == "pole"
    assert optimizer.nodes.loc["p-0", "x"] == 100.0
    assert optimizer.nodes.loc["p-0", "y"] == 200.0
    assert optimizer.nodes.loc["p-0", "consumer_type"] == "household"
    assert optimizer.nodes.loc["p-0", "parent"] == "unknown"


def test_consumers_poles_and_grid_shs_filters(optimizer: GridOptimizer) -> None:
    optimizer._add_node("p-0", node_type="pole")

    assert optimizer.consumers().index.tolist() == ["0", "1"]
    assert optimizer.get_grid_consumers().index.tolist() == ["0"]
    assert optimizer.get_shs_consumers().index.tolist() == ["1"]
    assert optimizer._poles().index.tolist() == ["2", "p-0"]


def test_distance_between_nodes_returns_euclidean_distance(optimizer: GridOptimizer) -> None:
    assert optimizer.distance_between_nodes("0", "1") == pytest.approx(5.0)
    assert math.isinf(optimizer.distance_between_nodes("0", "missing"))


def test_add_links_creates_connection_link_and_sets_endpoint_metadata(
    optimizer: GridOptimizer,
) -> None:
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0, latitude=1.0, longitude=2.0)
    optimizer.nodes.loc["0", ["x", "y", "latitude", "longitude"]] = [3.0, 4.0, 5.0, 6.0]

    optimizer._add_links("p-0", "0")

    link = optimizer.links.loc["(p-0, 0)"]
    assert link["link_type"] == "connection"
    assert link["length"] == pytest.approx(5.0)
    assert link["from_node"] == "p-0"
    assert link["to_node"] == "0"
    assert link["lat_from"] == 1.0
    assert link["lon_to"] == 6.0


def test_add_links_treats_consumer_to_pole_as_connection_regression(
    optimizer: GridOptimizer,
) -> None:
    """A consumer-pole link should be a connection regardless of argument order."""
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0)

    optimizer._add_links("0", "p-0")

    assert optimizer.links.loc["(0, p-0)", "link_type"] == "connection"


def test_add_links_between_two_poles_creates_distribution_link_and_sorts_label(
    optimizer: GridOptimizer,
) -> None:
    optimizer._add_node("p-1", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=10.0)

    optimizer._add_links("p-1", "p-0")

    assert "(p-0, p-1)" in optimizer.links.index
    assert optimizer.links.loc["(p-0, p-1)", "link_type"] == "distribution"
    assert optimizer.links.loc["(p-0, p-1)", "length"] == pytest.approx(10.0)


def test_total_lengths_and_cost(optimizer: GridOptimizer) -> None:
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-1", node_type="pole", x=0.0, y=10.0)
    optimizer.nodes.loc["0", ["x", "y", "is_connected"]] = [3.0, 4.0, True]
    optimizer.nodes.loc["1", "is_connected"] = False

    optimizer._add_links("p-0", "0")  # 5 m connection
    optimizer._add_links("p-0", "p-1")  # 10 m distribution

    assert optimizer.total_length_connection_cable() == pytest.approx(5.0)
    assert optimizer.total_length_distribution_cable() == pytest.approx(10.0)
    # 3 poles/power-houses * 100 + 1 grid consumer * 50 + 5*2 + 10*5
    assert optimizer.cost() == pytest.approx(410.0)


def test_cost_is_infinite_without_links_or_without_poles(optimizer: GridOptimizer) -> None:
    assert math.isinf(optimizer.cost())

    optimizer._add_node("p-0", node_type="pole")
    optimizer._add_links("p-0", "0")
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] == "consumer"]

    assert math.isinf(optimizer.cost())


def test_clear_links_and_clear_all_links(optimizer: GridOptimizer) -> None:
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-1", node_type="pole", x=0.0, y=10.0)
    optimizer._add_links("p-0", "0")
    optimizer._add_links("p-0", "p-1")

    optimizer._clear_links("connection")
    assert set(optimizer.links["link_type"]) == {"distribution"}

    optimizer._clear_all_links()
    assert optimizer.links.empty


def test_find_index_longest_distribution_link(optimizer: GridOptimizer) -> None:
    optimizer.distribution_links = pd.DataFrame(
        {
            "length": [99.0, 100.0, 101.0],
        },
        index=["short", "equal", "long"],
    )

    assert optimizer.find_index_longest_distribution_link() == ["long"]


def test_add_number_of_distribution_and_connection_cables(optimizer: GridOptimizer) -> None:
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-1", node_type="pole", x=0.0, y=10.0)
    optimizer._add_links("p-0", "0")
    optimizer._add_links("p-0", "p-1")

    optimizer.add_number_of_distribution_and_connection_cables()

    assert optimizer.nodes.loc["p-0", "n_connection_links"] == 1
    assert optimizer.nodes.loc["p-0", "n_distribution_links"] == 1
    assert optimizer.nodes.loc["p-1", "n_distribution_links"] == 1
    assert optimizer.nodes.loc["0", "n_connection_links"] == 1


def test_process_nodes_formats_coordinates_and_converts_unknown_parent_to_none(
    optimizer: GridOptimizer,
) -> None:
    result = optimizer._process_nodes()

    assert result["label"] == ["0", "1", "2"]
    assert result["latitude"][0] == "0.000000"
    assert result["longitude"][1] == "0.000100"
    assert result["parent"][0] is None
    assert "x" not in result
    assert "cost_per_pole" not in result


def test_process_links_formats_coordinates_and_drops_internal_columns(
    optimizer: GridOptimizer,
) -> None:
    optimizer._add_node("p-0", node_type="pole", latitude=1.23456789, longitude=2.34567891)
    optimizer.nodes.loc["0", ["latitude", "longitude"]] = [3.45678912, 4.56789123]
    optimizer._add_links("p-0", "0")

    result = optimizer._process_links()

    assert result["label"] == ["(p-0, 0)"]
    assert result["lat_from"] == ["1.234568"]
    assert result["lon_to"] == ["4.567891"]
    assert "x_from" not in result
    assert "n_consumers" not in result


def test_change_direction_of_links_renames_index_when_needed() -> None:
    links = pd.DataFrame(index=["(p-1, p-0)"])

    result = GridOptimizer.change_direction_of_links("p-0", "p-1", links)

    assert result.index.tolist() == ["(p-0, p-1)"]


def test_cut_specific_pole_disconnects_consumers_and_removes_related_links(
    optimizer: GridOptimizer,
) -> None:
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0, parent="2", n_distribution_links=1)
    optimizer.nodes.loc["2", "n_distribution_links"] = 1
    optimizer.nodes.loc["0", ["parent", "branch", "is_connected"]] = ["p-0", "p-0", True]
    optimizer._add_links("p-0", "0")
    optimizer._add_links("2", "p-0")

    optimizer._cut_specific_pole("p-0")

    assert "p-0" not in optimizer.nodes.index
    assert optimizer.nodes.loc["0", "is_connected"] is False
    assert pd.isna(optimizer.nodes.loc["0", "parent"])
    assert optimizer.links.empty
    assert optimizer.nodes.loc["2", "n_distribution_links"] == 0


@pytest.mark.integration
def test_convert_lonlat_xy_round_trip_preserves_coordinates(optimizer: GridOptimizer) -> None:
    pytest.importorskip("utm")
    pytest.importorskip("pyproj")

    original = optimizer.nodes[["latitude", "longitude"]].astype(float).copy()

    optimizer.convert_lonlat_xy()

    assert optimizer.nodes["x"].notna().all()
    assert optimizer.nodes["y"].notna().all()
    assert not np.allclose(optimizer.nodes["x"].astype(float), original["longitude"].to_numpy())

    optimizer.convert_lonlat_xy(inverse=True)

    assert optimizer.nodes["latitude"].astype(float).to_numpy() == pytest.approx(
        original["latitude"].to_numpy(), abs=1e-6
    )
    assert optimizer.nodes["longitude"].astype(float).to_numpy() == pytest.approx(
        original["longitude"].to_numpy(), abs=1e-6
    )


@pytest.mark.integration
def test_create_minimum_spanning_tree_selects_shortest_pole_network(
    optimizer: GridOptimizer,
) -> None:
    pytest.importorskip("scipy")

    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "consumer"].copy()
    optimizer.nodes.loc["2", ["x", "y", "node_type"]] = [0.0, 0.0, "power-house"]
    optimizer._add_node("p-0", node_type="pole", consumer_type="n.a.", x=0.0, y=3.0)
    optimizer._add_node("p-1", node_type="pole", consumer_type="n.a.", x=4.0, y=0.0)

    optimizer.create_minimum_spanning_tree()
    optimizer.connect_grid_poles()

    assert len(optimizer.links) == 2
    assert set(optimizer.links["link_type"]) == {"distribution"}
    assert optimizer.total_length_distribution_cable() == pytest.approx(7.0)


@pytest.mark.integration
def test_set_direction_of_links_orients_distribution_links_toward_power_house(
    optimizer: GridOptimizer,
) -> None:
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "consumer"].copy()

    optimizer.nodes.loc[
        "2",
        ["x", "y", "node_type", "parent", "n_distribution_links"],
    ] = [0.0, 0.0, "power-house", "unknown", 1]

    optimizer._add_node(
        "p-0",
        node_type="pole",
        consumer_type="n.a.",
        consumer_detail="n.a.",
        x=10.0,
        y=0.0,
        parent="unknown",
        n_distribution_links=2,
    )
    optimizer._add_node(
        "p-1",
        node_type="pole",
        consumer_type="n.a.",
        consumer_detail="n.a.",
        x=20.0,
        y=0.0,
        parent="unknown",
        n_distribution_links=1,
    )

    # Physical/tree topology:
    # p-1 -> p-0 -> 2(power-house)
    optimizer._add_links("p-0", "2")
    optimizer._add_links("p-1", "p-0")

    optimizer._set_direction_of_links()

    assert "(p-0, 2)" in optimizer.links.index
    assert "(p-1, p-0)" in optimizer.links.index
    assert optimizer.nodes.loc["2", "parent"] == "2"
    assert optimizer.nodes.loc["p-0", "parent"] == "2"
    assert optimizer.nodes.loc["p-1", "parent"] == "p-0"

    assert optimizer.links.loc["(p-0, 2)", "from_node"] == "p-0"
    assert optimizer.links.loc["(p-0, 2)", "to_node"] == "2"
    assert optimizer.links.loc["(p-1, p-0)", "from_node"] == "p-1"
    assert optimizer.links.loc["(p-1, p-0)", "to_node"] == "p-0"


@pytest.mark.integration
def test_kmeans_clustering_adds_poles_and_assigns_consumers_to_clusters(
    grid_design: dict,
) -> None:
    pytest.importorskip("k_means_constrained")
    pytest.importorskip("utm")
    pytest.importorskip("pyproj")

    payload = {
        "nodes": [
            {
                "latitude": 52.5200,
                "longitude": 13.4050,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "a",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
            {
                "latitude": 52.5201,
                "longitude": 13.4051,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "b",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
            {
                "latitude": 52.5210,
                "longitude": 13.4060,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "c",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
            {
                "latitude": 52.5211,
                "longitude": 13.4061,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "d",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            },
        ],
        "grid_design": grid_design,
        "yearly_demand": 1_200.0,
    }
    opt = GridOptimizer(payload)
    opt.convert_lonlat_xy()

    opt.kmeans_clustering(n_clusters=2)

    poles = opt._poles()
    consumers = opt.consumers()
    assert len(poles) == 2
    assert set(poles["node_type"]) == {"pole"}
    assert consumers["cluster_label"].notna().all()
    assert poles[["latitude", "longitude", "x", "y"]].notna().all().all()


@pytest.mark.integration
def test_connect_grid_consumers_links_each_consumer_to_cluster_pole(
    optimizer: GridOptimizer,
) -> None:
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "power-house"].copy()
    optimizer.nodes.loc["0", ["cluster_label", "is_connected", "x", "y"]] = [0, True, 0.0, 1.0]
    optimizer.nodes.loc["1", ["cluster_label", "is_connected", "x", "y"]] = [0, True, 0.0, 2.0]
    optimizer._add_node("p-0", node_type="pole", cluster_label=0, type_fixed=False, x=0.0, y=0.0)

    optimizer.connect_grid_consumers()

    assert set(optimizer.links.index) == {"(p-0, 0)", "(p-0, 1)"}
    assert set(optimizer.links["link_type"]) == {"connection"}
    assert optimizer.nodes.loc["0", "parent"] == "p-0"
    assert optimizer.nodes.loc["1", "parent"] == "p-0"
