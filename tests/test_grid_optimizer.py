"""Unit tests for grid_optimizer.py.

These tests focus on deterministic behavior and avoid the full optimization flow,
which is better covered by integration tests because it depends on constrained
k-means clustering, UTM projection, and SciPy's MST implementation.
"""

from __future__ import annotations

import copy
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
@pytest.fixture
def make_optimizer_with_single_node(grid_opt_json):
    def _make(latitude: float, longitude: float) -> GridOptimizer:
        single_node_grid_opt_json = copy.deepcopy(grid_opt_json)
        single_node_grid_opt_json["nodes"] = [
            {
                "latitude": latitude,
                "longitude": longitude,
                "x": 0.0,
                "y": 0.0,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": "test",
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            }
        ]
        return GridOptimizer(single_node_grid_opt_json)

    return _make


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


def test_optimize_grid_delegates_to_grid_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_distance_between_nodes_returns_euclidean_distance(
    optimizer: GridOptimizer,
) -> None:
    assert optimizer.distance_between_nodes("0", "1") == pytest.approx(5.0)
    assert math.isinf(optimizer.distance_between_nodes("0", "missing"))


def test_add_links_creates_connection_link_and_sets_endpoint_metadata(
    optimizer: GridOptimizer,
) -> None:
    optimizer._add_node(
        "p-0", node_type="pole", x=0.0, y=0.0, latitude=1.0, longitude=2.0
    )
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


def test_cost_is_infinite_without_links_or_without_poles(
    optimizer: GridOptimizer,
) -> None:
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
    # Considering that max_length_distribution_link is 100m
    optimizer._add_node("p-0", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-1", node_type="pole", x=0.0, y=100.0)
    optimizer._add_node("p-2", node_type="pole", x=0.0, y=200.0)
    optimizer._add_node("p-3", node_type="pole", x=0.0, y=350.0)
    optimizer._add_links("p-0", "p-1")
    optimizer._add_links("p-1", "p-2")
    optimizer._add_links("p-2", "p-3")

    assert optimizer.find_index_longest_distribution_link() == ['(p-2, p-3)']


def test_add_number_of_distribution_and_connection_cables(
    optimizer: GridOptimizer,
) -> None:
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
    optimizer._add_node(
        "p-0", node_type="pole", latitude=1.23456789, longitude=2.34567891,
        x=0.0, y=5.0,
    )
    optimizer.nodes.loc["0", ["latitude", "longitude", "x", "y"]] = [
        3.45678912, 4.56789123, 0.0, 0.0,
    ]
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
    optimizer._add_node(
        "p-0", node_type="pole", x=0.0, y=0.0, parent="2", n_distribution_links=1
    )
    optimizer.nodes.loc["2", "n_distribution_links"] = 1
    optimizer.nodes.loc["0", ["parent", "branch", "is_connected"]] = [
        "p-0",
        "p-0",
        True,
    ]
    optimizer._add_links("p-0", "0")
    optimizer._add_links("2", "p-0")

    optimizer._cut_specific_pole("p-0")

    assert "p-0" not in optimizer.nodes.index
    assert optimizer.nodes.loc["0", "is_connected"] is False
    assert pd.isna(optimizer.nodes.loc["0", "parent"])
    assert optimizer.links.empty
    assert optimizer.nodes.loc["2", "n_distribution_links"] == 0


@pytest.mark.integration
@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (52.5200, 13.4050),   # Germany: Berlin
        (9.0765, 7.3986),     # Nigeria: Abuja
        (13.7563, 100.5018),  # Thailand: Bangkok
    ],
)
def test_convert_lonlat_xy_round_trip_different_locations(
    make_optimizer_with_single_node,
    latitude: float,
    longitude: float,
) -> None:
    optimizer = make_optimizer_with_single_node(
        latitude=latitude,
        longitude=longitude,
    )

    optimizer.convert_lonlat_xy()
    assert optimizer.nodes.loc["0", "x"] != 0
    assert optimizer.nodes.loc["0", "y"] != 0

    optimizer.convert_lonlat_xy(inverse=True)

    assert float(optimizer.nodes.loc["0", "latitude"]) == pytest.approx(
        latitude,
        abs=1e-5,
    )
    assert float(optimizer.nodes.loc["0", "longitude"]) == pytest.approx(
        longitude,
        abs=1e-5,
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
    optimizer.nodes = optimizer.nodes[
        optimizer.nodes["node_type"] != "power-house"
    ].copy()
    optimizer.nodes.loc["0", ["cluster_label", "is_connected", "x", "y"]] = [
        0,
        True,
        0.0,
        1.0,
    ]
    optimizer.nodes.loc["1", ["cluster_label", "is_connected", "x", "y"]] = [
        0,
        True,
        0.0,
        2.0,
    ]
    optimizer._add_node(
        "p-0", node_type="pole", cluster_label=0, type_fixed=False, x=0.0, y=0.0
    )

    optimizer.connect_grid_consumers()

    assert set(optimizer.links.index) == {"(p-0, 0)", "(p-0, 1)"}
    assert set(optimizer.links["link_type"]) == {"connection"}
    assert optimizer.nodes.loc["0", "parent"] == "p-0"
    assert optimizer.nodes.loc["1", "parent"] == "p-0"


def _has_link(opt: GridOptimizer, a: str, b: str) -> bool:
    """Check a link between poles a and b exists in either direction."""
    return f"({a}, {b})" in opt.links.index or f"({b}, {a})" in opt.links.index


@pytest.mark.parametrize("to_from", [False, True])
@pytest.mark.parametrize("n_intermediate", [1, 2, 3])
def test_break_long_link_creates_complete_chain(
    optimizer: GridOptimizer, n_intermediate: int, to_from: bool
) -> None:
    """Regression: _break_long_link must form an unbroken chain for any N.
    """
    optimizer._add_node("p-from", node_type="pole", x=0.0, y=0.0)
    optimizer._add_node("p-to", node_type="pole", x=100.0, y=0.0)

    inter_ids = [f"p-mid-{i}" for i in range(n_intermediate)]
    for i, idx in enumerate(inter_ids):
        x = (i + 1) * 100.0 / (n_intermediate + 1)
        optimizer._add_node(idx, node_type="pole", type_fixed=True, x=x, y=0.0)

    added_poles_df = optimizer._poles().loc[inter_ids]
    added_poles = (added_poles_df, to_from)

    optimizer._break_long_link("p-from", "p-to", added_poles)

    assert len(optimizer.links) == n_intermediate + 1, (
        f"Expected {n_intermediate + 1} links, got {len(optimizer.links)}: "
        f"{list(optimizer.links.index)}"
    )

    # to_from=True: poles added from mst_to direction, so chain runs in reverse.
    ordered_inter = list(reversed(inter_ids)) if to_from else inter_ids
    chain = ["p-from"] + ordered_inter + ["p-to"]
    for a, b in zip(chain, chain[1:]):
        assert _has_link(optimizer, a, b), (
            f"Missing link between {a} and {b}. Links present: {list(optimizer.links.index)}"
        )

    for idx in inter_ids:
        assert optimizer.nodes.loc[idx, "how_added"] == "long-distance"


# ---------------------------------------------------------------------------
# Full-pipeline integration test
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_grid_payload() -> dict:
    """4-consumer grid, hand-calculable layout.

    At lat=1.0, lon=10.0 (UTM zone 32):
      - 0.00009 deg ≈ 10 m
      - 0.00270 deg ≈ 300 m

    Layout (approx, power house at origin):

        PH(0,0)   C0(10m E)   C1(10m N)   ...295m gap...   C2(300m E)   C3(300m E, 10m N)

    Near cluster (C0, C1):  centroid ≈ 7m NE of PH
    Far  cluster (C2, C3):  centroid ≈ 300m E of PH

    distribution_cable.max_length=100m  →  the ~295m near-to-far pole span forces
    intermediate poles (ceil(295/100)-1 = 2 poles inserted).
    SHS threshold set astronomically high so all consumers stay grid-connected.
    max_n_connections=3, so 2 consumers/pole is within limit.
    """
    return {
        "nodes": {
            "latitude":  [1.0,       1.000090, 1.0,       1.000090, 1.0],
            "longitude": [10.000090, 10.0,     10.002700, 10.002700, 10.0],
            "node_type": ["consumer", "consumer", "consumer", "consumer", "power-house"],
            "consumer_type":   ["household", "household", "household", "household", "n.a."],
            "consumer_detail": ["default",   "default",   "default",   "default",   "n.a."],
            "how_added":       ["manual",    "manual",    "manual",    "manual",    "manual"],
            "is_connected":    [True,        True,        True,        True,        True],
            "shs_options":     [0,           0,           0,           0,           0],
            "custom_specification": ["", "", "", "", ""],
        },
        "grid_design": {
            "distribution_cable": {"max_length": 100.0, "epc": 5.0},
            "connection_cable":   {"max_length": 30.0,  "epc": 2.0},
            "pole":               {"max_n_connections": 3, "epc": 100.0},
            "mg":                 {"epc": 50.0},
            "shs":                {"include": True, "max_grid_cost": 1_000_000.0},
        },
        "yearly_demand": 1_200.0,
    }



@pytest.mark.integration
def test_optimize_full_pipeline_simple_grid(simple_grid_payload: dict) -> None:
    """End-to-end smoke test: verifies the full optimize() chain on a minimal,
    hand-calculable grid.

    What this catches:
    - Long link breaking: intermediate poles inserted, all distribution links ≤ max_length
    - Clustering: all 4 consumers assigned to a pole (parent set)
    - Connectivity: every pole reachable from power house via distribution links
    - Pole connection limit: n_connection_links ≤ max_n_connections for every pole
    - No consumer left behind: all 4 are grid-connected (SHS threshold is very high)
    """
    pytest.importorskip("scipy")
    pytest.importorskip("utm")
    pytest.importorskip("k_means_constrained")
    pytest.importorskip("pyproj")

    dist_max = 100.0
    conn_max = 30.0
    max_n_conn = 3

    grid_opt = GridOptimizer(simple_grid_payload)
    result = grid_opt.optimize()

    nodes_out = result["nodes"]
    links_out = result["links"]

    # --- All 4 input consumers present, connected, and parented ---
    consumer_positions = [
        i for i, t in enumerate(nodes_out["node_type"]) if t == "consumer"
    ]
    assert len(consumer_positions) == 4

    for i in consumer_positions:
        assert nodes_out["is_connected"][i] is True, (
            f"Consumer at index {i} should be grid-connected"
        )
        assert nodes_out["parent"][i] is not None, (
            f"Consumer at index {i} should have a parent pole"
        )

    # --- No link exceeds its cable max length ---
    for label, ltype, length in zip(
        links_out["label"], links_out["link_type"], links_out["length"]
    ):
        if ltype == "distribution":
            assert length <= dist_max, (
                f"Distribution link {label} length {length:.1f}m > max {dist_max}m"
            )
        elif ltype == "connection":
            assert length <= conn_max, (
                f"Connection link {label} length {length:.1f}m > max {conn_max}m"
            )

    # --- At least one intermediate (long-distance) pole was inserted ---
    long_distance_poles = grid_opt.nodes[
        grid_opt.nodes["how_added"] == "long-distance"
    ]
    assert len(long_distance_poles) >= 1, (
        "No intermediate pole found; long link breaking did not fire"
    )

    # --- Pole connection-link count within limit ---
    for pole_idx, row in grid_opt.nodes[
        grid_opt.nodes["node_type"] == "pole"
    ].iterrows():
        assert row["n_connection_links"] <= max_n_conn, (
            f"Pole {pole_idx} has {row['n_connection_links']} connection links "
            f"(max {max_n_conn})"
        )

    # --- Full network connectivity: all poles reachable from power house ---
    dist_links = grid_opt.links[grid_opt.links["link_type"] == "distribution"]
    power_house_idx = grid_opt.nodes[
        grid_opt.nodes["node_type"] == "power-house"
    ].index[0]
    all_poles = grid_opt.nodes[
        grid_opt.nodes["node_type"].isin(["pole", "power-house"])
    ].index

    reachable: set = {power_house_idx}
    queue = [power_house_idx]
    while queue:
        current = queue.pop()
        neighbors = set(
            dist_links[dist_links["from_node"] == current]["to_node"].tolist()
            + dist_links[dist_links["to_node"] == current]["from_node"].tolist()
        )
        for neighbor in neighbors - reachable:
            reachable.add(neighbor)
            queue.append(neighbor)

    unreachable = [p for p in all_poles if p not in reachable]
    assert not unreachable, (
        f"Poles not reachable from power house: {unreachable}"
    )




@pytest.fixture
def shs_grid_payload() -> dict:
    """3-consumer grid designed so that the isolated far consumer becomes SHS.

    Layout (approx, power house at origin):

        PH(0,0)   C0(40m E)   C1(40m N)   ...460m gap...   C2(500m E)

    Near consumers (C0, C1): placed 40 m from PH — beyond the 30 m
    connection-cable auto-attach threshold in _connect_power_house_consumer_manually
    — so they go through k-means and get a proper cluster pole.
    The near-cluster pole's marginal cost is ~0.4 (well below 1.0) so it is
    never cut.

    max_n_connections=3 gives 3 placeholder nodes at PH.  The binary search in
    _find_opt_number_of_poles converges to 3 clusters (PH, near, far), which
    cleanly separates C0/C1 from C2.  With only 2 clusters the k_means_constrained
    capacity check (size_max x n_clusters >= n_samples) would fail.

    Far consumer (C2): single consumer, ~500 m from PH.

    Cost estimate for C2 (hand-calculated):
      yearly_consumption  = 1200 / 3  = 400 Wh/year
      distribution chain  = 5 poles x (epc_pole + ~96m x epc_dist)
                          = 5 x (100 + 96x5) = 5 x 580 = 2900 currency/year
      connection cost C2  = mg.epc = 50
      marginal_cost       = (2900 + 50) / 400 = 7.4 currency/Wh
      max_levelized_cost  = max_grid_cost / 1000  = 1000/1000 = 1.0

    7.4 >> 1.0  ->  C2 pole is cut, C2 becomes SHS.
    Intermediate long-distance poles then cascade-removed by
    _cut_leaf_poles_without_connection (no consumers left on that branch).
    """
    return {
        "nodes": {
            # 40 m E / N of PH so _connect_power_house_consumer_manually
            # (threshold = connection_cable.max_length = 30 m) does NOT
            # grab C0/C1 before k-means runs.
            "latitude":  [1.0,       1.000360, 1.0,       1.0],
            "longitude": [10.000359, 10.0,     10.004493, 10.0],
            "node_type": ["consumer", "consumer", "consumer", "power-house"],
            "consumer_type":   ["household", "household", "household", "n.a."],
            "consumer_detail": ["default",   "default",   "default",   "n.a."],
            "how_added":       ["manual",    "manual",    "manual",    "manual"],
            "is_connected":    [True,        True,        True,        True],
            "shs_options":     [0,           0,           0,           0],
            "custom_specification": ["", "", "", ""],
        },
        "grid_design": {
            "distribution_cable": {"max_length": 100.0, "epc": 5.0},
            "connection_cable":   {"max_length": 30.0,  "epc": 2.0},
            # 3 connections/pole -> binary search finds 3 clusters (PH + near + far)
            "pole":               {"max_n_connections": 3, "epc": 100.0},
            "mg":                 {"epc": 50.0},
            "shs":                {"include": True, "max_grid_cost": 1_000.0},
        },
        "yearly_demand": 1_200.0,
    }


@pytest.mark.integration
def test_optimize_full_pipeline_shs_consumer(shs_grid_payload: dict) -> None:
    """End-to-end: optimizer assigns isolated far consumer to SHS.

    Consumer "2" (C2, 500m east) is too expensive to connect relative to the
    SHS threshold — the optimizer should cut its pole and mark it SHS.
    Consumers "0" and "1" (near cluster, ~40m from PH) must stay grid-connected.

    Also verifies the intermediate long-distance poles are cascade-removed by
    _cut_leaf_poles_without_connection after the far cluster pole is cut —
    the remaining grid contains only the near cluster.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("utm")
    pytest.importorskip("k_means_constrained")
    pytest.importorskip("pyproj")

    grid_opt = GridOptimizer(shs_grid_payload)
    result = grid_opt.optimize()

    nodes_out = result["nodes"]

    label_to_idx = {lbl: i for i, lbl in enumerate(nodes_out["label"])}

    # --- Near consumers remain grid-connected with a parent ---
    for consumer_label in ("0", "1"):
        i = label_to_idx[consumer_label]
        assert nodes_out["is_connected"][i] is True, (
            f"Near consumer {consumer_label} should be grid-connected"
        )
        assert nodes_out["parent"][i] is not None, (
            f"Near consumer {consumer_label} should have a parent pole"
        )

    # --- Far isolated consumer assigned to SHS ---
    i = label_to_idx["2"]
    assert nodes_out["is_connected"][i] is False, (
        "Consumer '2' (500m isolated) should be SHS (is_connected=False)"
    )
    assert nodes_out["parent"][i] is None, (
        "Consumer '2' (SHS) should have no parent"
    )

    # --- No orphaned poles: every remaining pole reachable from power house ---
    dist_links = grid_opt.links[grid_opt.links["link_type"] == "distribution"]
    power_house_idx = grid_opt.nodes[
        grid_opt.nodes["node_type"] == "power-house"
    ].index[0]
    all_poles = grid_opt.nodes[
        grid_opt.nodes["node_type"].isin(["pole", "power-house"])
    ].index

    reachable: set = {power_house_idx}
    queue = [power_house_idx]
    while queue:
        current = queue.pop()
        neighbors = set(
            dist_links[dist_links["from_node"] == current]["to_node"].tolist()
            + dist_links[dist_links["to_node"] == current]["from_node"].tolist()
        )
        for neighbor in neighbors - reachable:
            reachable.add(neighbor)
            queue.append(neighbor)

    unreachable = [p for p in all_poles if p not in reachable]
    assert not unreachable, (
        f"Poles not reachable from power house after SHS pruning: {unreachable}"
    )


# ---------------------------------------------------------------------------
# Road graph construction and road-following intermediate poles
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_build_road_graph_creates_correct_weighted_adjacency(
    optimizer: GridOptimizer,
) -> None:
    """_build_road_graph produces a symmetric sparse adjacency matrix with correct weights.

    Hand-calculable triangle (all lengths exact integers or 3-4-5 right triangles):

        V0 = (0, 0)   V1 = (3, 4)   V2 = (6, 0)

        V0 → V1 : sqrt(3²+4²) = 5 m
        V1 → V2 : sqrt(3²+4²) = 5 m
        V0 → V2 : 6 m

    Expected 3×3 symmetric graph (row/col indexed by vertex):

        V0  [  0   5   6 ]
        V1  [  5   0   5 ]
        V2  [  6   5   0 ]
    """
    pytest.importorskip("scipy")

    optimizer.roads = pd.DataFrame({
        "x0": [0.0, 3.0, 0.0],
        "y0": [0.0, 4.0, 0.0],
        "x1": [3.0, 6.0, 6.0],
        "y1": [4.0, 0.0, 0.0],
    })
    optimizer._build_road_graph()

    assert optimizer._road_graph.shape == (3, 3), "Expected 3 unique vertices"
    assert len(optimizer._road_vertices) == 3

    verts = optimizer._road_vertices

    def vidx(x, y):
        for i, (vx, vy) in enumerate(verts):
            if abs(vx - x) < 0.01 and abs(vy - y) < 0.01:
                return i
        raise AssertionError(f"Vertex ({x}, {y}) not found in {list(verts)}")

    i0 = vidx(0, 0)
    i1 = vidx(3, 4)
    i2 = vidx(6, 0)
    assert len({i0, i1, i2}) == 3, "Vertices not distinct"

    dense = optimizer._road_graph.toarray()

    assert dense[i0, i1] == pytest.approx(5.0), "V0-V1 edge weight wrong"
    assert dense[i1, i0] == pytest.approx(5.0), "Graph not symmetric at V0-V1"
    assert dense[i1, i2] == pytest.approx(5.0), "V1-V2 edge weight wrong"
    assert dense[i2, i1] == pytest.approx(5.0), "Graph not symmetric at V1-V2"
    assert dense[i0, i2] == pytest.approx(6.0), "V0-V2 edge weight wrong"
    assert dense[i2, i0] == pytest.approx(6.0), "Graph not symmetric at V0-V2"

    assert dense[i0, i0] == 0.0
    assert dense[i1, i1] == 0.0
    assert dense[i2, i2] == 0.0

    # 3 undirected edges → 6 non-zero entries
    assert optimizer._road_graph.nnz == 6


@pytest.mark.integration
def test_build_road_graph_deduplicates_shared_endpoints(
    optimizer: GridOptimizer,
) -> None:
    """Shared segment endpoints produce a single vertex, not duplicates.

    Two collinear segments: (0,0)→(100,0) and (100,0)→(200,0).
    The shared endpoint (100,0) must appear exactly once → 3 vertices total,
    not 4.  The graph must be a simple path V0-V1-V2 (2 edges).
    """
    pytest.importorskip("scipy")

    optimizer.roads = pd.DataFrame({
        "x0": [0.0,   100.0],
        "y0": [0.0,   0.0],
        "x1": [100.0, 200.0],
        "y1": [0.0,   0.0],
    })
    optimizer._build_road_graph()

    assert optimizer._road_graph.shape == (3, 3), "Shared endpoint must be deduplicated"
    assert optimizer._road_graph.nnz == 4   # 2 edges × 2 directions


@pytest.mark.integration
def test_road_following_places_intermediate_poles_along_road(
    optimizer: GridOptimizer,
) -> None:
    """Intermediate poles on long links follow road geometry, not a straight line.

    Layout (UTM metres):

        A(0,0) -------- B(300,0)   straight line (y=0)

        A(0,0)          B(300,0)
           |             |
           |             |
        (0,150) ---- (300,150)   road runs along y=150

    distribution_cable_max_length = 100 m.
    Straight-line A→B = 300 m → 2 intermediate poles, both at y = 0.
    Road path A→B = 600 m → 5 intermediate poles, all at y > 0.
    """
    pytest.importorskip("scipy")

    optimizer.roads = pd.DataFrame({
        "x0": [0.0,   0.0,   300.0],
        "y0": [0.0,   150.0, 150.0],
        "x1": [0.0,   300.0, 300.0],
        "y1": [150.0, 150.0, 0.0],
    })
    optimizer._build_road_graph()

    waypoints = optimizer._road_path_between(0.0, 0.0, 300.0, 0.0)
    assert waypoints is not None, "No road path found — U-shaped road should connect (0,0) to (300,0)"

    raw = GridOptimizer._sample_points_along_polyline(
        waypoints, optimizer.distribution_cable_max_length
    )
    # Drop positions coinciding with endpoints (same filter as production code).
    poles = [
        (px, py) for px, py in raw
        if math.sqrt((px - 0.0) ** 2 + (py - 0.0) ** 2) > 1.0
        and math.sqrt((px - 300.0) ** 2 + (py - 0.0) ** 2) > 1.0
    ]

    assert len(poles) >= 3, (
        f"Expected at least 3 intermediate poles along 600 m road path, got {len(poles)}: {poles}"
    )

    for px, py in poles:
        assert py > 1.0, (
            f"Pole at ({px:.1f}, {py:.1f}) lies on the straight line (y≈0) — "
            f"road following did not activate. All road poles should have y > 0."
        )

    # Consecutive poles (plus endpoints) must not exceed max_length.
    chain = [(0.0, 0.0)] + poles + [(300.0, 0.0)]
    max_len = optimizer.distribution_cable_max_length
    for (ax, ay), (bx, by) in zip(chain, chain[1:]):
        dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        assert dist <= max_len + 1.0, (
            f"Gap between ({ax:.0f},{ay:.0f}) and ({bx:.0f},{by:.0f}) = {dist:.1f} m "
            f"> max {max_len} m"
        )


# ---------------------------------------------------------------------------
# _binary_search_n
# ---------------------------------------------------------------------------

def test_binary_search_n_finds_minimum_satisfying_n(optimizer: GridOptimizer) -> None:
    result = optimizer._binary_search_n(list(range(1, 11)), lambda n: n >= 4)
    assert result == 4


def test_binary_search_n_returns_last_element_when_nothing_satisfies(
    optimizer: GridOptimizer,
) -> None:
    # probe always False — last element is the guaranteed fallback
    result = optimizer._binary_search_n([1, 2, 3], lambda n: False)
    assert result == 3


def test_binary_search_n_single_candidate(optimizer: GridOptimizer) -> None:
    result = optimizer._binary_search_n([7], lambda n: n >= 7)
    assert result == 7


def test_binary_search_n_all_satisfy_returns_first(optimizer: GridOptimizer) -> None:
    result = optimizer._binary_search_n(list(range(1, 20)), lambda n: True)
    assert result == 1


# ---------------------------------------------------------------------------
# _sample_road_poles
# ---------------------------------------------------------------------------

def test_sample_road_poles_places_poles_at_interval(optimizer: GridOptimizer) -> None:
    """250 m road split into two segments, max_length=100 m → 4 poles at 0, 100, 200, 250."""
    optimizer.roads = pd.DataFrame({
        "x0": [0.0, 125.0],
        "y0": [0.0, 0.0],
        "x1": [125.0, 250.0],
        "y1": [0.0, 0.0],
        "road_id": ["r-0", "r-1"],
        "parent_road_id": ["road-1", "road-1"],
    })

    count = optimizer._sample_road_poles()

    road_poles = optimizer.nodes[optimizer.nodes.index.str.startswith("rp-")]
    assert count == 4
    assert len(road_poles) == 4
    xs = sorted(road_poles["x"].tolist())
    assert xs == pytest.approx([0.0, 100.0, 200.0, 250.0], abs=1.0)


def test_sample_road_poles_deduplicates_shared_endpoint(
    optimizer: GridOptimizer,
) -> None:
    """Two separate polylines share endpoint (100, 0) — must appear exactly once."""
    optimizer.roads = pd.DataFrame({
        "x0": [0.0, 100.0],
        "y0": [0.0, 0.0],
        "x1": [100.0, 200.0],
        "y1": [0.0, 0.0],
        "road_id": ["r-0", "r-1"],
        "parent_road_id": ["road-1", "road-2"],
    })

    optimizer._sample_road_poles()

    road_poles = optimizer.nodes[optimizer.nodes.index.str.startswith("rp-")]
    xs_at_100 = [x for x in road_poles["x"].tolist() if abs(x - 100.0) < 0.1]
    assert len(xs_at_100) == 1, "Shared endpoint (100, 0) must appear exactly once"


def test_sample_road_poles_adds_required_columns(optimizer: GridOptimizer) -> None:
    """Road poles must carry all columns expected by downstream methods."""
    optimizer.roads = pd.DataFrame({
        "x0": [0.0], "y0": [0.0], "x1": [50.0], "y1": [0.0],
        "road_id": ["r-0"], "parent_road_id": ["road-1"],
    })

    optimizer._sample_road_poles()

    road_poles = optimizer.nodes[optimizer.nodes.index.str.startswith("rp-")]
    for col in ("n_connection_links", "n_distribution_links", "parent",
                "custom_specification", "shs_options"):
        assert col in road_poles.columns, f"Missing column: {col}"
    assert (road_poles["n_connection_links"] == 0).all()
    assert (road_poles["shs_options"] == 0).all()
    assert (road_poles["node_type"] == "pole").all()


# ---------------------------------------------------------------------------
# _associate_consumers_to_road_poles
# ---------------------------------------------------------------------------

def test_associate_consumers_assigns_nearby_returns_far_unassigned(
    optimizer: GridOptimizer,
) -> None:
    """Consumer within connection_cable_max_length is assigned; one beyond is not."""
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "power-house"].copy()
    optimizer.nodes.loc["0", ["x", "y", "is_connected", "node_type"]] = [0.0, 0.0, True, "consumer"]
    optimizer.nodes.loc["1", ["x", "y", "is_connected", "node_type"]] = [500.0, 0.0, True, "consumer"]
    optimizer._add_node(
        "rp-0", node_type="pole", x=10.0, y=0.0,
        how_added="road-sampled", cluster_label=100000, is_connected=True,
    )

    unassigned = optimizer._associate_consumers_to_road_poles()

    assert "0" not in unassigned, "Consumer within range must be assigned"
    assert "1" in unassigned, "Consumer beyond max_length must be unassigned"
    assert optimizer.nodes.at["0", "cluster_label"] == 100000


def test_associate_consumers_drops_empty_road_pole(optimizer: GridOptimizer) -> None:
    """Road pole that attracts no consumers must be removed from self.nodes."""
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "power-house"].copy()
    optimizer.nodes.loc["0", ["x", "y", "is_connected", "node_type"]] = [0.0, 0.0, True, "consumer"]
    optimizer.nodes.loc["1", ["x", "y", "is_connected", "node_type"]] = [0.0, 1.0, True, "consumer"]
    # rp-0 close to consumers; rp-1 far beyond connection_cable_max_length (40 m)
    optimizer._add_node("rp-0", node_type="pole", x=0.0, y=0.0,
                        how_added="road-sampled", cluster_label=100000, is_connected=True)
    optimizer._add_node("rp-1", node_type="pole", x=999.0, y=0.0,
                        how_added="road-sampled", cluster_label=100001, is_connected=True)

    optimizer._associate_consumers_to_road_poles()

    assert "rp-0" in optimizer.nodes.index, "Used road pole must be kept"
    assert "rp-1" not in optimizer.nodes.index, "Empty road pole must be dropped"


# ---------------------------------------------------------------------------
# _build_branch_hierarchy
# ---------------------------------------------------------------------------

def test_allocate_branches_assigns_branch_and_propagates_to_consumers(
    optimizer: GridOptimizer,
) -> None:
    """Linear chain: c-0/c-1 → p-0/p-1 → power-house.

    Both poles must end up in branch p-0; both consumers must inherit that branch.
    """
    optimizer.nodes = optimizer.nodes[optimizer.nodes["node_type"] != "consumer"].copy()
    optimizer.nodes.loc["2", ["x", "y", "node_type", "parent", "n_distribution_links"]] = [
        0.0, 0.0, "power-house", "unknown", 1,
    ]
    optimizer._add_node("p-0", node_type="pole", consumer_type="n.a.", consumer_detail="n.a.",
                        x=10.0, y=0.0, parent="unknown", n_distribution_links=2)
    optimizer._add_node("p-1", node_type="pole", consumer_type="n.a.", consumer_detail="n.a.",
                        x=20.0, y=0.0, parent="unknown", n_distribution_links=1)
    optimizer._add_node("c-0", node_type="consumer", x=15.0, y=0.0, parent="p-0")
    optimizer._add_node("c-1", node_type="consumer", x=25.0, y=0.0, parent="p-1")

    optimizer._add_links("p-0", "2")
    optimizer._add_links("p-1", "p-0")
    optimizer._set_direction_of_links()
    optimizer.distribution_links = optimizer.links[optimizer.links["link_type"] == "distribution"]

    optimizer.allocate_poles_to_branches()
    optimizer.allocate_subbranches_to_branches()
    optimizer.label_branch_of_consumers()

    assert optimizer.nodes.at["p-0", "branch"] == "p-0"
    assert optimizer.nodes.at["p-1", "branch"] == "p-0"
    assert optimizer.nodes.at["c-0", "branch"] == "p-0"
    assert optimizer.nodes.at["c-1", "branch"] == "p-0"


# ---------------------------------------------------------------------------
# _find_opt_kmeans_for_unassigned
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_find_opt_kmeans_for_unassigned_satisfies_cable_length(
    grid_design: dict,
) -> None:
    """Binary search must find n such that all connection cables ≤ max_length.

    4 consumers spaced ~50 m apart; connection_cable_max_length = 40 m.
    One cluster centred between two adjacent consumers is ~25 m from each,
    so n=2 should be sufficient.  The method must return n ≥ 1 and the
    final clustering must produce no violated cable.
    """
    pytest.importorskip("k_means_constrained")
    pytest.importorskip("utm")
    pytest.importorskip("pyproj")

    design = copy.deepcopy(grid_design)
    design["pole"]["max_n_connections"] = 2  # size_max must not exceed n_samples (4)
    payload = {
        "nodes": [
            {
                "latitude": 0.0,
                "longitude": float(i) * 0.00045,
                "node_type": "consumer",
                "consumer_type": "household",
                "consumer_detail": str(i),
                "how_added": "manual",
                "shs_options": 0,
                "custom_specification": "",
            }
            for i in range(4)
        ],
        "grid_design": design,
        "yearly_demand": 1_200.0,
    }
    opt = GridOptimizer(payload)
    opt.convert_lonlat_xy()

    n = opt._find_opt_kmeans_for_unassigned(opt.get_grid_consumers().index)

    assert n >= 1
    opt.kmeans_clustering(n_clusters=n, consumer_indices=opt.get_grid_consumers().index)
    opt.connect_grid_consumers()
    conn = opt.links[opt.links["link_type"] == "connection"]
    assert (conn["length"] <= opt.connection_cable_max_length).all(), (
        f"Cable length violated with n={n}: max={conn['length'].max():.1f} m "
        f"> limit={opt.connection_cable_max_length} m"
    )
