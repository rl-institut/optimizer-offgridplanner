"""
This module is an intricate part of a FastAPI application designed for grid optimization in energy projects, with a
focus on efficiently connecting consumers to a power house through a distribution grid. The grid comprises poles,
connection cables, and distribution cables. Here's an expanded overview highlighting the key aspects and
functionalities:

1. **Initial Data Retrieval:**
   - The module begins by querying a database to retrieve node data, which includes all consumers needing power supply.
     It also gathers project parameters like the Weighted Average Cost of Capital (WACC), crucial for financial
     calculations in the optimization process.

2. **Grid Optimization Objective:**
   - The primary goal is to connect these consumers to the power house in the most efficient manner. This involves
     determining the optimal placement of poles and routing of distribution and connection cables.

3. **Core Optimization Process:**
   - Utilizing the `GridOptimizer` class, it employs k-means clustering for determining the optimal locations for poles.
   - The optimizer connects consumers to the nearest poles and interlinks poles using a minimum spanning tree approach,
     ensuring an efficient energy distribution network.
   - The grid optimization takes into account the maximum permissible length for connection cables to avoid overly long
     connections that might be inefficient or impractical.


4. **Cost Calculations and Constraints Handling in Grid Optimization (Refined):**

   - **Initial Setup and User Specifications:**
     - The process begins with all potential grid consumers included. Users of the system have the option to manually
       specify certain consumers to be either definitely connected, excluded, or equipped with solar home systems (SHS).
       This user input is crucial in guiding the initial setup and subsequent optimization steps.

   - **Initial Cost Distribution Among Consumers:**
     - The initial phase involves distributing the cost of grid components (poles, cables) among all consumers. This
       allocation is based on each consumer's specific connection details, ensuring a fair and proportionate
       distribution of the grid's total cost.

   - **Exclusion of Consumers Based on Cost Threshold:**
     - The algorithm then evaluates each consumer, starting from the ends of the grid's branches, to check if their
       individual connection cost exceeds a user-defined maximum threshold.
     - Consumers whose connection costs are too high are considered for exclusion. This is a crucial step in
       maintaining the financial viability of the grid.

   - **Iterative Optimization Process:**
     - After each exclusion, the grid's cost allocation is recalculated. This iterative process is key to understanding
       the financial implications of each exclusion and adjusting the grid design accordingly.
     - If a consumer's cost is within the acceptable range, all consumers upstream in the same branch are automatically
       retained in the grid. This decision point helps streamline the optimization process by finalizing sections of
       the grid without further analysis.

   - **Determining the Direction of Grid Links:**
     - A critical component of this process is establishing the directionality of each grid link. Since the Kruskal
       algorithm, used for the minimum spanning tree (MST) calculation, does not provide link direction, the module
       includes a specific function for this purpose.
     - Determining the direction of flow for each link is essential for accurately assigning costs and for the logical
       distribution of power within the grid.

   - **Dynamic Grid Adjustment:**
     - Based on the ongoing optimization, the grid configuration is dynamically adjusted. Consumers may be excluded
       based on cost-effectiveness, and the layout of poles and cables is modified to reflect the most efficient
       design under the given constraints.
     - This dynamic adjustment ensures that the final grid layout is not only technically sound but also adheres to the
       financial and user-defined parameters, striking a balance between efficiency, cost-effectiveness,
       and user preferences.

5. **Final Output and Database Interaction:**
   - Post-optimization, the `nodes` object, now containing both consumers and poles, is written back to the database.
     This provides a comprehensive view of the grid layout and participant nodes.
   - The `links` object is also stored in the database. It details the start and end points of all cables, categorizes
     the type of cables (distribution or connection), and identifies the start and end nodes (consumers, poles,
     power-house).
   - The result is a database-driven representation of the optimized grid, providing a foundation for further analysis,
     implementation, or modification.

6. **Error Handling and Logging:**
   - Throughout the process, the module ensures robust error handling and logging, crucial for diagnosing issues and
     optimizing performance.

"""

import copy
import json
import logging
import math
import time

import numpy as np
import pandas as pd
from k_means_constrained import KMeansConstrained
from pyproj import Proj
from scipy.sparse.csgraph import minimum_spanning_tree

logger = logging.getLogger(__name__)


pd.options.mode.chained_assignment = None  # default='warn'


def optimize_grid(grid_opt_json):
    grid_opt = GridOptimizer(grid_opt_json)
    results = grid_opt.optimize()
    return results


class GridOptimizer:
    def __init__(self, grid_opt_json):
        print("Initiating grid optimizer...")
        # TODO go through the helper functions and figure out what they do / document
        self.start_execution_time = time.monotonic()
        self.grid_opt_json = grid_opt_json
        self.nodes = pd.DataFrame(self.grid_opt_json["nodes"])
        self.grid_design_dict = self.grid_opt_json["grid_design"]
        self.yearly_demand = self.grid_opt_json["yearly_demand"]
        self.nodes_df, self.power_house = self._query_nodes()
        self.links = pd.DataFrame(
            {
                "label": pd.Series([], dtype=str),
                "lat_from": pd.Series([], dtype=np.dtype(float)),
                "lon_from": pd.Series([], dtype=np.dtype(float)),
                "lat_to": pd.Series([], dtype=np.dtype(float)),
                "lon_to": pd.Series([], dtype=np.dtype(float)),
                "x_from": pd.Series([], dtype=np.dtype(float)),
                "y_from": pd.Series([], dtype=np.dtype(float)),
                "x_to": pd.Series([], dtype=np.dtype(float)),
                "y_to": pd.Series([], dtype=np.dtype(float)),
                "link_type": pd.Series([], dtype=str),
                "length": pd.Series([], dtype=int),
                "n_consumers": pd.Series([], dtype=int),
                "total_power": pd.Series([], dtype=int),
                "from_node": pd.Series([], dtype=str),
                "to_node": pd.Series([], dtype=str),
            },
        ).set_index("label")
        self.distribution_links = self.links[
            self.links["link_type"] == "distribution"
        ].copy()
        self.pole_max_connection = self.grid_design_dict["pole"]["max_n_connections"]
        self.grid_mst = pd.DataFrame({}, dtype=np.dtype(float))
        self.max_levelized_grid_cost = self.grid_design_dict["shs"]["max_grid_cost"]
        self.connection_cable_max_length = self.grid_design_dict["connection_cable"][
            "max_length"
        ]
        self.distribution_cable_max_length = self.grid_design_dict[
            "distribution_cable"
        ]["max_length"]

    def optimize(self):
        print("Optimizing distribution grid...")
        self.convert_lonlat_xy()
        self._clear_poles()
        n_total_consumers = len(self.nodes)
        n_shs_consumers = len(self.nodes[self.nodes["is_connected"] == False])  # noqa: E712
        n_grid_consumers = n_total_consumers - n_shs_consumers
        self.nodes = self.nodes.sort_index(key=lambda x: x.astype("int64"))
        if self.power_house is not None:
            power_house_consumers = self._connect_power_house_consumer_manually(
                self.connection_cable_max_length,
            )
            self._placeholder_consumers_for_power_house()
        else:
            power_house_consumers = None
        print("Determining number of poles...")
        n_poles = self._find_opt_number_of_poles(n_grid_consumers)
        self.determine_poles(
            min_n_clusters=n_poles,
            power_house_consumers=power_house_consumers,
        )
        # Find the connection links_df in the network with lengths greater than the
        # maximum allowed length for `connection` cables, specified by the user.
        long_links = self.find_index_longest_distribution_link()
        # Add poles to the identified long `distribution` links_df, so that the
        # distance between all poles remains below the maximum allowed distance.
        self._add_fixed_poles_on_long_links(long_links=long_links)
        # Update the (lon,lat) coordinates based on the newly inserted poles
        # which only have (x,y) coordinates.
        self.convert_lonlat_xy(inverse=True)
        # Connect all poles together using the minimum spanning tree algorithm.
        print("Determining distribution links...")
        self.connect_grid_poles(long_links=long_links)
        # Calculate distances of all poles from the load centroid.
        # Find the location of the power house.
        self.add_number_of_distribution_and_connection_cables()
        n_iter = 2 if self.power_house is None else 1
        print("Determining power house location...")
        for i in range(n_iter):
            if self.power_house is None and i == 0:
                self._select_location_of_power_house()
            self._set_direction_of_links()
            self.allocate_poles_to_branches()
            self.allocate_subbranches_to_branches()
            self.label_branch_of_consumers()
            self.determine_cost_per_pole()
            self._connection_cost_per_consumer()
            self.determine_costs_per_branch()
            consumer_idxs = self.nodes[self.nodes["node_type"] == "consumer"].index
            self.nodes.loc[consumer_idxs, "yearly_consumption"] = (
                self.yearly_demand / len(consumer_idxs)
            )
            self._determine_shs_consumers()
            if self.power_house is None and len(self.links) > 0:
                old_power_house = self.nodes[
                    self.nodes["node_type"] == "power-house"
                ].index[0]
                self._select_location_of_power_house()
                new_power_house = self.nodes[
                    self.nodes["node_type"] == "power-house"
                ].index[0]
                if old_power_house == new_power_house:
                    break
            else:
                break

        return self._process_results()

    def _process_results(self):
        """
        Returns a json object with processed nodes and links
        :return json: Links and nodes calculated in optimization
        """
        results = {
                "nodes": self._process_nodes(),
                "links": self._process_links(),
            }
        return results

    def _process_nodes(self):
        nodes_df = self.nodes.reset_index(drop=True)
        nodes_df = nodes_df.drop(
            labels=[
                "x",
                "y",
                "cluster_label",
                "type_fixed",
                "n_connection_links",
                "n_distribution_links",
                "cost_per_pole",
                "branch",
                "parent_branch",
                "total_grid_cost_per_consumer_per_a",
                "connection_cost_per_consumer",
                "cost_per_branch",
                "distribution_cost_per_branch",
                "yearly_consumption",
            ],
            axis=1,
        )
        nodes_df = nodes_df.round(decimals=6)
        nodes_df = nodes_df.replace(np.nan, None)
        if not nodes_df.empty:
            nodes_df.latitude = nodes_df.latitude.map(lambda x: f"{x:.6f}")
            nodes_df.longitude = nodes_df.longitude.map(lambda x: f"{x:.6f}")
            if len(nodes_df.index) != 0:
                if "parent" in nodes_df.columns:
                    nodes_df["parent"] = nodes_df["parent"].where(
                        nodes_df["parent"] != "unknown",
                        None,
                    )

        return nodes_df.reset_index(drop=True).to_dict(orient="list")

    def _process_links(self):
        links_df = self.links.reset_index(drop=True)
        links_df = links_df.drop(
            labels=[
                "x_from",
                "y_from",
                "x_to",
                "y_to",
                "n_consumers",
                "total_power",
                "from_node",
                "to_node",
            ],
            axis=1,
        )
        links_df.lat_from = links_df.lat_from.map(lambda x: f"{x:.6f}")
        links_df.lon_from = links_df.lon_from.map(lambda x: f"{x:.6f}")
        links_df.lat_to = links_df.lat_to.map(lambda x: f"{x:.6f}")
        links_df.lon_to = links_df.lon_to.map(lambda x: f"{x:.6f}")
        links_df = links_df.replace(np.nan, None)

        return links_df.reset_index(drop=True).to_dict(orient="list")

    def _query_nodes(self):
        """

        :return:
        """
        nodes_df = self.nodes
        nodes_df.loc[nodes_df["shs_options"] == 2, "is_connected"] = False  # noqa: PLR2004 -> TODO check what shs_options=2 means
        nodes_df["is_connected"] = True
        nodes_df.index = nodes_df.index.astype(str)
        nodes_df = nodes_df[nodes_df["node_type"].isin(["consumer", "power-house"])]
        power_houses = nodes_df.loc[nodes_df["node_type"] == "power-house"]
        # TODO what is happening here? is manual not the only way to add power houses?
        if len(power_houses) > 0 and power_houses["how_added"].iloc[0] != "manual":
            nodes_df = nodes_df.drop(index=power_houses.index)
            power_houses = None
        elif len(power_houses) == 0:
            power_houses = None
        return nodes_df, power_houses

    # -------------------- NODES -------------------- #
    def _get_load_centroid(self):
        """
        This function obtains the ideal location for the power house, which is
        at the load centroid of the village.
        """
        grid_consumers = self.nodes[self.nodes["is_connected"] == True]  # noqa:E712
        lat = np.average(grid_consumers["latitude"])
        lon = np.average(grid_consumers["longitude"])
        self.load_centroid = [lat, lon]

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        radius = 6371.0 * 1000

        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        # Haversine formula
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Distance
        distance = radius * c
        return distance

    def _get_poles_distances_from_load_centroid(self):
        """
        This function calculates all distances between the poles and the load
        centroid of the settlement.
        """
        self._get_load_centroid()
        lat2, lon2 = self.load_centroid
        for pole_index in self._poles().index:
            lat1 = self.nodes.latitude.loc[pole_index]
            lon1 = self.nodes.longitude.loc[pole_index]
            self.nodes.loc[pole_index, "distance_to_load_center"] = (
                GridOptimizer.haversine_distance(lat1, lon1, lat2, lon2)
            )

    def _select_location_of_power_house(self):
        """
        This function assumes the closest pole to the calculated location for
        the power house, as the new location of the power house.
        """
        self._get_poles_distances_from_load_centroid()
        poles_with_consumers = self._poles()
        poles_with_consumers = poles_with_consumers[
            poles_with_consumers["n_connection_links"] > 0
        ]
        min_distance_nearest_pole = poles_with_consumers[
            "distance_to_load_center"
        ].min()
        nearest_pole = self._poles()[
            self._poles()["distance_to_load_center"] == min_distance_nearest_pole
        ]
        self.nodes.loc[self.nodes["node_type"] == "power-house", "node_type"] = "pole"
        self.nodes.loc[nearest_pole.index, "node_type"] = "power-house"

    def _connect_power_house_consumer_manually(self, max_length):
        power_house = self.nodes.loc[self.nodes["node_type"] == "power-house"]
        self.convert_lonlat_xy()
        x2 = power_house["x"].to_numpy()[0]
        y2 = power_house["y"].to_numpy()[0]
        for consumer in self.nodes[self.nodes["node_type"] == "consumer"].index:
            x1 = self.nodes.x.loc[consumer]
            y1 = self.nodes.y.loc[consumer]
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            self.nodes.loc[consumer, "distance_to_load_center"] = distance
        consumers = self.nodes[
            self.nodes["distance_to_load_center"] <= max_length
        ].copy()
        if len(consumers) > 0:
            self.nodes = self.nodes.drop(index=consumers.index)
        return consumers

    def _placeholder_consumers_for_power_house(self, *, remove=False):
        n_max = self.pole_max_connection
        label_start = 100000
        power_house = self.nodes.loc[self.nodes["node_type"] == "power-house"]
        for i in range(n_max):
            label = str(label_start + i)
            if remove is True:
                self.nodes = self.nodes.drop(index=label)
            else:
                self._add_node(
                    label,
                    latitude=power_house["latitude"].to_numpy()[0],
                    longitude=power_house["longitude"].to_numpy()[0],
                    x=np.nan,
                    y=np.nan,
                    cluster_label=np.nan,
                    n_connection_links=np.nan,
                    n_distribution_links=np.nan,
                    parent=np.nan,
                )
        if remove is False:
            self.convert_lonlat_xy()

    def _clear_nodes(self):
        """
        Removes all nodes from the grid.
        """
        self.nodes = self.nodes.drop(list(self.nodes.index), axis=0)

    def _clear_poles(self):
        """
        Removes all poles from the grid.
        """
        self.nodes = self.nodes.drop(
            [
                label
                for label in self.nodes.index
                if self.nodes.node_type.loc[label] in ["pole"]
            ],
            axis=0,
        )

    def get_grid_consumers(self):
        df = self.nodes[
            (self.nodes["is_connected"] == True)  # noqa:E712
            & (self.nodes["node_type"] == "consumer")
        ]
        return df.copy()

    def get_shs_consumers(self):
        df = self.nodes[
            (self.nodes["is_connected"] == False)  # noqa:E712
            & (self.nodes["node_type"] == "consumer")
        ]
        return df.copy()

    def find_index_longest_distribution_link(self):
        # Find the links longer than two times of the allowed distance
        critical_link = self.distribution_links[
            self.distribution_links["length"] > self.distribution_cable_max_length
        ]

        return list(critical_link.index)

    def _add_fixed_poles_on_long_links(
        self,
        long_links,
    ):
        for long_link in long_links:
            # Get start and end coordinates of the long link.
            x_from = self.links.x_from[long_link]
            x_to = self.links.x_to[long_link]
            y_from = self.links.y_from[long_link]
            y_to = self.links.y_to[long_link]

            # Calculate the number of additional poles required.
            n_required_poles = int(
                np.ceil(
                    self.links.length[long_link] / self.distribution_cable_max_length,
                )
                - 1,
            )

            # Get the index of the last pole in the grid. The new pole's index
            # will start from this index.
            last_pole = self._poles().index[-1]
            # Split the pole's index using `-` as the separator, because poles
            # are labeled in `p-x` format. x represents the index number, which
            # must be an integer.
            index_last_pole = int(last_pole.split("-")[1])

            # Calculate the slope of the line, connecting the start and end
            # points of the long link.
            slope = (y_to - y_from) / (x_to - x_from)

            # Calculate the final length of the smaller links after splitting
            # the long links into smaller parts.
            length_smaller_links = self.links.length[long_link] / (n_required_poles + 1)

            # Add all poles between the start and end points of the long link.
            for i in range(1, n_required_poles + 1):
                x = x_from + np.sign(
                    x_to - x_from,
                ) * i * length_smaller_links * np.sqrt(1 / (1 + slope**2))
                y = y_from + np.sign(y_to - y_from) * i * length_smaller_links * abs(
                    slope,
                ) * np.sqrt(1 / (1 + slope**2))

                pole_label = f"p-{i + index_last_pole}"

                # In adding the pole, the `how_added` attribute is considered
                # `long-distance-init`, which means the pole is added because
                # of long distance in a distribution link.
                # The reason for using the `long_link` part is to distinguish
                # it with the poles which are already `connected` to the grid.
                # The poles in this stage are only placed on the line, and will
                # be connected to the other poles using another function.
                # The `cluster_label` is given as 1000, to avoid inclusion in
                # other clusters.
                self._add_node(
                    label=pole_label,
                    x=x,
                    y=y,
                    node_type="pole",
                    consumer_type="n.a.",
                    consumer_detail="n.a.",
                    is_connected=True,
                    how_added=long_link,
                    type_fixed=True,
                    cluster_label=1000,
                    custom_specification="",
                    shs_options=0,
                )

    def _add_node(self, label, **kwargs):
        """
        adds a node to the grid's node dataframe.

        Parameters
        ----------
        already defined in the 'Grid' object definition
        """
        default_values = {
            "latitude": 0,
            "longitude": 0,
            "x": 0,
            "y": 0,
            "node_type": "consumer",
            "consumer_type": "household",
            "consumer_detail": "default",
            "distance_to_load_center": 0,
            "is_connected": True,
            "how_added": "automatic",
            "type_fixed": False,
            "cluster_label": 0,
            "n_connection_links": "0",
            "n_distribution_links": 0,
            "parent": "unknown",
            "distribution_cost": 0,
            "custom_specification": "",
            "shs_options": 0,
        }

        node_data = {**default_values, **kwargs}

        for key, value in node_data.items():
            self.nodes.loc[label, key] = value

    def consumers(self):
        """
        Returns only the 'consumer' nodes from the grid.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            filtered pandas dataframe containing all 'consumer' nodes
        """
        return self.nodes[self.nodes["node_type"] == "consumer"]

    def _poles(self):
        """
        Returns all poles and the power house in the grid.

        Returns
        ------
        class:`pandas.core.frame.DataFrame`
            filtered pandas dataframe containing all 'pole' nodes
        """
        return self.nodes[
            (self.nodes["node_type"] == "pole")
            | (self.nodes["node_type"] == "power-house")
        ]

    def distance_between_nodes(self, label_node_1: str, label_node_2: str):
        """
        Returns the distance between two nodes of the grid

        Parameters
        ----------
        label_node_1: str
            label of the first node
        label_node_2: str
            label of the second node

        Return
        -------
            distance between the two nodes in meter
        """
        if (label_node_1 and label_node_2) in self.nodes.index:
            # (x,y) coordinates of the points
            x1 = self.nodes.x.loc[label_node_1]
            y1 = self.nodes.y.loc[label_node_1]

            x2 = self.nodes.x.loc[label_node_2]
            y2 = self.nodes.y.loc[label_node_2]

            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return np.inf

    # -------------------- LINKS -------------------- #

    def get_links(self):
        """
        Returns all links of the grid.

        Returns
        -------
        class:`pandas.core.frame.DataFrame`
            a pandas dataframe containing all links of the grid
        """
        return self.links

    def get_nodes(self):
        """
        Return all nodes of the grid.

        Returns
        -------
        class:`pandas.core.frame.DataFrame`
            a pandas dataframe containing all nodes of the grid
        """
        return self.nodes

    def _clear_all_links(self):
        """
        Removes all links from the grid.
        """
        self.links = self.get_links().drop(
            list(self.get_links().index),
            axis=0,
        )

    def _clear_links(self, link_type):
        """
        Removes all link types given by the user from the grid.
        """
        self.links = self.get_links().drop(
            list(self.get_links()[self.get_links()["link_type"] == link_type].index),
            axis=0,
        )

    def _add_links(self, label_node_from: str, label_node_to: str):
        """
        Adds a link between two nodes of the grid and
        calculates the distance of the link.

        Parameters
        ----------
        label_node_from: str
            label of the first node
        label_node_to: str
            label of the second node

        Notes
        -----
        The method first makes sure that the two labels belong to the grid.
        Otherwise, no link will be added.
        """

        # specify the type of the link which is obtained based on the start/end nodes of the link
        if (
            self.nodes.node_type.loc[label_node_from]
            and self.nodes.node_type.loc[label_node_to]
        ) == "pole":
            # convention: if two poles are getting connected, the beginning will be the one with lower number
            (label_node_from, label_node_to) = sorted([label_node_from, label_node_to])
            link_type = "distribution"
        else:
            link_type = "connection"

        # calculate the length of the link
        length = self.distance_between_nodes(
            label_node_1=label_node_from,
            label_node_2=label_node_to,
        )

        # define a label for the link and add all other characteristics to the grid object
        label = f"({label_node_from}, {label_node_to})"
        self.links.loc[label, "lat_from"] = self.nodes.latitude.loc[label_node_from]
        self.links.loc[label, "lon_from"] = self.nodes.longitude.loc[label_node_from]
        self.links.loc[label, "lat_to"] = self.nodes.latitude.loc[label_node_to]
        self.links.loc[label, "lon_to"] = self.nodes.longitude.loc[label_node_to]
        self.links.loc[label, "x_from"] = self.nodes.x.loc[label_node_from]
        self.links.loc[label, "y_from"] = self.nodes.y.loc[label_node_from]
        self.links.loc[label, "x_to"] = self.nodes.x.loc[label_node_to]
        self.links.loc[label, "y_to"] = self.nodes.y.loc[label_node_to]
        self.links.loc[label, "link_type"] = link_type
        self.links.loc[label, "length"] = length
        self.links.loc[label, "n_consumers"] = 0
        self.links.loc[label, "total_power"] = 0
        self.links.loc[label, "from_node"] = ""
        self.links.loc[label, "to_node"] = ""

    def total_length_distribution_cable(self):
        """
        Calculates the total length of all cables connecting only poles in the grid.

        Returns
        ------
        type: float
            the total length of the distribution cable in the grid
        """
        return self.links.length[self.links.link_type == "distribution"].sum()

    def total_length_connection_cable(self):
        """
        Calculates the total length of all cables between each pole and
        consumers.

        Returns
        ------
        type: float
            total length of the connection cable in the grid.
        """
        return self.links.length[self.links.link_type == "connection"].sum()

    # -------------------- OPERATIONS -------------------- #

    def convert_lonlat_xy(self, *, inverse=False):
        """
        +++ ok +++

        Converts (longitude, latitude) coordinates into (x, y)
        plane coordinates using a python package 'pyproj'.

        Parameter
        ---------
        inverse: bool (default=false)
            this parameter indicates the direction of conversion
            + false: lon,lat --> x,y
            + true: x,y --> lon/lat
        """

        p = Proj(proj="utm", zone=32, ellps="WGS84", preserve_units=False)

        # if inverse=true, this is the case when the (x,y) coordinates of the obtained
        # poles (from the optimization) are converted into (lon,lat)
        if inverse:
            # First the possible candidates for inverse conversion are picked.
            nodes_for_inverse_conversion = self.nodes[
                (self.nodes["node_type"] == "pole")
                | (self.nodes["node_type"] == "power-house")
            ]

            for node_index in nodes_for_inverse_conversion.index:
                lon, lat = p(
                    self.nodes.x.loc[node_index],
                    self.nodes.y.loc[node_index],
                    inverse=inverse,
                )
                self.nodes.loc[node_index, "longitude"] = lon
                self.nodes.loc[node_index, "latitude"] = lat
        else:
            for node_index in self.nodes.index:
                x, y = p(
                    self.nodes.longitude.loc[node_index],
                    self.nodes.latitude.loc[node_index],
                    inverse=inverse,
                )
                self.nodes.loc[node_index, "x"] = x
                self.nodes.loc[node_index, "y"] = y

            # store reference values for (x,y) to use later when converting (x,y) to (lon,lat)

    # -------------------- COSTS ------------------------ #

    def cost(self):
        """
        Computes the cost of the grid taking into account the number
        of nodes, their types (consumer or poles) and the length of
        different types of cables between nodes.

        Return
        ------
        cost of the grid
        """

        # get the number of poles, consumers and links from the grid
        n_poles = self._poles().shape[0]
        n_mg_consumers = self.consumers()[
            self.consumers()["is_connected"] == True  # noqa:E712
        ].shape[0]
        n_links = self.get_links().shape[0]

        # if there is no poles in the grid, or there is no link,
        # the function returns an infinite value
        if (n_poles == 0) or (n_links == 0):
            return np.inf

        # calculate the total length of the cable used between poles [m]
        total_length_distribution_cable = self.total_length_distribution_cable()

        # calculate the total length of the `connection` cable between poles and consumers
        total_length_connection_cable = self.total_length_connection_cable()

        grid_cost = (
            n_poles * self.grid_design_dict["pole"]["epc"]
            + n_mg_consumers * self.grid_design_dict["mg"]["epc"]
            + total_length_connection_cable
            * self.grid_design_dict["connection_cable"]["epc"]
            + total_length_distribution_cable
            * self.grid_design_dict["distribution_cable"]["epc"]
        )

        return np.around(grid_cost, decimals=2)

    def add_number_of_distribution_and_connection_cables(self):
        poles = self._poles().copy()
        links = self.get_links().copy()
        links["from_node"] = (
            links.index.str.extract(r"^\(\s*([^,]+)")[0].str.strip().tolist()
        )
        links["to_node"] = (
            links.index.str.extract(r",\s*([^,]+)\s*\)$")[0].str.strip().tolist()
        )
        connection_links = links[links["link_type"] == "connection"].copy()
        for pole_idx in poles.index:
            n_distribution = len(
                self.distribution_links[
                    self.distribution_links["from_node"] == pole_idx
                ].index,
            )
            n_distribution += len(
                self.distribution_links[
                    self.distribution_links["to_node"] == pole_idx
                ].index,
            )
            self.nodes.loc[pole_idx, "n_distribution_links"] = n_distribution
            n_connection = len(
                connection_links[connection_links["from_node"] == pole_idx].index,
            )
            n_connection += len(
                connection_links[connection_links["to_node"] == pole_idx].index,
            )
            self.nodes.loc[pole_idx, "n_connection_links"] = n_connection
        self.nodes.loc[self.nodes["node_type"] == "consumer", "n_connection_links"] = 1
        self.nodes["n_distribution_links"] = (
            self.nodes["n_distribution_links"].fillna(0).astype(int)
        )
        self.nodes["n_connection_links"] = (
            self.nodes["n_connection_links"].fillna(0).astype(int)
        )

    def allocate_poles_to_branches(self):
        poles = self._poles().copy()
        leaf_poles = pd.Series(
            poles[poles["n_distribution_links"] == 1].index
        ).to_numpy()
        split_poles = pd.Series(
            poles[poles["n_distribution_links"] > 2].index  # noqa: PLR2004
        ).to_numpy()
        power_house = poles[poles["node_type"] == "power-house"].index[0]
        start_poles = self.distribution_links[
            (self.distribution_links["to_node"] == power_house)
        ]["from_node"].to_numpy()
        start_set = set(start_poles)
        split_set = set(split_poles)
        diff_set = start_set - split_set
        start_poles = np.array(list(diff_set))
        for split_pole in split_poles:
            for start_pole in self.distribution_links[
                (self.distribution_links["to_node"] == split_pole)
            ]["from_node"].to_list():
                start_poles = np.append(start_poles, start_pole)
        start_poles = pd.Series(start_poles).drop_duplicates().to_numpy()
        self.nodes["branch"] = None
        tmp_idxs = self.nodes[self.nodes.index.isin(start_poles)].index
        self.nodes.loc[start_poles, "branch"] = pd.Series(tmp_idxs, index=tmp_idxs)

        for start_pole in start_poles:
            next_pole = copy.deepcopy(start_pole)
            for _ in range(len(poles.index)):
                next_pole = self.distribution_links[
                    (self.distribution_links["to_node"] == next_pole)
                ]["from_node"]
                if len(next_pole.index) == 1:
                    next_pole = next_pole.to_numpy()[0]
                    self.nodes.loc[next_pole, "branch"] = start_pole
                    if next_pole in split_poles or next_pole in leaf_poles:
                        break
                else:
                    break

        self.nodes.loc[
            (self.nodes["branch"].isna())
            & (self.nodes["node_type"].isin(["pole", "power-house"])),
            "branch",
        ] = power_house

    def label_branch_of_consumers(self):
        branch_map = self.nodes.loc[
            self.nodes.node_type.isin(["pole", "power-house"]),
            "branch",
        ]
        self.nodes.loc[self.nodes.node_type == "consumer", "branch"] = self.nodes.loc[
            self.nodes.node_type == "consumer",
            "parent",
        ].map(branch_map)

    def determine_parent_branches(self, start_poles):
        poles = self._poles().copy()
        for pole in start_poles:
            branch_start_pole = poles[poles.index == pole]["branch"].iloc[0]
            split_pole = self.nodes[self.nodes.index == branch_start_pole][
                "parent"
            ].iloc[0]
            parent_branch = poles[poles.index == split_pole]["branch"].iloc[0]
            self.nodes.loc[
                self.nodes["branch"] == branch_start_pole,
                "parent_branch",
            ] = parent_branch

    def allocate_subbranches_to_branches(self):
        poles = self._poles().copy()
        self.nodes["parent_branch"] = None
        power_house = poles[poles["node_type"] == "power-house"].index[0]

        if len(poles["branch"].unique()) > 1:
            leaf_poles = poles[poles["n_distribution_links"] == 1].index
            self.determine_parent_branches(leaf_poles)
            poles_expect_power_house = poles[poles["node_type"] != "power-house"]
            split_poles = poles_expect_power_house[
                poles_expect_power_house["n_distribution_links"] > 2  # noqa: PLR2004
            ].index
            if len(split_poles) > 0:
                self.determine_parent_branches(split_poles)

        self.nodes.loc[
            (self.nodes["parent_branch"].isna())
            & (self.nodes["node_type"].isin(["pole", "power-house"])),
            "parent_branch",
        ] = power_house

    def determine_cost_per_pole(self):
        poles = self._poles().copy()
        links = self.get_links().copy()
        self.nodes["cost_per_pole"] = None
        self.nodes["cost_per_pole"] = self.nodes["cost_per_pole"].astype(float)
        power_house = poles[poles["node_type"] == "power-house"].index[0]
        for pole in poles.index:
            if pole != power_house:
                parent_pole = poles[poles.index == pole]["parent"].iloc[0]
                try:
                    length = links[
                        (links["from_node"] == pole) & (links["to_node"] == parent_pole)
                    ]["length"].iloc[0]
                except IndexError:
                    try:
                        length = links[
                            (links["from_node"] == parent_pole)
                            & (links["to_node"] == pole)
                        ]["length"].iloc[0]
                    except IndexError:
                        length = 20
                self.nodes.loc[pole, "cost_per_pole"] = (
                    self.grid_design_dict["pole"]["epc"]
                    + length * self.grid_design_dict["distribution_cable"]["epc"]
                )
            else:
                self.nodes.loc[pole, "cost_per_pole"] = self.grid_design_dict["pole"][
                    "epc"
                ]

    def determine_costs_per_branch(self, branch=None):
        poles = self._poles().copy()

        def _(branch):
            branch_df = self.nodes[
                (self.nodes["branch"] == branch) & (self.nodes["is_connected"] == True)  # noqa:E712
            ].copy()
            cost_per_branch = self.nodes[self.nodes.index.isin(branch_df.index)][
                "cost_per_pole"
            ].sum()
            cost_per_branch += self.nodes[self.nodes.index.isin(branch_df.index)][
                "connection_cost_per_consumer"
            ].sum()
            self.nodes.loc[branch_df.index, "distribution_cost_per_branch"] = (
                cost_per_branch
            )
            self.nodes.loc[branch_df.index, "cost_per_branch"] = cost_per_branch

        if branch is None:
            for unique_branch in poles["branch"].unique():
                _(unique_branch)
        else:
            _(branch)

    def _connection_cost_per_consumer(self):
        links = self.get_links()
        grid_consumers = self.nodes[
            (self.nodes["node_type"] == "consumer")
            & (self.nodes["is_connected"] == True)  # noqa:E712
        ].index
        for consumer in grid_consumers:
            parent_pole = self.nodes[self.nodes.index == consumer]["parent"].iloc[0]
            length = min(
                links[
                    (links["from_node"] == consumer) & (links["to_node"] == parent_pole)
                ]["length"].iloc[0],
                3,
            )
            connection_cost = (
                self.grid_design_dict["mg"]["epc"]
                + length * self.grid_design_dict["connection_cable"]["epc"]
            )
            self.nodes.loc[consumer, "connection_cost_per_consumer"] = connection_cost

    def get_subbranches(self, branch):
        subbranches = self.nodes[self.nodes["branch"] == branch].index.tolist()
        leaf_branches = self.nodes[self.nodes["n_distribution_links"] == 1][
            "branch"
        ].index
        next_sub_branches = self.nodes[self.nodes["parent_branch"] == branch][
            "parent_branch"
        ].tolist()
        for _ in range(len(self.nodes["branch"].unique())):
            next_next_sub_branches = []
            for sub_branch in next_sub_branches:
                if sub_branch in leaf_branches:
                    break
                for b in next_sub_branches:
                    subbranches.append(b)
                next_next_sub_branches.append(sub_branch)
            next_sub_branches = next_next_sub_branches
            if len(next_sub_branches) == 0:
                break
        return subbranches

    def get_all_consumers_of_subbranches(self, branch):
        branches = self.get_subbranches(branch)
        consumers = self.nodes[
            (self.nodes["node_type"] == "consumer")
            & (self.nodes["branch"].isin(branches))
            & (self.nodes["is_connected"] == True)  # noqa:E712
        ].index
        return consumers

    def get_all_consumers_of_branch(self, branch):
        consumers = self.nodes[
            (self.nodes["node_type"] == "consumer")
            & (self.nodes["branch"].isin(branch))
            & (self.nodes["is_connected"] == True)  # noqa:E712
        ].index
        return consumers

    def _determine_distribution_links(self):
        idxs = self.links[self.links.index.to_series().str.count("p") == 2].index  # noqa: PLR2004
        self.links.loc[idxs, "link_type"] = "distribution"
        self.distribution_links = self.links.loc[idxs]
        return idxs

    def _distribute_cost_among_consumers(self):
        self.nodes["total_grid_cost_per_consumer_per_a"] = np.nan
        self.nodes.loc[
            self.nodes[self.nodes["is_connected"] == True].index,  # noqa:E712
            "total_grid_cost_per_consumer_per_a",
        ] = self.nodes["connection_cost_per_consumer"]
        leaf_branches = self.nodes[self.nodes["n_distribution_links"] == 1][
            "branch"
        ].unique()
        for branch in self.nodes["branch"].unique():
            if branch is not None:
                if branch in leaf_branches:
                    poles_of_branch = self.nodes[self.nodes["branch"] == branch]
                    next_pole = poles_of_branch[
                        poles_of_branch["n_distribution_links"] == 1
                    ]
                    for _ in range(len(poles_of_branch)):
                        consumers_of_pole = poles_of_branch[
                            (poles_of_branch["node_type"] == "consumer")
                            & (poles_of_branch["is_connected"] == True)  # noqa:E712
                            & (poles_of_branch["parent"] == next_pole.index[0])
                        ]
                        consumers_down_the_line = list(consumers_of_pole.index)
                        total_consumption = self.nodes[
                            self.nodes.index.isin(consumers_down_the_line)
                        ]["yearly_consumption"].sum()
                        cost_of_pole = self.nodes.loc[
                            self.nodes[self.nodes.index == next_pole.index[0]].index,
                            "cost_per_pole",
                        ].iloc[0]
                        for consumer in consumers_down_the_line:
                            self.nodes.loc[
                                consumer,
                                "total_grid_cost_per_consumer_per_a",
                            ] += (
                                cost_of_pole
                                * self.nodes.loc[consumer, "yearly_consumption"]
                                / total_consumption
                            )
                        next_pole = self.nodes[
                            self.nodes.index == next_pole["parent"].iloc[0]
                        ]
                        if len(next_pole) == 0 or (
                            self.nodes[self.nodes.index == next_pole.index[0]][
                                "branch"
                            ].iloc[0]
                            != branch
                        ):
                            break
                else:
                    continue
        self.nodes["total_grid_cost_per_consumer_per_a"] = (
            self.nodes["total_grid_cost_per_consumer_per_a"]
            / self.nodes["yearly_consumption"]
        )

    def marginal_cost_per_consumer(self, pole, consumer_of_pole):
        total_consumption = self.nodes[self.nodes.index.isin(consumer_of_pole.index)][
            "yearly_consumption"
        ].sum()
        cost_of_pole = self.nodes.loc[
            self.nodes[self.nodes.index == pole].index,
            "cost_per_pole",
        ].iloc[0]
        connection_cost_consumers = self.nodes[
            self.nodes.index.isin(consumer_of_pole.index)
        ]["connection_cost_per_consumer"].sum()
        next_pole = self.nodes[self.nodes.index == pole]["parent"].iloc[0]
        for _ in range(100):
            if next_pole == "unknown":
                continue
            if (
                self.nodes[self.nodes.index == next_pole]["n_connection_links"].iloc[0]
                == 0
            ):
                if (
                    self.nodes[self.nodes.index == next_pole]["node_type"].iloc[0]
                    == "power-house"
                ):
                    break
                cost_of_pole += self.nodes.loc[
                    self.nodes[self.nodes.index == next_pole].index,
                    "cost_per_pole",
                ].iloc[0]
                next_pole = self.nodes[self.nodes.index == next_pole]["parent"].iloc[0]
            else:
                break
        marginal_cost_of_pole = (cost_of_pole + connection_cost_consumers) / (
            total_consumption + 0.0000001
        )
        return marginal_cost_of_pole

    def _cut_leaf_poles_on_condition(self):
        exclude_lst = [self.nodes[self.nodes["node_type"] == "power-house"].index[0]]
        exclude_lst.extend(
            self.nodes[self.nodes["shs_options"] == 1]["parent"].unique()
        )
        for _ in range(100):
            counter = 0
            leaf_poles = self.nodes[self.nodes["n_distribution_links"] == 1].index
            for pole in leaf_poles:
                if pole in exclude_lst:
                    continue
                consumer_of_pole = self.nodes[self.nodes["parent"] == pole]
                branch = self.nodes[self.nodes.index == pole]["branch"].iloc[0]
                consumer_of_branch = self.nodes[self.nodes["branch"] == branch].index
                average_total_cost_of_pole = consumer_of_pole[
                    "total_grid_cost_per_consumer_per_a"
                ].mean()
                average_marginal_cost_of_pole = self.marginal_cost_per_consumer(
                    pole,
                    consumer_of_pole,
                )
                self.determine_costs_per_branch(branch)
                average_marginal_branch_cost_of_pole = self.nodes.loc[
                    consumer_of_branch,
                    "cost_per_branch",
                ].iloc[0] / (
                    self.nodes.loc[consumer_of_branch, "yearly_consumption"].sum()
                    + 1e-9
                )
                if average_marginal_cost_of_pole > self.max_levelized_grid_cost or (
                    average_total_cost_of_pole > self.max_levelized_grid_cost
                    and average_marginal_branch_cost_of_pole
                    > self.max_levelized_grid_cost
                ):
                    self._cut_specific_pole(pole)
                    counter += 1
                else:
                    exclude_lst.append(pole)
            if counter == 0:
                break

    def _cut_specific_pole(self, pole):
        mask = (self.nodes["parent"] == pole) & (self.nodes["node_type"] == "consumer")
        self.nodes.loc[mask, "is_connected"] = False
        self.nodes.loc[mask, "parent"] = np.nan
        self.nodes.loc[mask, "branch"] = np.nan
        self.nodes.loc[mask, "total_grid_cost_per_consumer_per_a"] = np.nan
        self._remove_poles_and_links(pole)
        self._cut_leaf_poles_without_connection()

    def _cut_leaf_poles_without_connection(self):
        exclude_lst = [self.nodes[self.nodes["node_type"] == "power-house"].index[0]]
        for _ in range(len(self.nodes[self.nodes["node_type"] == "pole"])):
            leaf_poles = self.nodes[self.nodes["n_distribution_links"] == 1].index
            counter = 0
            for pole in leaf_poles:
                if pole in exclude_lst:
                    continue
                consumer_of_pole = self.nodes[self.nodes["parent"] == pole]
                if len(consumer_of_pole.index) == 0:
                    self._remove_poles_and_links(pole)
                    counter += 1
                else:
                    exclude_lst.append(pole)
            if counter == 0:
                break

    def _remove_poles_and_links(self, pole):
        self._correct_n_distribution_links_of_parent_poles(pole)
        self.nodes = self.nodes.drop(index=pole)
        drop_idxs = self.links[
            (self.links["from_node"] == pole) | (self.links["to_node"] == pole)
        ].index
        self.links = self.links.drop(index=drop_idxs)

    def _correct_n_distribution_links_of_parent_poles(self, pole):
        parent_pole = self.nodes[self.nodes.index == pole]["parent"].iloc[0]
        self.nodes.loc[parent_pole, "n_distribution_links"] -= 1

    def _determine_shs_consumers(self, max_iter=20):
        for _ in range(max_iter):
            self._distribute_cost_among_consumers()
            if (
                self.nodes["total_grid_cost_per_consumer_per_a"].max()
                < self.max_levelized_grid_cost
            ):
                if self.nodes["n_connection_links"].sum() == 0:
                    self.nodes = self.nodes.drop(
                        index=self.nodes[
                            self.nodes["node_type"].isin(["power-house", "pole"])
                        ].index,
                    )
                    self.links = self.links.drop(index=self.links.index)
                break
            self._cut_leaf_poles_on_condition()
        self._remove_power_house_if_no_poles_connected()

    def _remove_power_house_if_no_poles_connected(self):
        if self.nodes[self.nodes["node_type"] == "pole"].empty:
            self.nodes = self.nodes[self.nodes["node_type"] == "consumer"]
            self.links = self.links.drop(index=self.links.index)
            self.nodes["is_connected"] = False

    @staticmethod
    def change_direction_of_links(from_pole, to_pole, links):
        row_idxs = f"({from_pole}, {to_pole})"
        if row_idxs not in links.index:
            row_idxs = f"({to_pole}, {from_pole})"
        if from_pole in row_idxs.split(",")[1]:
            new_row_idxs = f"({from_pole}, {to_pole})"
            links = links.rename(index={row_idxs: new_row_idxs})
        return links

    def get_linked_poles(self, pole):
        self.distribution_links["poles"] = self.distribution_links.index.str.replace(
            r"[\(\) ]", "", regex=True
        )
        linked_poles = self.distribution_links[
            (self.distribution_links["poles"].str.contains(pole + ","))
            | ((self.distribution_links["poles"] + "#").str.contains(pole + "#"))
        ]["poles"].str.split(",")
        return linked_poles

    def _extract_and_process_poles(
        self, current_pole, linked_poles, links, examined_poles
    ):
        """Extract poles connected to `current_pole`, process their parent-child relationship,
        and return a new list of poles along with updated links.

        Args:
            current_pole (str): The pole to check for connections.
            links (DataFrame): The link structure between poles.
            examined_poles (list): List of already processed poles.

        Returns:
            list, DataFrame: Updated list of poles and modified links.
        """
        new_poles = []

        for linked_pole in linked_poles:
            # Ensure both ends of the link aren't already processed
            if (
                linked_pole[0] not in examined_poles
                and linked_pole[1] not in examined_poles
            ):
                pos = 0

                # Assign correct parent-child relation based on pole_type
                if linked_pole[pos] == current_pole:
                    assigned_pole = copy.deepcopy(linked_pole[0])
                    opposite_pole = linked_pole[1]
                else:
                    assigned_pole = copy.deepcopy(linked_pole[1])
                    opposite_pole = linked_pole[0]

                # Update parent-child relation
                self.nodes.loc[opposite_pole, "parent"] = assigned_pole
                new_poles.append(opposite_pole)

                # Modify link directions if needed
                links = self.change_direction_of_links(
                    opposite_pole, current_pole, links
                )

        return new_poles, links

    def check_all_child_poles(self, parent_pole_list, links, examined_pole_list):
        """Find and process child poles for given parent poles."""
        new_parent_poles = []
        for parent_pole in parent_pole_list:
            # Find linked poles
            linked_poles = self.get_linked_poles(parent_pole)

            child_poles, links = self._extract_and_process_poles(
                parent_pole, linked_poles, links, examined_pole_list
            )
            new_parent_poles.extend(child_poles)

        examined_pole_list += parent_pole_list
        return new_parent_poles, links, examined_pole_list

    def check_all_parent_poles(self, child_pole_list, links, examined_pole_list):
        """Find and process parent poles for given child poles."""
        new_child_poles = []
        for child_pole in child_pole_list:
            linked_poles = self.get_linked_poles(child_pole)

            parent_poles, links = self._extract_and_process_poles(
                child_pole, linked_poles, links, examined_pole_list
            )
            new_child_poles.extend(parent_poles)

        examined_pole_list += child_pole_list
        return new_child_poles, links, examined_pole_list

    def _set_direction_of_links(self):
        self._determine_distribution_links()
        links = self.get_links().copy()
        links["poles"] = links.index.str.replace(r"[\(\) ]", "", regex=True)
        poles = self._poles().copy()
        power_house_idx = poles[poles["node_type"] == "power-house"].index[0]
        parent_pole_list = [power_house_idx]
        examined_pole_list = []

        for _ in range(len(links.index)):
            if len(parent_pole_list) > 0:
                parent_pole_list, links, examined_pole_list = (
                    self.check_all_child_poles(
                        parent_pole_list,
                        links,
                        examined_pole_list,
                    )
                )
            else:
                self.nodes.loc[self.nodes["node_type"] == "power-house", "parent"] = (
                    self.nodes[self.nodes["node_type"] == "power-house"].index[0]
                )
                if len(self.nodes["parent"][self.nodes["parent"] == "unknown"]) == 0:
                    break
                child_pole_list = self.nodes[
                    (self.nodes["parent"] == "unknown")
                    & (self.nodes["n_distribution_links"] == 1)
                    & (self.nodes["node_type"] == "pole")
                ].index.tolist()
                for __ in range(len(links.index)):
                    if (
                        len(child_pole_list) > 0
                        and len(self.nodes["parent"][self.nodes["parent"] == "unknown"])
                        > 0
                    ):
                        child_pole_list, links, examined_pole_list = (
                            self.check_all_parent_poles(
                                child_pole_list,
                                links,
                                examined_pole_list,
                            )
                        )
                    else:
                        break

        links["from_node"] = (
            links.index.str.extract(r"^\(\s*([^,]+)")[0].str.strip().tolist()
        )
        links["to_node"] = (
            links.index.str.extract(r",\s*([^,]+)\s*\)$")[0].str.strip().tolist()
        )
        links = links.drop(columns=["poles"])
        mask = links["link_type"] == "connection"
        links.loc[mask, ["from_node", "to_node"]] = links.loc[
            mask,
            ["to_node", "from_node"],
        ].to_numpy()
        self.links = links.copy(deep=True)

    # ------------ CONNECT NODES USING TREE-STAR SHAPE ------------#
    def connect_grid_consumers(self):
        """
        This method create the connections between each consumer and the
        nearest pole


        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """
        # Remove all existing connections between poles and consumers
        self._clear_links(link_type="connection")

        # calculate the number of clusters and their labels obtained from kmeans clustering
        n_clusters = len(self._poles()[self._poles()["type_fixed"] == False])  # noqa: E712
        cluster_labels = self._poles()["cluster_label"]

        # create links between each node and the corresponding centroid
        for cluster in range(n_clusters):
            if len(self.nodes[self.nodes["cluster_label"] == cluster]) == 1:
                continue

            # first filter the nodes and only select those with cluster labels equal to 'cluster'
            filtered_nodes = self.nodes[
                self.nodes["cluster_label"] == cluster_labels.iloc[cluster]
            ]

            # then obtain the label of the pole which is in this cluster (as the center)
            pole_label = filtered_nodes.index[filtered_nodes["node_type"] == "pole"][0]

            filtered_nodes_idx = [str(node) for node in filtered_nodes.index]

            for node_label in filtered_nodes_idx:
                # adding consumers
                if node_label != pole_label:
                    if self.nodes.loc[node_label, "is_connected"]:
                        self._add_links(
                            label_node_from=str(pole_label),
                            label_node_to=str(node_label),
                        )
                        self.nodes.loc[node_label, "parent"] = str(pole_label)

    def connect_grid_poles(self, long_links=None):
        """
        Connect grid poles using a minimum spanning tree, breaking long links if necessary.
        """
        long_links = [] if long_links is None else long_links
        # First, all links in the grid should be removed.
        self._clear_links(link_type="distribution")

        # Now, all links from the sparse matrix obtained using the minimum
        # spanning tree are stored in `links_mst`.
        # All poles in the `links_mst` should be connected together considering:
        #   + The number of rows in the 'links_mst' reveals the number of
        #     connections.
        #   + (x,y) of each nonzero element of the 'links_mst' correspond to the
        #     (pole_from, pole_to) labels.
        links_mst = np.argwhere(self.grid_mst != 0)
        poles_index = self._poles().index

        for link_mst in links_mst:
            mst_pole_from, mst_pole_to = (
                poles_index[link_mst[0]],
                poles_index[link_mst[1]],
            )
            # Create two different combinations for each link obtained from the
            # minimum spanning tree: (px, py) and (py, px).
            # Since the direction of the link is not important here, it is
            # assumed (px, py) = (py, px).
            mst_from_to, mst_to_from = (
                f"({mst_pole_from}, {mst_pole_to})",
                f"({mst_pole_to}, {mst_pole_from})",
            )

            # If the link obtained from the minimum spanning tree is one of the
            # long links that should be removed and replaced with smaller links,
            # this part will be executed.
            if mst_from_to in long_links or mst_to_from in long_links:
                # Both `mst_from_to` and `mst_to_from` will be checked to find
                # the added poles, but only the dataframe which is not empty is
                # considered as the final `added_poles`
                added_poles = self._find_added_poles(mst_from_to, mst_to_from)
                self._break_long_link(mst_pole_from, mst_pole_to, added_poles)
            else:
                self._add_links(
                    label_node_from=mst_pole_from, label_node_to=mst_pole_to
                )

    def _find_added_poles(self, mst_from_to, mst_to_from):
        """
        Find poles added to break long links.
        In addition to the `added_poles` a flag is defined here to
        deal with the direction of adding additional poles.
        """
        added_poles_from_to = self._poles()[
            (self._poles()["type_fixed"] == True)  # noqa:E712
            & (self._poles()["how_added"] == mst_from_to)
        ]
        added_poles_to_from = self._poles()[
            (self._poles()["type_fixed"] == True)  # noqa:E712
            & (self._poles()["how_added"] == mst_to_from)
        ]

        if not added_poles_from_to.empty:
            to_from = False
            return added_poles_from_to, to_from
        elif not added_poles_to_from.empty:
            to_from = True
            return added_poles_to_from, to_from
        else:
            msg = "'added_poles' unkown"
            raise UnboundLocalError(msg)

    def _break_long_link(self, mst_pole_from, mst_pole_to, added_poles):
        """Replace long links with smaller links through added poles."""
        added_poles, to_from = added_poles
        n_added_poles = len(added_poles)
        added_pole_indices = added_poles.index

        for counter, index_added_pole in enumerate(added_pole_indices):
            if counter == 0:
                # The first `added poles` should be connected to
                # the beginning or to the end of the long link,
                # depending on the `to_from` flag.
                self._add_links(
                    label_node_from=index_added_pole if to_from else mst_pole_from,
                    label_node_to=mst_pole_to if to_from else index_added_pole,
                )
            elif counter == n_added_poles - 1:
                # The last `added poles` should be connected to
                # the end or to the beginning of the long link,
                # depending on the `to_from` flag.
                self._add_links(
                    label_node_from=mst_pole_from if to_from else index_added_pole,
                    label_node_to=index_added_pole if to_from else mst_pole_to,
                )
            else:
                self._add_links(
                    label_node_from=added_pole_indices[counter - 1],
                    label_node_to=index_added_pole,
                )
            self.nodes.loc[index_added_pole, "how_added"] = "long-distance"

    def create_minimum_spanning_tree(self):
        """
        Creates links between all poles using the Kruskal's algorithm for
        the minimum spanning tree method from scipy.sparse.csgraph.

        Parameters
        ----------
        grid (~grids.Grid):
            grid object
        """

        # total number of poles (i.e., clusters)
        poles = self._poles()
        n_poles = poles.shape[0]

        # generate all possible edges between each pair of poles
        graph_matrix = np.zeros((n_poles, n_poles))
        for i in range(n_poles):
            for j in range(n_poles):
                # since the graph does not have a direction, only the upper part of the matrix must be filled
                if j > i:
                    graph_matrix[i, j] = self.distance_between_nodes(
                        label_node_1=poles.index[i],
                        label_node_2=poles.index[j],
                    )
        # obtain the optimal links between all poles (grid_mst) and copy it in the grid object
        grid_mst = minimum_spanning_tree(graph_matrix)
        self.grid_mst = grid_mst

    #  --------------------- K-MEANS CLUSTERING ---------------------#
    def kmeans_clustering(self, n_clusters: int):
        """
        Uses a k-means clustering algorithm and returns the coordinates of the centroids.

        Parameters
        ----------
            grid (~grids.Grid):
                grid object
            n_cluster (int):
                number of clusters (i.e., k-value) for the k-means clustering algorithm

        Return
        ------
            coord_centroids: numpy.ndarray
                A numpy array containing the coordinates of the cluster centroids.
                Suppose there are two cluster with centers at (x1, y1) & (x2, y2),
                then the output array would look like:
                    array([
                        [x1, y1],
                        [x2 , y2]
                        ])
        """

        # first, all poles must be removed from the nodes list
        self._clear_poles()
        grid_consumers = self.get_grid_consumers()

        # gets (x,y) coordinates of all nodes in the grid
        nodes_coord = grid_consumers[grid_consumers["is_connected"]][
            ["x", "y"]
        ].to_numpy()

        # call kmeans clustering with constraints (min and max number of members in each cluster )
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            init="k-means++",  # 'k-means++' or 'random'
            n_init=10,
            max_iter=300,
            tol=1e-4,
            size_min=0,
            size_max=self.pole_max_connection,
            random_state=0,
            n_jobs=5,
        )
        # fit clusters to the data
        kmeans.fit(nodes_coord)

        # coordinates of the centroids of the clusters
        grid_consumers["cluster_label"] = kmeans.predict(nodes_coord)
        poles = pd.DataFrame(kmeans.cluster_centers_, columns=["x", "y"])
        poles.index.name = "cluster_label"
        poles = poles.reset_index(drop=False)
        poles.index = "p-" + poles.index.astype(str)
        poles["node_type"] = "pole"
        poles["consumer_type"] = "n.a."
        poles["consumer_detail"] = "n.a."
        poles["is_connected"] = True
        poles["how_added"] = "k-means"
        poles["latitude"] = 0
        poles["longitude"] = 0
        poles["distance_to_load_center"] = 0
        poles["type_fixed"] = False
        poles["n_connection_links"] = "0"
        poles["n_distribution_links"] = 0
        poles["parent"] = "unknown"
        poles["distribution_cost"] = 0
        self.nodes = pd.concat(
            [grid_consumers, poles, self.get_shs_consumers()],
            axis=0,
        )
        self.nodes.index = self.nodes.index.astype("str")

        # compute (lon,lat) coordinates for the poles
        self.convert_lonlat_xy(inverse=True)

    def determine_poles(self, min_n_clusters, power_house_consumers):
        """
        Computes the cost of grid based on the configuration obtained from
        the k-means clustering algorithm for different numbers of poles, and
        returns the number of poles corresponding to the lowest cost.

        Parameters
        ----------
        grid (~grids.Grid):
            'grid' object which was defined before
        min_n_clusters: int
            the minimum number of clusters required for the grid to satisfy
            the maximum number of pole connections criteria

        Return
        ------
        number_of_poles: int
            the number of poles corresponding to the minimum cost of the grid
        """
        # obtain the location of poles using kmeans clustering method
        self.kmeans_clustering(n_clusters=min_n_clusters)
        # create the minimum spanning tree to obtain the optimal links between poles
        if self.power_house is not None:
            cluster_label = self.nodes.loc["100000", "cluster_label"]
            power_house_idx = self.nodes[
                (self.nodes["node_type"] == "pole")
                & (self.nodes["cluster_label"] == cluster_label)
            ].index
            power_house_consumers["cluster_label"] = cluster_label
            power_house_consumers["consumer_type"] = np.nan
            self.nodes = pd.concat(
                [self.nodes, power_house_consumers],
            )
            self._placeholder_consumers_for_power_house(remove=True)

        self.create_minimum_spanning_tree()

        # connect all links in the grid based on the previous calculations
        self.connect_grid_consumers()
        self.connect_grid_poles()
        if self.power_house is not None:
            self.nodes.loc[self.nodes.index == power_house_idx[0], "node_type"] = (
                "power-house"
            )
            self.nodes.loc[self.nodes.index == power_house_idx[0], "how_added"] = (
                "manual"
            )

    def is_enough_poles(self, n):
        self.kmeans_clustering(n_clusters=n)
        self.connect_grid_consumers()
        constraints_violation = self.links[self.links["link_type"] == "connection"]
        constraints_violation = constraints_violation[
            constraints_violation["length"] > self.connection_cable_max_length
        ]
        return not constraints_violation.shape[0] > 0

    def _find_opt_number_of_poles(self, n_mg_consumers):
        # calculate the minimum number of poles based on the
        # maximum number of connections at each pole
        if self.pole_max_connection == 0:
            min_number_of_poles = 1
        else:
            min_number_of_poles = int(
                np.ceil(n_mg_consumers / self.pole_max_connection),
            )

        space = pd.Series(range(min_number_of_poles, n_mg_consumers, 1))

        for _ in range(min_number_of_poles, n_mg_consumers, 1):
            median_search_threshold = 5
            if len(space) >= median_search_threshold:
                next_n = int(space.median())
                if self.is_enough_poles(next_n) is True:
                    space = space[space <= next_n]
                else:
                    space = space[space > next_n]
            else:
                for next_n in space:
                    if next_n == space.iloc[-1] or self.is_enough_poles(next_n) is True:
                        return next_n
