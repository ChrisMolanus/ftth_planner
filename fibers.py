import math
from enum import Enum
from typing import List, Dict, Tuple, Set, Union, Any
import pickle
import os

import networkx

# os.environ["PROJ_LIB"] = r"C:\Users\823278\Anaconda3\envs\ftth_planner\Library\share"

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
from k_means_constrained import KMeansConstrained
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from cost_parameters import CostParameters
from trenches2 import TrenchNetwork, TrenchCorner, get_trench_network


class CableType(Enum):
    CoreToDS = "Core To DS Cable"
    DSToSplitter96Cores = "DSToSplitter96Cores Cable"
    SplitterToHouseDropCable = "Splitter To House Drop Cable"


class EquipmentType(Enum):
    ONT = "ONT"
    Splitter = "Splitter"
    StreetCabinet = "Street Cabinet"
    DecentralLocation = "Decentral Location"
    POP = "POP"


def plot_network(g_box: networkx.MultiDiGraph, building_gdf: gpd.GeoDataFrame, cabinet_df: gpd.GeoDataFrame = None):
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"] else
          "blue" if "house_trench" in d and d["house_trench"] else
          "green" if "cable" in d and d["cable"] else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    if cabinet_df is not None:
        fig, ax = cabinet_df.plot(ax=ax)
    plt.show()


class FiberCable:
    def __init__(self, trench_node_ids: List[Tuple[int, int]], length: float, cable_type: CableType):
        """
        A Fiber optic cable
        :param trench_node_ids: A list of trench nodes as a line string of trenches
        :param length: The total length of the cable
        :param cable_type: The type of cable
        """
        self.cable_type = cable_type
        self.length = length
        self.trench_node_ids = trench_node_ids


class Equipment:
    def __init__(self, e_type: EquipmentType):
        self.e_type = e_type


class StreetCabinet(Equipment):
    def __init__(self, cabinet_id: int, trench_corner: TrenchCorner):
        """
        A Street cabinet
        :param cabinet_id: The ID of the cabinet
        :param trench_corner: The Trench corner that this cabinet is on
        """
        super(StreetCabinet, self).__init__(EquipmentType.StreetCabinet)
        self.cabinet_id = cabinet_id
        self.trench_corner = trench_corner


class Splitter(Equipment):
    def __init__(self, street_cabinet: StreetCabinet):
        """
        A passive fibber splitter
        :param street_cabinet: The street cabinet this splitter is in
        """
        super(Splitter, self).__init__(EquipmentType.Splitter)
        self.street_cabinet = street_cabinet


class ONT(Equipment):
    def __init__(self, building_index, splitter: Splitter):
        """
        A Optical Network Interface, a device that terminates the fiber in a building
        :param building_index: The index of the building in the OSMX
        :param splitter: The splitter this ONT is connected to
        """
        super(ONT, self).__init__(EquipmentType.ONT)
        self.building_index = building_index
        self.splitter = splitter


class FiberNetwork:
    def __init__(self):
        """
        A Fiberoptic Network
        """
        self.fibernetwork: networkx.MultiDiGraph = None
        self.fibers: Dict[CableType, List[FiberCable]] = dict()
        self.equipment: Dict[EquipmentType, List[Equipment]] = dict()
        self.trenches: pd.DataFrame = None


class DecentralLocation(Equipment):
    def __init__(self, trench_corner: TrenchCorner, street_cabinets: List[StreetCabinet]):
        """
        A Decentralized location
        :param trench_corner: The trench corner this DS is on
        :param street_cabinets: The Street cabinets that are connected to this DS
        """
        super(DecentralLocation, self).__init__(EquipmentType.DecentralLocation)
        self.street_cabinets = street_cabinets
        self.trench_corner = trench_corner


def _get_cs_location(trench_corner_gdf, ds_look_up: Dict[int, StreetCabinet]) -> Dict[int, StreetCabinet]:
    """
    Create a Central Office Location
    :param trench_network: The Trench Network
    :param ds_look_up
    """

    cs_gdf = trench_corner_gdf[trench_corner_gdf.x.max(), trench_corner_gdf.y.min()]



    return ds_look_up


def _get_ds_locations(trench_network: TrenchNetwork, cabinet_look_up: Dict[int, StreetCabinet],
                      decentral_location_candidates: pd.DataFrame) -> Dict[int, StreetCabinet]:
    """
    Create Decentral locations
    :param trench_network: The Trench Network
    :param cabinet_look_up: The Street Cabinets
    :param decentral_location_candidates: Possible locations for Decental locations
    :return: Decental locations
    """
    cabinet_list = list()
    for cabinet_id, street_cabinet in cabinet_look_up.items():
        cabinet_list.append({"cabinet_id": cabinet_id, **street_cabinet.trench_corner})

    cabinets_df = pd.DataFrame.from_records(cabinet_list)

    decentral_location_candidates_gdf = gpd.GeoDataFrame(decentral_location_candidates,
                                                         geometry=gpd.points_from_xy(
                                                             decentral_location_candidates.x,
                                                             decentral_location_candidates.y))

    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=3, min_samples=2).fit(cabinets_df[["x", "y"]])
    print(clustering.labels_)
    print(clustering)

    cabinets_df["ds_id"] = clustering.labels_

    ds_ids = cabinets_df["ds_id"].unique()
    dc_centroid = list()
    for ds_id in ds_ids:
        points = cabinets_df[cabinets_df.ds_id == ds_id]
        x = np.sum(points.x) / len(points)
        y = np.sum(points.y) / len(points)
        dc_centroid.append({"x": x, "y": y, "ds_id": ds_id, "cabinet_ids": set(points.cabinet_id)})

    dc_centroid_df = pd.DataFrame(dc_centroid)
    dc_controid_gdf = gpd.GeoDataFrame(dc_centroid_df,
                                       geometry=gpd.points_from_xy(
                                           dc_centroid_df.x,
                                           dc_centroid_df.y))

    centroid_to_building_trench_distances = ckdnearest(decentral_location_candidates_gdf, dc_controid_gdf)
    # Find the street cabinet candidates (corners of houses) that is closest to the centroid
    idx = centroid_to_building_trench_distances.groupby('ds_id', sort=False)["dist"].transform(min) == \
          centroid_to_building_trench_distances['dist']
    dc_locations_ids = centroid_to_building_trench_distances.loc[idx, ['street_corner_id', 'ds_id', "cabinet_ids"]]
    dc_locations_ids.rename(columns={'street_corner_id': "ds_corner_id"}, inplace=True)

    ds_look_up: Dict[int, StreetCabinet] = dict()
    for index, row in dc_locations_ids.iterrows():
        sc = list()
        for cabinet_id in row["cabinet_ids"]:
            sc.append(cabinet_look_up[cabinet_id])
        ds_look_up[row['ds_id']] = DecentralLocation(trench_corner=trench_network.corner_by_id[row['ds_corner_id']],
                                                     street_cabinets=sc)
    return ds_look_up


def get_fiber_network(trench_network: TrenchNetwork, cost_parameters: CostParameters,
                      building_gdf: gpd.GeoDataFrame, g_box: networkx.MultiGraph) -> FiberNetwork:
    """
    Create a Fiber Optic Network
    :param trench_network: The Trench Network
    :param cost_parameters: The cost parameters
    :param building_gdf: The Geo DataFrame of all Buildings
    :param g_box: The Road network graph
    :return: The Fiber Optic Network
    """
    # Create a geoDataFrame with all the corners of the network (nodes)
    trench_corner_gdf = _get_trench_corner_dataframe(trench_network)

    # Create a geoDataFrame containing all the trenches in the network (edges in LineString object)
    trenches_df, trenches_gdf = _get_trench_dataframe(trench_network, cost_parameters)

    # Create Street-Cabinet-Candidate locations from building trench road side nodes
    building_trenches_df = _get_building_trenches(trench_network, building_gdf)

    # Create Dataframe for clustering
    cabinet_look_up, building_trenches_with_cabinet_df = _get_street_cabinets(trench_network, building_trenches_df)

    ds_look_up = _get_ds_locations(trench_network, cabinet_look_up, building_trenches_df)

    # Find shortest paths between the buildings and the cabinets
    fiber_network, building_fiber_graph, trenches_gdf = _get_drop_cable_network(building_trenches_with_cabinet_df,
                                                         g_box,
                                                         trench_corner_gdf,
                                                         trenches_df,
                                                         trenches_gdf,
                                                         cabinet_look_up,
                                                         cost_parameters)

    fiber_network.equipment[EquipmentType.DecentralLocation] = list(ds_look_up.values())

    fiber_network, fiber_dc_graph = _get_ds_cable_network(fiber_network,
                                                       g_box,
                                                       trench_corner_gdf,
                                                       trenches_df,
                                                       trenches_gdf,
                                                       ds_look_up,
                                                       cost_parameters)

    fig = plot_fiber_network(g_box, building_fiber_graph, fiber_dc_graph, building_gdf, cabinet_look_up, ds_look_up, None)

    return fiber_network, fig


def ckdnearest(gdA: gpd.GeoDataFrame, gdB: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    The join of the two DataFrames with a "dist" column which is teh distance between the two row points
    :param gdA: Dataframe with rows of points
    :param gdB: Dataframe with rows of points
    :return: Join of the two DataFrames with a "dist"
    """
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ],
        axis=1)

    return gdf


def _get_trench_corner_dataframe(trench_network: TrenchNetwork) -> gpd.GeoDataFrame:
    """
    Create a Dataframe from trench_network.corner_by_id
    :param trench_network: The Trench Network created by trenches.py
    :return: Dataframe from trench_network.corner_by_id
    """
    trenchCorners = trench_network.corner_by_id.values()
    trench_corner_df = pd.DataFrame(trenchCorners)
    trench_corner_gdf = gpd.GeoDataFrame(trench_corner_df, geometry=gpd.points_from_xy(
        trench_corner_df.x,
        trench_corner_df.y))
    trench_corner_gdf.set_index('node_for_adding', inplace=True)
    trench_corner_gdf = trench_corner_gdf[~trench_corner_gdf.index.duplicated(keep='first')]
    return trench_corner_gdf


def weight_calculator(length, trench_crossing, house_trench, cost_parameters: CostParameters):
    """
    Caclculate the digging cost of a trench
    :param length: The length of the trench
    :param trench_crossing: True/False is the trench a road crossing trench
    :param house_trench: True/False is the trench a trench to a building
    :param cost_parameters: The cost parameters
    :return: The digging cost in euros
    """
    if trench_crossing:
        return cost_parameters.dig_per_road_crossing
    elif house_trench:
        return length * cost_parameters.dig_building_trench_per_km
    else:
        return length * cost_parameters.dig_road_side_trench_per_km


def _get_trench_dataframe(trench_network: TrenchNetwork, cost_parameters: CostParameters) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Create Trenches Dataframes from trench_network.trenches
    :param trench_network: The Trench Network created by trenches.py
    :return: Both teh Pandas and GeoPandas dataframe
    """
    trenches_df = pd.DataFrame(trench_network.trenches)
    linestrings = list()
    for index, row in trenches_df.iterrows():
        u_id = row["u_for_edge"]
        v_id = row["v_for_edge"]
        u_node = trench_network.corner_by_id[u_id]
        v_node = trench_network.corner_by_id[v_id]
        linestring = LineString([[u_node['x'], u_node['y']], [v_node['x'], v_node['y']]])
        linestrings.append(linestring)
    trenches_df["geometry"] = linestrings
    trenches_df["dig_weight"] = trenches_df.apply(lambda row: weight_calculator(row['length'], row['trench_crossing'], row["house_trench"], cost_parameters), axis=1)
    trenches_gdf = gpd.GeoDataFrame(trenches_df)
    trenches_gdf.rename({"u_for_edge": "u", "v_for_edge": "v"}, inplace=True, axis=1)
    trenches_gdf['key'] = 1
    #trenches_gdf.set_index(['u', 'v', 'key'], inplace=True)

    trenches_gdf["min_node_id"] = trenches_gdf[['u', 'v']].min(axis=1)
    trenches_gdf["max_node_id"] = trenches_gdf[['u', 'v']].max(axis=1)
    mi = pd.MultiIndex.from_frame(trenches_gdf[["min_node_id", "max_node_id", "key"]])
    trenches_gdf.index = mi

    return trenches_df, trenches_gdf


def _get_building_trenches(trench_network: TrenchNetwork, building_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Create a Building Trench Dataframe from trench_network.building_trenches_lookup
    :param trench_network: The Trench Network created by trenches.py
    :param building_gdf: The GeoPandas Dataframe of buildings
    :return: A Building-Trench Dataframe
    """
    building_trenches = list()
    for building_index, corner_tuple in trench_network.building_trenches_lookup.items():
        building = building_gdf.loc[building_index]
        building_trenches.append({'building_corner_id': corner_tuple[0], "street_corner_id": corner_tuple[1],
                                  'street': building['addr:street'], "building_index": building_index,
                                  **trench_network.corner_by_id[corner_tuple[1]]})
    building_trenches_df = pd.DataFrame(building_trenches)
    return building_trenches_df


def _get_street_cabinets(trench_network: TrenchNetwork,
                         building_trenches_df: pd.DataFrame) -> Union[Dict[int, StreetCabinet], pd.DataFrame]:
    """
    Create street cabinets close ot teh center of building clusters (KMeansConstrained)
    :param trench_network: The Trench Network created by trenches.py
    :param building_trenches_df: The GeoPandas Dataframe of buildings
    :return: A Building-Trench Dataframe and a building_trenches_df with a cabinet_id
    """
    house_centroid_df = building_trenches_df[["x", "y", "street"]]
    house_centroids_with_street_dimensions = pd.get_dummies(house_centroid_df, columns=['street'])
    for column_name in house_centroids_with_street_dimensions.columns.values:
        if column_name.startswith("street"):
            house_centroids_with_street_dimensions[column_name].replace(1, 0.0001, inplace=True)
    min_number_of_house_clusters = int(math.ceil(len(trench_network.building_trenches_lookup) / 48))
    cabinet_clusters = KMeansConstrained(n_clusters=min_number_of_house_clusters, size_max=48, init='k-means++',
                                         n_init=10, max_iter=300, tol=0.0001, verbose=False, random_state=42,
                                         copy_x=True, n_jobs=3)
    cabinet_clusters.fit(house_centroids_with_street_dimensions)
    building_trenches_df["cabinet_id"] = cabinet_clusters.labels_
    # find the centre for each cluster and create geoDataFrame
    building_cluster_centroids = []
    for i in range(len(cabinet_clusters.cluster_centers_)):
        building_cluster_centroids.append({'x': cabinet_clusters.cluster_centers_[i][0],
                                           'y': cabinet_clusters.cluster_centers_[i][1],
                                           "centroid_id": i})
    building_centroids_df = pd.DataFrame(building_cluster_centroids)
    building_centroids_gdf = gpd.GeoDataFrame(building_centroids_df,
                                              geometry=gpd.points_from_xy(building_centroids_df.x,
                                                                          building_centroids_df.y))
    # calculation to find out distance between street cabinet candidates (corners of houses) and the centroid
    # to create street cabinet location
    streetcabinet_candidates_gdf = gpd.GeoDataFrame(building_trenches_df,
                                                    geometry=gpd.points_from_xy(
                                                        building_trenches_df.x,
                                                        building_trenches_df.y))
    centroid_to_building_trench_distances = ckdnearest(streetcabinet_candidates_gdf, building_centroids_gdf)
    # Find the street cabinet candidates (corners of houses) that is closest to the centroid
    idx = centroid_to_building_trench_distances.groupby('centroid_id', sort=False)["dist"].transform(min) == \
          centroid_to_building_trench_distances['dist']
    cabinets_ids = centroid_to_building_trench_distances.loc[idx, ['street_corner_id', 'centroid_id']]
    cabinets_ids.rename(columns={'street_corner_id': "cabinet_corner_id", 'centroid_id': 'cabinet_id'}, inplace=True)
    # cabinets_ids.set_index('cabinet_id', inplace=True)
    # cabinet_look_up = cabinets_ids.to_dict(orient="index")
    cabinet_look_up: Dict[int, StreetCabinet] = dict()
    for index, row in cabinets_ids.iterrows():
        cabinet_look_up[row['cabinet_id']] = StreetCabinet(cabinet_id=row['cabinet_id'],
                                                           trench_corner=trench_network.corner_by_id[
                                                               row['cabinet_corner_id']])
    return cabinet_look_up, building_trenches_df


def _get_drop_cable_network(building_trenches_df: pd.DataFrame, g_box: networkx.MultiGraph,
                            trench_corner_gdf: gpd.GeoDataFrame, trenches_df: pd.DataFrame, trenches_gdf: gpd.GeoDataFrame,
                            cabinet_look_up: Dict[int, StreetCabinet], cost_parameters: CostParameters) -> Union[FiberNetwork, networkx.MultiGraph]:
    """
    Create a last mile optical network which is cables form splitters to buildings
    :param building_trenches_df: The GeoPandas Dataframe of buildings with cabinet IDs
    :param g_box: The OSMX graph
    :param trench_corner_gdf: The trench corner DataFrame
    :param trenches_df: A Trench DataFrame
    :param trenches_gdf: A Trench Geo DataFrame
    :param cabinet_look_up: The Street Cabinets
    :param cost_parameters: The cost parameters
    :return: A Fiber Network object and a Building Fiber graph as a NetworkX graph
    """
    fiber_network = FiberNetwork()
    building_drop_cables, trenches_gdf = _find_shortest_path_to_buildings(cabinet_look_up, g_box,
                                                            building_trenches_df,
                                                            trench_corner_gdf, trenches_gdf, cost_parameters)
    trench_look_up = trenches_df
    # Newer versions of Geo Pandas alter the underlying Pandas Dataframe when you change it.
    if 'u' in trenches_df.columns:
        trenches_df["min_node_id"] = trenches_df[['u', 'v']].min(axis=1)
        trenches_df["max_node_id"] = trenches_df[['u', 'v']].max(axis=1)
        trenches_df["key"] = 1
        mi = pd.MultiIndex.from_frame(trenches_df[["min_node_id", "max_node_id", "key"]])
        trench_look_up.index = mi

    cables: List[FiberCable] = list()
    fiber_network.fibers[CableType.SplitterToHouseDropCable] = cables
    onts: List[ONT] = list()
    spliters: List[Splitter] = list()
    streetcabinets: List[StreetCabinet] = cabinet_look_up.values()
    fiber_network.equipment[EquipmentType.ONT] = onts
    fiber_network.equipment[EquipmentType.Splitter] = spliters
    fiber_network.equipment[EquipmentType.StreetCabinet] = streetcabinets

    building_fiber_graph = ox.graph_from_gdfs(trench_corner_gdf, gpd.GeoDataFrame(), graph_attrs=g_box.graph)
    dropcable_edges = []
    sub_cable_dict: List[dict] = list()
    all_trench_ids: Set[Tuple[int, int]] = set()
    for cable in building_drop_cables:
        path_edge = cable['shortest_path']
        cabinet_id = cable["cabinet_id"]
        dropcable_edges.append(path_edge)
        trench_ids: List[Tuple[int, int]] = list()
        length = 0.0
        for pair in list(zip(path_edge[::1], path_edge[1::1])):
            trench_id = (min(pair), max(pair), 1)
            trench_ids.append(trench_id)
            all_trench_ids.add(trench_id)
            trench = trench_look_up[trench_look_up.index == trench_id]
            if len(trench) > 0:
                length += trench.iloc[0].length
                building_fiber_graph.add_edge(pair[0], pair[1], 1, name="Fiber", cable=True,
                                     cable_type=CableType.SplitterToHouseDropCable)
                sub_cable_dict.append({"u": pair[0], "v": pair[1], "key": 1, "name": "Fiber", "cable": True,
                                       "cable_type": CableType.SplitterToHouseDropCable})
            else:
                print(f"Warning could not find trench based in u and v nodes")

        cables.append(FiberCable(trench_ids, length, CableType.SplitterToHouseDropCable))
        splitter = Splitter(cabinet_look_up[cabinet_id])
        spliters.append(splitter)
        onts.append(ONT(building_index=cable["building_index"], splitter=splitter))
    sub_cable_df = pd.DataFrame(sub_cable_dict)
    sub_cable_gdf = gpd.GeoDataFrame(sub_cable_df)
    sub_cable_gdf.set_index(['u', 'v', 'key'], inplace=True)

    fiber_network.trenches = trench_look_up.reindex(index=all_trench_ids)

    return fiber_network, building_fiber_graph, trenches_gdf


def _find_shortest_path_to_buildings(cabinet_look_up: Dict[int, StreetCabinet], g_box: networkx.MultiGraph,
                                     building_trenches_df: pd.DataFrame, trench_corner_gdf: gpd.GeoDataFrame,
                                     trenches_gdf: gpd.GeoDataFrame, cost_parameters: CostParameters) -> List[Dict[str, Any]]:
    """
    Find the shortest path from each building to its associated cabinet
    :param cabinet_look_up: The Street Cabinets
    :param g_box: The OSMX graph
    :param building_trenches_df: The building Dataframe with a "cabinet_id" column
    :param trench_corner_gdf:
    :param trenches_gdf:
    :param cost_parameters: The cost parameters
    :return: A list of last mile fiber routes
    """
    # Make a graph so we can find teh shortest paths dig_weight
    trenches_gdf["weight"] = trenches_gdf["dig_weight"] + (trenches_gdf["length"] *
                                                           (cost_parameters.fiber_install_per_km +
                                                            cost_parameters.fiber_drop_pair_per_km))
    graph = ox.graph_from_gdfs(trench_corner_gdf, trenches_gdf.drop("key", axis=1), graph_attrs=g_box.graph)
    # make sure to convert to undirected graph
    graph = graph.to_undirected()
    building_drop_cables = list()
    for index, street_trench in building_trenches_df.iterrows():
        building_index = street_trench["building_index"]
        house_node_id = street_trench['building_corner_id']
        cabinet_id = street_trench['cabinet_id']
        cabinet_corner = cabinet_look_up[cabinet_id].trench_corner
        cabinet_corner_id = cabinet_corner['node_for_adding']
        try:
            s_path = nx.algorithms.shortest_paths.shortest_path(graph, source=house_node_id, target=cabinet_corner_id, weight="weight")
            building_drop_cables.append(
                {"building_corner_id": house_node_id, "cabinet_id": cabinet_id, "cabinet_corner_id": cabinet_corner_id,
                 "shortest_path": s_path, "building_index": building_index})
        except networkx.exception.NetworkXNoPath:
            pass
            # print(f"No drop cable path could be found for building_index {building_index}")
        # update graph edge "weight" for every edge in s_path by removing the one time cost for digging
        for pair in list(zip(s_path[::1], s_path[1::1])):
            edge = graph.edges[pair[0], pair[1], 1]
            edge["weight"] = (edge["length"] * (cost_parameters.fiber_install_per_km + cost_parameters.fiber_drop_pair_per_km))
        # update trenches_gdf["dig_weight"] to 0 for all the trench that were in the s_path(s)
        trenches_gdf["dig_weight"] = trenches_gdf["dig_weight"].mask(trenches_gdf["u"].isin(s_path), 0)\
                                                               .mask(trenches_gdf["v"].isin(s_path), 0)

    return building_drop_cables, trenches_gdf


def _find_shortest_path_to_cabinets(ds_look_up, g_box: networkx.MultiGraph, trench_corner_gdf: gpd.GeoDataFrame,
                                    trenches_gdf: gpd.GeoDataFrame, cost_parameters: CostParameters) -> List[Dict[str, Any]]:
    """
    Find the shortest path from each street cabinet to its associated decentral location
    :param ds_look_up: The Street Cabinets
    :param g_box: The OSMX graph
    :param trench_corner_gdf:
    :param trenches_gdf:
    :param cost_parameters: The cost parameters
    :return: A list of last mile fiber routes
    """
    # Make a graph so we can find teh shortest paths
    trenches_gdf["weight"] = trenches_gdf["dig_weight"] + (trenches_gdf["length"] *
                                                           (cost_parameters.fiber_install_per_km +
                                                            cost_parameters.fiber_96core_per_km))
    graph = ox.graph_from_gdfs(trench_corner_gdf, trenches_gdf.drop("key", axis=1), graph_attrs=g_box.graph)
    # make sure to convert to undirected graph
    graph = graph.to_undirected()
    ds_fiber_cables = list()
    for index, ds in ds_look_up.items():
        ds_corner_id = ds.trench_corner['node_for_adding']
        for sc_index in ds.street_cabinets:
            street_cabinet_id = sc_index.cabinet_id
            street_cabinet_corner_id = sc_index.trench_corner
            cabinet_corner_id = street_cabinet_corner_id['node_for_adding']
            try:
                s_path = nx.algorithms.shortest_paths.shortest_path(graph, source=cabinet_corner_id,
                                                                    target=ds_corner_id, weight="weight")
                ds_fiber_cables.append(
                    {"cabinet_corner_id": cabinet_corner_id, "ds_id": ds, "ds_corner_id": ds_corner_id,
                     "shortest_path": s_path, "decentral_locations": ds, 'street_cabinet_id': street_cabinet_id})
            except networkx.exception.NetworkXNoPath:
                pass
                # print(f"No drop cable path could be found for building_index {building_index}")
            for pair in list(zip(s_path[::1], s_path[1::1])):
                edge = graph.edges[pair[0], pair[1], 1]
                edge["weight"] = (edge["length"] * (
                            cost_parameters.fiber_install_per_km + cost_parameters.fiber_96core_per_km))
            # update trenches_gdf["dig_weight"] to 0 for all the trench that were in the s_path(s)
            trenches_gdf["dig_weight"] = trenches_gdf["dig_weight"].mask(trenches_gdf["u"].isin(s_path), 0)\
                                                                   .mask(trenches_gdf["v"].isin(s_path), 0)

    return ds_fiber_cables


def _get_ds_cable_network(fiber_network: FiberNetwork(), g_box: networkx.MultiGraph,
                          trench_corner_gdf: gpd.GeoDataFrame, trenches_df, trenches_gdf,
                          ds_look_up: Dict[int, StreetCabinet], cost_parameters: CostParameters) -> Union[FiberNetwork, networkx.MultiGraph]:
    """
    Create a last mile optical network which is cables form splitters to buildings
    :param building_trenches_df: The GeoPandas Dataframe of buildings with cabinet IDs
    :param g_box: The OSMX graph
    :param trench_corner_gdf: The trench corner DataFrame
    :param trenches_df: A Trench DataFrame
    :param trenches_gdf: A Trench Geo DataFrame
    :param ds_look_up: The Decentralized Locations
    :param cost_parameters: The cost parameters
    :return: A Fiber Network object and a Fiber graph as a NetworkX graph
    """
    ds_fiber_cables = _find_shortest_path_to_cabinets(ds_look_up, g_box, trench_corner_gdf, trenches_gdf, cost_parameters)

    fiber_dc_graph = ox.graph_from_gdfs(trench_corner_gdf, gpd.GeoDataFrame(), graph_attrs=g_box.graph)

    trenches_df["min_node_id"] = trenches_df[['u', 'v']].min(axis=1)
    trenches_df["max_node_id"] = trenches_df[['u', 'v']].max(axis=1)
    mi = pd.MultiIndex.from_frame(trenches_df[["min_node_id", "max_node_id"]])
    trench_look_up = trenches_df
    trench_look_up.index = mi

    cables: List[FiberCable] = list()
    fiber_network.fibers[CableType.DSToSplitter96Cores] = cables

    ds_fiber_cable_edges = []
    sub_cable_dict: List[dict] = list()
    all_trench_ids: Set[Tuple[int, int]] = set()
    for cable in ds_fiber_cables:
        path_edge = cable['shortest_path']
        ds_id = cable["ds_id"]
        cabinet_id = cable['cabinet_corner_id']
        ds_fiber_cable_edges.append(path_edge)
        trench_ids: List[Tuple[int, int]] = list()
        length = 0.0
        for pair in list(zip(path_edge[::1], path_edge[1::1])):
            trench_id = (min(pair), max(pair))
            trench_ids.append(trench_id)
            all_trench_ids.add(trench_id)
            trench = trench_look_up[trench_look_up.index == trench_id]
            length += trench.iloc[0].length
            fiber_dc_graph.add_edge(pair[0], pair[1], 1, name="DS_Fiber", cable=True,
                                 cable_type=CableType.DSToSplitter96Cores)
            sub_cable_dict.append({"u": pair[0], "v": pair[1], "key": 2, "name": "DS_Fiber", "cable": True,
                                   "cable_type": CableType.DSToSplitter96Cores})

        cables.append(FiberCable(trench_ids, length, CableType.DSToSplitter96Cores))
    sub_cable_df = pd.DataFrame(sub_cable_dict)
    sub_cable_gdf = gpd.GeoDataFrame(sub_cable_df)
    sub_cable_gdf.set_index(['u', 'v', 'key'], inplace=True)

    fiber_network.trenches = pd.concat([fiber_network.trenches, trench_look_up.loc[all_trench_ids]])

    return fiber_network, fiber_dc_graph


def _find_shortest_path_to_cs(cs_look_up, g_box: networkx.MultiGraph, trench_corner_gdf: gpd.GeoDataFrame,
                              trenches_gdf: gpd.GeoDataFrame) -> List[Dict[str, Any]]:
    """
    Find the shortest path from each decentrale to its associated central location
    :param ds_look_up: The Decentral Locations
    :param g_box: The OSMX graph
    :param trench_corner_gdf:
    :param trenches_gdf:
    :return: A list of last mile fiber routes
    """
    # Make a graph so we can find teh shortest paths
    graph = ox.graph_from_gdfs(trench_corner_gdf, trenches_gdf, graph_attrs=g_box.graph)
    # make sure to convert to undirected graph
    graph = graph.to_undirected()
    cs_fiber_cables = list()
    for index, ds in cs_look_up.items():
        ds_corner_id = ds.trench_corner['node_for_adding']
        for sc_index in ds.street_cabinets:
            street_cabinet_id = sc_index.cabinet_id
            street_cabinet_corner_id = sc_index.trench_corner
            cabinet_corner_id = street_cabinet_corner_id['node_for_adding']
            try:
                s_path = nx.algorithms.shortest_paths.shortest_path(graph, source=cabinet_corner_id,
                                                                    target=ds_corner_id, weight="length")
                cs_fiber_cables.append(
                    {"cabinet_corner_id": cabinet_corner_id, "ds_id": ds, "ds_corner_id": ds_corner_id,
                     "shortest_path": s_path, "decentral_locations": ds, 'street_cabinet_id': street_cabinet_id})
            except networkx.exception.NetworkXNoPath:
                pass
                # print(f"No drop cable path could be found for building_index {building_index}")

    return cs_fiber_cables


def plot_fiber_network(road_graph, building_fiber_graph, fiber_dc_graph, building_gdf, cabinet_look_up: Dict[int, StreetCabinet], ds_look_up,
                       cs_lookup = None):
    cabinet_list = list()
    for cluster_id, d in cabinet_look_up.items():
        node = d.trench_corner
        cabinet_list.append(
            {"x": node["x"], "y": node["y"], "key": 2, "name": "cabinet " + str(cluster_id), "equipment": True,
             "equipment_type": EquipmentType.StreetCabinet})
    cabinet_df = pd.DataFrame(cabinet_list)

    ds_list = list()
    for cluster_id, d in ds_look_up.items():
        node = d.trench_corner
        ds_list.append(
            {"x": node["x"], "y": node["y"], "key": 1, "name": "ds " + str(cluster_id), "equipment": True,
             "equipment_type": EquipmentType.DecentralLocation})
    ds_df = pd.DataFrame(ds_list)

    fig, ax = ox.plot_graph(road_graph, bgcolor='white', edge_color="lightgrey",
                            node_size=0, edge_linewidth=0.8, edge_alpha=1,
                            show=False, close=False)

    ec = ["grey" if "trench_crossing" in d and d["trench_crossing"] else
          "pink" if "house_trench" in d and d["house_trench"] else
          'blue' if "cable" in d and d["cable_type"] == CableType.SplitterToHouseDropCable else
          'red' for _, _, _, d in building_fiber_graph.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(building_fiber_graph, bgcolor=None, edge_color=ec,
                            node_size=0, edge_linewidth=1.8, edge_alpha=1,
                            show=False, close=False, ax=ax)

    fig, ax = ox.plot_graph(fiber_dc_graph, bgcolor=None, edge_color="lime",
                            node_size=0, edge_linewidth=2, edge_alpha=1,
                            show=False, close=False, ax=ax)

    fig, ax = ox.plot_footprints(building_gdf, ax=ax, color="burlywood", alpha=0.6, show=False, close=False)

    ax.scatter(None, None, label='Splitter to House Drop Cable', color='blue')
    ax.scatter(None, None, label='DSToSplitter96Cores Cable', color='lime')
    ax.scatter(cabinet_df.x, cabinet_df.y, s=35, color="m", label='Street Cabinet')
    ax.scatter(ds_df.x, ds_df.y, s=70, color="yellow", label='Decentral Location')

    return(fig)


if __name__ == "__main__":
    # Try and load cached data for speed
    # box2 = (51.98446, 51.98000, 5.64113, 5.6575)
    box = (50.843217, 50.833949, 4.439903, 4.461962)
    # box = (52.38132054097, 52.36193148749, 4.84358307250, 4.884481392928)
    if not os.path.isfile("g_box.p"):
        g_box = ox.graph_from_bbox(*box,
                                   network_type='drive',
                                   simplify=False,
                                   retain_all=False,
                                   truncate_by_edge=True)
        pickle.dump(g_box, open("g_box.p", "wb"))
    else:
        g_box: networkx.MultiGraph = pickle.load(open("g_box.p", "rb"))

    if not os.path.isfile("building_gdf.p"):
        building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
        pickle.dump(building_gdf, open("building_gdf.p", "wb"))
    else:
        building_gdf: gpd.GeoDataFrame = pickle.load(open("building_gdf.p", "rb"))

    if not os.path.isfile("trench_network.p"):
        trench_network = get_trench_network(g_box, building_gdf)
        pickle.dump(trench_network, open("trench_network.p", "wb"))
    else:
        trench_network: TrenchNetwork = pickle.load(open("trench_network.p", "rb"))

    cost_parameters = CostParameters()
    fiber_network, fig = get_fiber_network(trench_network, cost_parameters, building_gdf, g_box)
    plt.show()
