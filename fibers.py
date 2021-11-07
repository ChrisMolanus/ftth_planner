import math
from enum import Enum
from typing import List, Dict, Tuple, Set
import pickle
import os

import networkx


#os.environ["PROJ_LIB"] = r"C:\Users\823278\Anaconda3\envs\ftth_planner\Library\share"

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from k_means_constrained import KMeansConstrained
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

from costs import CostParameters
from trenches2 import TrenchNetwork, TrenchCorner, add_trenches_to_network, get_trench_network


class CableType(Enum):
    CoreToDS = "CoreToDS"
    DSToSplitter96Cores = "DSToSplitter96Cores"
    SpliterToHouseDropCable = "SpliterToHouseDropCable"


class EquipmentType(Enum):
    ONT = "ONT"
    Splitter = "Splitter"
    StreetCabinet = "StreetCabinet"
    DecentralLocation = "DecentralLocation"
    CentralLocation = "CentralLocation"
    POP = "POP"


def plot_network(g_box: networkx.MultiDiGraph, building_gdf: gpd.GeoDataFrame, cabinet_df: gpd.GeoDataFrame=None):
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"] else
          "blue" if "house_trench" in d and d["house_trench"] else
          "green"if "cable" in d and d["cable"] else
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
        pass


class Equipment:
    def __init__(self, e_type: EquipmentType):
        self.e_type = e_type


class StreetCabinet(Equipment):
    def __init__(self, cabinet_id: int, trench_corner: TrenchCorner):
        super(StreetCabinet, self).__init__(EquipmentType.StreetCabinet)
        self.cabinet_id = cabinet_id
        self.trench_corner = trench_corner


class Splitter(Equipment):
    def __init__(self, street_cabinet: StreetCabinet):
        super(Splitter, self).__init__(EquipmentType.Splitter)
        self.street_cabinet = street_cabinet


class ONT(Equipment):
    def __init__(self, building_index, splitter: Splitter):
        super(ONT, self).__init__(EquipmentType.ONT)
        self.building_index = building_index
        self.splitter = splitter


class FiberNetwork:
    def __init__(self):
        self.fiber_network: networkx.MultiDiGraph = None
        self.fibers: Dict[CableType, List[FiberCable]] = dict()
        self.equipment: Dict[EquipmentType, List[Equipment]] = dict()
        self.trenches: pd.DataFrame = None


def get_fiber_network(trench_network: TrenchNetwork, cost_parameters: CostParameters,
                      building_gdf: gpd.GeoDataFrame, g_box: networkx.MultiGraph) -> FiberNetwork:

    # Create a geoDataFrame with all the corners of the network (nodes)
    trench_corner_gdf = get_trench_corner_dataframe(trench_network)

    # Create a geoDataFrame containing all the trenches in the network (edges in LineString object)
    trenches_df, trenches_gdf = get_trench_dataframe(trench_network)

    # Create Street-Cabinet-Candidate locations from building trench road side nodes
    streetcabinet_candidates_df = get_streetcabinet_candidates(trench_network, building_gdf)

    # Create Dataframe for clustering
    cabinet_look_up = get_street_cabinets(trench_network, streetcabinet_candidates_df)

    # Find shortest paths between the buildings and the cabinets
    fiber_network, fiber_graph = get_drop_cable_network(streetcabinet_candidates_df,
                                                        g_box,
                                                        trench_corner_gdf,
                                                        trenches_df,
                                                        trenches_gdf,
                                                        cabinet_look_up)

    plot_fiber_network(fiber_graph, building_gdf, cabinet_look_up)

    return FiberNetwork()


def ckdnearest(gdA, gdB):
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


def get_trench_corner_dataframe(trench_network: TrenchNetwork):
    trenchCorners = trench_network.corner_by_id.values()
    trench_corner_df = pd.DataFrame(trenchCorners)
    trench_corner_gdf = gpd.GeoDataFrame(trench_corner_df, geometry=gpd.points_from_xy(
        trench_corner_df.x,
        trench_corner_df.y))
    trench_corner_gdf.set_index('node_for_adding', inplace=True)
    trench_corner_gdf = trench_corner_gdf[~trench_corner_gdf.index.duplicated(keep='first')]
    return trench_corner_gdf


def get_trench_dataframe(trench_network: TrenchNetwork):
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
    trenches_gdf = gpd.GeoDataFrame(trenches_df)
    trenches_gdf.rename({"u_for_edge": "u", "v_for_edge": "v"}, inplace=True, axis=1)
    trenches_gdf['key'] = 1
    trenches_gdf.set_index(['u', 'v', 'key'], inplace=True)
    return trenches_df, trenches_gdf


def get_streetcabinet_candidates(trench_network: TrenchNetwork, building_gdf: gpd.GeoDataFrame):
    streetcabinet_candidates = list()
    for building_index, corner_tuple in trench_network.building_trenches_lookup.items():
        building = building_gdf.loc[building_index]
        streetcabinet_candidates.append({'building_corner_id': corner_tuple[0], "street_corner_id": corner_tuple[1],
                                         'street': building['addr:street'], "building_index": building_index,
                                         **trench_network.corner_by_id[corner_tuple[1]]})
    streetcabinet_candidates_df = pd.DataFrame(streetcabinet_candidates)
    return streetcabinet_candidates_df


def get_street_cabinets(trench_network: TrenchNetwork,
                        streetcabinet_candidates_df: pd.DataFrame):
    house_centroid_df = streetcabinet_candidates_df[["x", "y", "street"]]
    house_centroids_with_street_dimentions = pd.get_dummies(house_centroid_df, columns=['street'])
    for column_name in house_centroids_with_street_dimentions.columns.values:
        if column_name.startswith("street"):
            house_centroids_with_street_dimentions[column_name].replace(1, 0.0001, inplace=True)
    min_number_of_house_clusters = int(math.ceil(len(trench_network.building_trenches_lookup) / 48))
    cabinet_clusters = KMeansConstrained(n_clusters=min_number_of_house_clusters, size_max=48, init='k-means++',
                                         n_init=10, max_iter=300, tol=0.0001, verbose=False, random_state=42,
                                         copy_x=True, n_jobs=3)
    cabinet_clusters.fit(house_centroids_with_street_dimentions)
    streetcabinet_candidates_df["cabinet_id"] = cabinet_clusters.labels_
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
    streetcabinet_candidates_gdf = gpd.GeoDataFrame(streetcabinet_candidates_df,
                                                    geometry=gpd.points_from_xy(
                                                        streetcabinet_candidates_df.x,
                                                        streetcabinet_candidates_df.y))
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
                                                           trench_corner=trench_network.corner_by_id[row['cabinet_corner_id']])
    return cabinet_look_up


def get_drop_cable_network(streetcabinet_candidates_df: pd.DataFrame, g_box, trench_corner_gdf, trenches_df, trenches_gdf, cabinet_look_up):
    fiber_network = FiberNetwork()
    building_drop_cables = find_shortest_path_to_buildings(cabinet_look_up, g_box,
                                                                           streetcabinet_candidates_df,
                                                                           trench_corner_gdf, trenches_gdf)

    trenches_df["min_node_id"] = trenches_df[['u', 'v']].min(axis=1)
    trenches_df["max_node_id"] = trenches_df[['u', 'v']].max(axis=1)
    mi = pd.MultiIndex.from_frame(trenches_df[["min_node_id", "max_node_id"]])
    trench_look_up = trenches_df
    trench_look_up.index = mi

    cables: List[FiberCable] = list()
    fiber_network.fibers[CableType.SpliterToHouseDropCable] = cables
    onts: List[ONT] = list()
    spliters: List[Splitter] = list()
    streetcabinets: List[StreetCabinet] = list()
    fiber_network.equipment[EquipmentType.ONT] = onts
    fiber_network.equipment[EquipmentType.Splitter] = spliters
    fiber_network.equipment[EquipmentType.StreetCabinet] = streetcabinets

    fiber_graph = ox.graph_from_gdfs(trench_corner_gdf, gpd.GeoDataFrame(), graph_attrs=g_box.graph)
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
            trench_id = (min(pair), max(pair))
            trench_ids.append(trench_id)
            all_trench_ids.add(trench_id)
            trench = trench_look_up[trench_look_up.index == trench_id]
            length += trench.length
            fiber_graph.add_edge(pair[0], pair[1], 1, name="Fiber", cable=True,
                                 cable_type=CableType.SpliterToHouseDropCable)
            sub_cable_dict.append({"u": pair[0], "v": pair[1], "key": 1, "name": "Fiber", "cable": True,
                                   "cable_type": CableType.SpliterToHouseDropCable})

        cables.append(FiberCable(trench_ids, length, CableType.SpliterToHouseDropCable))
        splitter = Splitter(cabinet_look_up[cabinet_id])
        spliters.append(splitter)
        onts.append(ONT(building_index=cable["building_index"], splitter=splitter))
    sub_cable_df = pd.DataFrame(sub_cable_dict)
    sub_cable_gdf = gpd.GeoDataFrame(sub_cable_df)
    sub_cable_gdf.set_index(['u', 'v', 'key'], inplace=True)

    fiber_network.trenches = trench_look_up.loc[all_trench_ids]

    return fiber_network, fiber_graph


def find_shortest_path_to_buildings(cabinet_look_up, g_box, streetcabinet_candidates_df,
                                    trench_corner_gdf, trenches_gdf):
    # Make a graph so we can find teh shortest paths
    graph = ox.graph_from_gdfs(trench_corner_gdf, trenches_gdf, graph_attrs=g_box.graph)
    # make sure to convert to undirected graph
    graph = graph.to_undirected()
    building_drop_cables = list()
    for index, street_trench in streetcabinet_candidates_df.iterrows():
        building_index = street_trench["building_index"]
        house_node_id = street_trench['building_corner_id']
        cabinet_id = street_trench['cabinet_id']
        cabinet_corner = cabinet_look_up[cabinet_id].trench_corner
        cabinet_corner_id = cabinet_corner['node_for_adding']
        try:
            s_path = nx.algorithms.shortest_paths.shortest_path(graph, source=house_node_id, target=cabinet_corner_id)
            building_drop_cables.append(
                {"building_corner_id": house_node_id, "cabinet_id": cabinet_id, "cabinet_corner_id": cabinet_corner_id,
                 "shortest_path": s_path, "building_index": building_index})
        except networkx.exception.NetworkXNoPath:
            pass
            # print(f"No drop cable path could be found for building_index {building_index}")

    return building_drop_cables


def plot_fiber_network(fiber_graph, building_gdf, cabinet_look_up):
    cabinet_list = list()
    for cluster_id, d in cabinet_look_up.items():
        node = d.trench_corner
        cabinet_list.append(
            {"x": node["x"], "y": node["y"], "key": 1, "name": "cabinet " + str(cluster_id), "equipment": True,
             "equipment_type": EquipmentType.StreetCabinet})
    cabinet_df = pd.DataFrame(cabinet_list)
    cabinet_gdf = gpd.GeoDataFrame(
        cabinet_df, geometry=gpd.points_from_xy(cabinet_df.x, cabinet_df.y))
    cabinet_gdf.set_index(['x', 'y', 'key'], inplace=True)
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"] else
          "blue" if "house_trench" in d and d["house_trench"] else
          "green" if "cable" in d and d["cable_type"] == CableType.SpliterToHouseDropCable else
          'red' for _, _, _, d in fiber_graph.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(fiber_graph, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    ax.scatter(cabinet_df.x, cabinet_df.y, s=7)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)


if __name__ == "__main__":
    # Try and load cached data for speed
    box = (50.843217, 50.833949, 4.439903, 4.461962)
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
    get_fiber_network(trench_network, cost_parameters, building_gdf, g_box)




