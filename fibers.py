from enum import Enum
from typing import List, Dict
import pickle

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
from trenches2 import TrenchNetwork, TrenchCorner, add_trenches_to_network


class CableType(Enum):
    CoreToDS = 1
    DSToSplitter96Cores = 2
    SpliterToHouseDropCable = 3


class EquipmentType(Enum):
    Splitter = 1
    StreetCabinet = 2
    DecentralLocation = 3
    CentralLocation = 4
    POP = 5


class FiberCable:
    def __init__(self, trench_osmids: List[int], length: float, cable_type: CableType):
        pass


class Equipment:
    def __init__(self, id: int, x: float, y: float, e_type: EquipmentType):
        self.id = id
        self.x = x
        self.y = y
        self.e_type = e_type


class FiberNetwork:
    def __init__(self):
        self.fiber_network: networkx.MultiDiGraph = None
        self.fibers: Dict[int, FiberCable] = None
        self.equipment: Dict[int, Equipment] = None


def get_fiber_network(trench_network: TrenchNetwork, cost_parameters: CostParameters,
                      building_gdf: gpd.GeoDataFrame) -> FiberNetwork:
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


if __name__ == "__main__":
    trench_network: TrenchNetwork = pickle.load(open("trench_network.p", "rb"))
    cost_parameters = CostParameters()
    get_fiber_network(trench_network, cost_parameters)

    g_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                               network_type='drive',
                               simplify=False,
                               retain_all=False,
                               truncate_by_edge=True)
    building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
    houses_filter = building_gdf.filter(trench_network.building_trenches_lookup.keys(), axis=0)

    # create a geoDataFrame with all the corners of the network (nodes)
    corner_by_id: Dict[int, TrenchCorner] = dict()
    trenchCorners: List[TrenchCorner] = list()
    for street_id, corners in trench_network.trenchCorners.items():
        for corner in corners:
            corner_by_id[corner['node_for_adding']] = corner
            trenchCorners.append(corner)

    street_corner_df = pd.DataFrame(trenchCorners)
    street_corner_gdf = gpd.GeoDataFrame(street_corner_df, geometry=gpd.points_from_xy(
        street_corner_df.x,
        street_corner_df.y))
    street_corner_gdf.set_index('node_for_adding', inplace=True)
    street_corner_unique_gdf = street_corner_gdf[~street_corner_gdf.index.duplicated(keep='first')]

    # create a geoDataFrame containing all the trenches in the network (edges in LineString object)
    trenches_df = pd.DataFrame(trench_network.trenches)
    linestrings = list()
    for index, row in trenches_df.iterrows():
        u_id = row["u_for_edge"]
        v_id = row["v_for_edge"]
        u_node = corner_by_id[u_id]
        v_node = corner_by_id[v_id]
        linestring = LineString([[u_node['x'], u_node['y']], [v_node['x'], v_node['y']]])
        linestrings.append(linestring)
    trenches_df["geometry"] = linestrings

    trenches_gdf = gpd.GeoDataFrame(trenches_df)
    trenches_gdf.rename({"u_for_edge": "u", "v_for_edge": "v"}, inplace=True, axis=1)
    trenches_gdf['key'] = 1
    trenches_gdf.set_index(['u', 'v', 'key'], inplace=True)

    # create network from nodes and edges geoDataFrames
    # plot the network
    G = ox.graph_from_gdfs(street_corner_unique_gdf, trenches_gdf)
    ox.plot_graph(G)
    G_undirect = G.to_undirected()


    # find out all the street corners of the houses where streetcabinets need to connect to
    cabinetcorners = list()
    for building_index, corner_tuple in trench_network.building_trenches_lookup.items():
        cabinetcorners.append({'building_corner_id': corner_tuple[0], **corner_by_id[corner_tuple[1]]})

    streetcabinet_candidates_df = pd.DataFrame(cabinetcorners)
    streetcabinet_candidates_gdf = gpd.GeoDataFrame(streetcabinet_candidates_df,
                                                    geometry=gpd.points_from_xy(
                                                        streetcabinet_candidates_df.x,
                                                        streetcabinet_candidates_df.y))

    # create a dummy dataset of all the houses in the trench network for the KMeans clustering
    houses_list = []
    for key, building in houses_filter.iterrows():
        centroid = building['geometry'].centroid
        building_centroid_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0], 'street': building['addr:street']}
        houses_list.append(building_centroid_node)

    houses_df = pd.DataFrame(houses_list)
    houses_dummy = pd.get_dummies(houses_df, columns=['street'])
    houses_dummy.iloc[:, 2:] = houses_dummy.iloc[:, 2:] / 1000

    house_clusters = int(round(len(houses_filter.index) / 48, 0))
    # scaler = StandardScaler()
    # scaler.fit(houses_dummy)
    kmeans = KMeansConstrained(n_clusters=house_clusters, size_max=48, init='k-means++', n_init=10, max_iter=300,
                               tol=0.0001, verbose=False, random_state=42, copy_x=True, n_jobs=3)
    kmeans.fit(houses_dummy)

    # find the centre for each cluster and create geoDataFrame
    houses_centroids = []
    for i in range(len(kmeans.cluster_centers_)):
        houses_centroids.append({'x': kmeans.cluster_centers_[i][0], 'y': kmeans.cluster_centers_[i][1]})

    hs_centroids_df = pd.DataFrame(houses_centroids)
    hs_centroids_gdf = gpd.GeoDataFrame(hs_centroids_df, geometry=gpd.points_from_xy(hs_centroids_df.x,
                                                                                     hs_centroids_df.y))
    hs_centroids_gdf["centroid_id"] = hs_centroids_gdf.index

    # calculation to find out distance between streetcabinet candidates (corners of houses) and the centroid
    # to create streetcabinet location
    houses_gdf = ckdnearest(streetcabinet_candidates_gdf, hs_centroids_gdf)
    houses_gdf["cluster_id"] = kmeans.labels_

    idx = houses_gdf.groupby('centroid_id', sort=False)["dist"].transform(min) == houses_gdf['dist']
    cabinets_ids = houses_gdf.loc[idx, ['node_for_adding', 'centroid_id']]

    streetcabinets_gdf = houses_gdf.iloc[:, [-3, -2, 1]].drop_duplicates()

    plt.scatter(x=streetcabinet_candidates_gdf.x, y=streetcabinet_candidates_gdf.y, c=kmeans.labels_)
    plt.scatter(x=houses_dummy.x, y=houses_dummy.y, c=kmeans.labels_)
    plt.scatter(x=streetcabinets_gdf.x, y=streetcabinets_gdf.y, c='green')
    plt.show()
    # TODO: connect houses and street cabinets to trench network, add column per row to add id for trenchCorners

    # create a network that connects the houses nodes to the corresponding street cabinets nodes, using the trenches and trenchcorners
    ## first create fiber from house to street trench (seperate fiber cable)
    ### second create shortest path (Dijkstra) from trench_network.trenchCorners that connects house to street cabinet using the trench_network.trenches

    cabinets_ids.set_index('centroid_id', inplace=True)
    cabinet_look_up = cabinets_ids.to_dict(orient="index")
    building_drop_cables = list()
    for index, row in houses_gdf.iterrows():
        house_node_id = row['building_corner_id']
        cluster_id = row['cluster_id']
        street_cabinet_node_id = cabinet_look_up[cluster_id]['node_for_adding']
        s_path = nx.algorithms.shortest_paths.shortest_path(G_undirect, source=house_node_id, target=street_cabinet_node_id)
        building_drop_cables.append(
            {"building_corner_id": house_node_id, "cluster_id": cluster_id, "streetcabinet_id": street_cabinet_node_id,
             "shortest_path": s_path})



    def gdf_to_nx(gdf_network):
        # generate graph from GeoDataFrame of LineStrings
        net = nx.Graph()
        net.graph['crs'] = gdf_network.crs
        fields = list(gdf_network.columns)

        for index, row in gdf_network.iterrows():
            first = row.geometry.coords[0]
            last = row.geometry.coords[-1]

            data = [row[f] for f in fields]
            attributes = dict(zip(fields, data))
            net.add_edge(first, last, **attributes)

        return net


    def nx_to_gdf(net, nodes=True, edges=True):
        # generate nodes and edges geodataframes from graph
        if nodes is True:
            node_xy, node_data = zip(*net.nodes(data=True))
            gdf_nodes = gpd.GeoDataFrame(list(node_data), geometry=[Point(i, j) for i, j in node_xy])
            gdf_nodes.crs = net.graph['crs']

        if edges is True:
            starts, ends, edge_data = zip(*net.edges(data=True))
            gdf_edges = gpd.GeoDataFrame(list(edge_data))
            gdf_edges.crs = net.graph['crs']

        if nodes is True and edges is True:
            return gdf_nodes, gdf_edges
        elif nodes is True and edges is False:
            return gdf_nodes
        else:
            return gdf_edges

    # TODO: houses dijkstra algorithm to streetcabinets
