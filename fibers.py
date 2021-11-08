from enum import Enum
from typing import List, Dict

import matplotlib.pyplot as plt
import networkx

from costs import CostParameters
from trenches2 import TrenchNetwork, TrenchCorner


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


def get_fiber_network(trench_network: TrenchNetwork, cost_parameters: CostParameters) -> FiberNetwork:
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
    import pickle
    trench_network: TrenchNetwork = pickle.load(open("trench_network.p", "rb"))
    cost_parameters = CostParameters()
    get_fiber_network(get_fiber_network, cost_parameters)

    import osmnx as ox
    import pandas as pd
    import sklearn
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    import geopandas
    from k_means_constrained import KMeansConstrained
    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
    houses_list = []
    houses_filter = building_gdf.filter(trench_network.building_trenches_lookup.keys(), axis=0)

    corner_by_id: Dict[int, TrenchCorner] = dict()
    for street_id, corners in trench_network.trenchCorners.items():
        for corner in corners:
            corner_by_id[corner['node_for_adding']] = corner

    cabinetcorners = list()
    for building_index, corner_tuple in trench_network.building_trenches_lookup.items():
        # cabinetcorners[corner_tuple[0]] = corner_by_id[corner_tuple[1]]
        cabinetcorners.append({'building_corner_id': corner_tuple[0], **corner_by_id[corner_tuple[1]]})

    streetcabinet_candidates_df = pd.DataFrame(cabinetcorners)
    streetcabinet_candidates_gdf = geopandas.GeoDataFrame(streetcabinet_candidates_df,
                                                          geometry=geopandas.points_from_xy(streetcabinet_candidates_df.x, streetcabinet_candidates_df.y))

    for key, building in houses_filter.iterrows():
        centroid = building['geometry'].centroid
        building_centroid_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0], 'street' : building['addr:street']}
        houses_list.append(building_centroid_node)

    houses_df = pd.DataFrame(houses_list)
    houses_dummy = pd.get_dummies(houses_df, columns=['street'])
    houses_dummy.iloc[:,2:] = houses_dummy.iloc[:,2:] / 1000

    house_clusters = int(round(len(houses_filter.index)/48, 0))
    # scaler = StandardScaler()
    kmeans = KMeansConstrained(n_clusters=house_clusters, size_min=None, size_max=48, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                      verbose=False, random_state=None, copy_x=True, n_jobs=1)
    kmeans.fit(houses_dummy)

    houses_centroids = []
    for i in range(len(kmeans.cluster_centers_)):
        houses_centroids.append({'x': kmeans.cluster_centers_[i][0], 'y': kmeans.cluster_centers_[i][1]})

    #TODO: add the cluster label as a column in the house geodataframe

    hs_centroids_df = pd.DataFrame(houses_centroids)
    hs_centroids_gdf = geopandas.GeoDataFrame(hs_centroids_df,
                                              geometry=geopandas.points_from_xy(hs_centroids_df.x, hs_centroids_df.y))

    houses_gdf = ckdnearest(streetcabinet_candidates_gdf, hs_centroids_gdf)

    streetcabinets_gdf = houses_gdf.iloc[:, [-3,-2]].drop_duplicates()

    plt.scatter(x=houses_dummy.x, y=houses_dummy.y, c=kmeans.labels_)
    plt.scatter(x=streetcabinets_gdf.x, y=streetcabinets_gdf.y, c='green')

    #TODO: connect houses and street cabinets to trench network, add column per row to add id for trenchCorners
    #TODO: houses dijkstra algorithm to streetcabinets




