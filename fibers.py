from enum import Enum
from typing import List, Dict

import networkx

from costs import CostParameters
from trenches2 import TrenchNetwork


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


if __name__ == "__main__":
    import pickle
    trench_network = pickle.load(open("trench_network.p", "rb"))
    cost_parameters = CostParameters()
    get_fiber_network(get_fiber_network, cost_parameters)

    import pandas as pd
    import sklearn
    import numpy as np
    from sklearn.cluster import KMeans

    building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
    houses_centroids = []
    for building in building_gdf.iterrows():
        centroid = building['geometry'].centroid
        building_centroid_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0]}
        houses_centroids.append(building_centroid_node)
    houses_array = np.array(houses_centroids)

    house_clusters = int(round(len(houses_array)/48, 0))

    kmeans = KMeans(n_clusters=house_clusters, random_state=42).fit(houses_array)
