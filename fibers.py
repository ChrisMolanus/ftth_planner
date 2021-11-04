from enum import Enum
from typing import List, Dict

import networkx

from costs import CostParameters
from trenches2 import TrenchNetwork, Trench


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
        """
        A Fiber cable
        :param trench_osmids: The ordered list (from root to leaf) of the trench ids that this fiber will be placed in
        :param length: The total length of the cable, no buffer
        :param cable_type: The cable type
        """
        self.trench_osmids = trench_osmids
        self.length = length
        self.cable_type = cable_type


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

        # Trenches where osm id is the key and the trench is the value
        self.trenches: Dict[int, Trench] = None


def get_fiber_network(trench_network: TrenchNetwork, cost_parameters: CostParameters) -> FiberNetwork:
    return FiberNetwork()


if __name__ == "__main__":
    import pickle
    trench_network = pickle.load(open("trench_network.p", "rb"))
    cost_parameters = CostParameters()
    get_fiber_network(get_fiber_network, cost_parameters)
