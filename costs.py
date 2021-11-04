import sys
from typing import Set, Dict

from cost_parameters import CostParameters
from fibers import FiberNetwork, CableType


#import ruamel.yaml

class DetailedCost:
    def __init__(self, fiber_network: FiberNetwork, cost_parameters: CostParameters):
        self.fiber_network: FiberNetwork = fiber_network
        self.cost_parameters = cost_parameters


def get_cost_for_cable(self, cable_type: CableType, length: float):
    if cable_type == CableType.CoreToDS:
        return length * CostParameters.fiber_ds_to_core_per_km
    elif cable_type == CableType.DSToSplitter96Cores:
        return length * CostParameters.fiber_96core_per_km
    elif cable_type == CableType.SpliterToHouseDropCable:
        return length * CostParameters.fiber_drop_pair_per_km


def get_costs(fiber_network: FiberNetwork, cost_parameters: CostParameters) -> DetailedCost:
    # Collect the trenches that we will have to dig and the lengths of the cables we will use
    used_trench_ids: Set[int] = set()
    total_length_of_cables: Dict[CableType, float] = dict()
    for t in CableType:
        total_length_of_cables[t] = 0.0
    for fiber in fiber_network.fibers.values():
        used_trench_ids = used_trench_ids.union(set(fiber.trench_osmids))
        total_length_of_cables[fiber.cable_type] += fiber.length

    # For the trenches that we have to dig get the lengths per Type since they have different costs
    total_length_road_side_trenches = 0.0
    total_length_building_trenches = 0.0
    total_length_road_crossing_trenches = 0.0
    for trench_id in used_trench_ids:
        if fiber_network.trenches[trench_id]['trench_crossing']:
            total_length_road_crossing_trenches += fiber_network.trenches[trench_id]['length']
        elif fiber_network.trenches[trench_id]['house_trench']:
            total_length_building_trenches += fiber_network.trenches[trench_id]['length']
        else:
            total_length_road_side_trenches += fiber_network.trenches[trench_id]['length']

    return DetailedCost()
