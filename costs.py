import sys
from typing import Set, Dict

from cost_parameters import CostParameters
from fibers import FiberNetwork, CableType, EquipmentType

# import ruamel.yaml
from trenches2 import TrenchType


class DetailedCostLine:
    def __init__(self, quantity: float, quantity_unit: str, total_cost: float):
        self.quantity_unit = quantity_unit
        self.total_cost = total_cost
        self.quantity = quantity


def get_cost_for_cable_installation(cable_type: CableType, length: float):
    if cable_type == CableType.CoreToDS:
        return length * CostParameters.fiber_ds_to_core_per_km
    elif cable_type == CableType.DSToSplitter96Cores:
        return length * CostParameters.fiber_96core_per_km
    elif cable_type == CableType.SpliterToHouseDropCable:
        return length * CostParameters.fiber_drop_pair_per_km


def get_cost_for_cable_material(self, cable_type: CableType, length: float):
    if cable_type == CableType.CoreToDS:
        return length * CostParameters.fiber_ds_to_core_per_km
    elif cable_type == CableType.DSToSplitter96Cores:
        return length * CostParameters.fiber_96core_per_km
    elif cable_type == CableType.SpliterToHouseDropCable:
        return length * CostParameters.fiber_drop_pair_per_km


class DetailedCost:
    def __init__(self, fiber_network: FiberNetwork, cost_parameters: CostParameters):
        self.cost_parameters = cost_parameters
        self.fiber_network: FiberNetwork = fiber_network

        # Materials
        # # Fiber costs
        self.fiber_cables_material: Dict[CableType, DetailedCostLine] = dict()
        # # Other
        self.equipment_material: Dict[EquipmentType, DetailedCostLine] = dict()

        # Labor costs
        # # Digging costs
        self.digging_labour: Dict[TrenchType, DetailedCostLine] = dict()
        # # Installation costs
        self.fiber_cables_installation: Dict[CableType, DetailedCostLine] = dict()
        self.equipment_installation: Dict[EquipmentType, DetailedCostLine] = dict()


def get_costs(fiber_network: FiberNetwork, cost_parameters: CostParameters) -> DetailedCost:
    costs = DetailedCost(fiber_network, cost_parameters)
    # Collect the trenches that we will have to dig and the lengths of the cables we will use
    used_trench_ids: Set[int] = set()
    for t in CableType:
        costs.fiber_cables_material[t] = DetailedCostLine(0.0, "km", 0.0)
        costs.fiber_cables_installation[t] = DetailedCostLine(0.0, "km", 0.0)
    for fiber in fiber_network.fibers.values():
        used_trench_ids = used_trench_ids.union(set(fiber.trench_osmids))
        costs.fiber_cables_material[fiber.cable_type].quantity += fiber.length
        costs.fiber_cables_installation[fiber.cable_type].quantity += fiber.length

    # For the trenches that we have to dig get the lengths per Type since they have different costs
    for t in TrenchType:
        costs.digging_labour[t] = DetailedCostLine(0.0, "km", 0.0)
    for trench_id in used_trench_ids:
        trench = fiber_network.trenches[trench_id]
        costs.digging_labour[trench.type].quantity += trench['length']

    # Account for the equipment
    for quantity, equipmentType in fiber_network.equipment:
        costs.equipment_material[equipmentType] = DetailedCostLine(quantity, "units", 0.0)
        costs.equipment_installation[equipmentType] = DetailedCostLine(quantity, "units", 0.0)

    # Calculate costs
    for t in CableType:
        costs.fiber_cables_material[t].total_cost = get_cost_for_cable_material(
            t, costs.fiber_cables_material[t].quantity)
        costs.fiber_cables_installation[t].total_cost = get_cost_for_cable_material(
            t, costs.fiber_cables_installation[t].quantity)

    return costs
