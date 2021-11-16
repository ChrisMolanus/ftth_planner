import sys
from typing import Set, Dict, List, Any, Tuple

import pandas as pd

from cost_parameters import CostParameters
from fibers import FiberNetwork, CableType, EquipmentType, FiberCable, DecentralLocation, Splitter, StreetCabinet, ONT

# import ruamel.yaml
from trenches import TrenchType, TrenchCorner, Trench


class DetailedCostLine:
    def __init__(self, quantity: float, quantity_unit: str, total_cost: float):
        self.quantity_unit = quantity_unit
        self.total_cost = total_cost
        self.quantity = quantity


def get_cost_for_cable_installation(cable_type: CableType, length: float, cost_parameters: CostParameters):
    if cable_type == CableType.CoreToDS:
        return length * cost_parameters.fiber_ds_to_core_per_km
    elif cable_type == CableType.DSToSplitter96Cores:
        return length * cost_parameters.fiber_96core_per_km
    elif cable_type == CableType.SpliterToHouseDropCable:
        return length * cost_parameters.fiber_drop_pair_per_km

def get_cost_for_equipment(equipment_type: EquipmentType, quantity: int, cost_parameters: CostParameters):
    if equipment_type == EquipmentType.StreetCabinet:
        return quantity * cost_parameters.street_cabinet
    elif equipment_type == EquipmentType.Splitter:
        return quantity * cost_parameters.splitter
    elif equipment_type == EquipmentType.DecentralLocation:
        return quantity * cost_parameters.placement_of_ds
    elif equipment_type == EquipmentType.ONT:
        return quantity * cost_parameters.placement_of_ont


def get_cost_for_cable_material(cable_type: CableType, length: float, cost_parameters: CostParameters):
    if cable_type == CableType.CoreToDS:
        return length * cost_parameters.fiber_ds_to_core_per_km
    elif cable_type == CableType.DSToSplitter96Cores:
        return length * cost_parameters.fiber_96core_per_km
    elif cable_type == CableType.SplitterToHouseDropCable:
        return length * cost_parameters.fiber_drop_pair_per_km


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

    def get_materials_dataframe(self):
        table_rows: List[Dict[str, Any]] = list()

        for cable_type, detailed_costLine in self.fiber_cables_material.items():
            table_rows.append({"Type": str(cable_type.value),
                               "Quantity": detailed_costLine.quantity,
                               "Quantity units": detailed_costLine.quantity_unit,
                               "Total Cost": detailed_costLine.total_cost})

        for equipment_type, detailed_costLine in self.equipment_material.items():
            table_rows.append({"Type": str(equipment_type.value),
                               "Quantity": detailed_costLine.quantity,
                               "Quantity units": detailed_costLine.quantity_unit,
                               "Total Cost": detailed_costLine.total_cost})

        return pd.DataFrame(table_rows)

    def get_labor_dataframe(self):
        table_rows: List[Dict[str, Any]] = list()

        for trench_type, detailed_costLine in self.digging_labour.items():
            table_rows.append({"Type": str(trench_type.value),
                               "Quantity": detailed_costLine.quantity,
                               "Quantity units": detailed_costLine.quantity_unit,
                               "Total Cost": detailed_costLine.total_cost})

        for cable_type, detailed_costLine in self.fiber_cables_installation.items():
            table_rows.append({"Type": str(cable_type.value),
                               "Quantity": detailed_costLine.quantity,
                               "Quantity units": detailed_costLine.quantity_unit,
                               "Total Cost": detailed_costLine.total_cost})

        for equipment_type, detailed_costLine in self.equipment_installation.items():
            table_rows.append({"Type": str(equipment_type.value),
                               "Quantity": detailed_costLine.quantity,
                               "Quantity units": detailed_costLine.quantity_unit,
                               "Total Cost": detailed_costLine.total_cost})

        return pd.DataFrame(table_rows)


def get_costs(fiber_network: FiberNetwork, cost_parameters: CostParameters) -> DetailedCost:
    costs = DetailedCost(fiber_network, cost_parameters)
    # Collect the trenches that we will have to dig and the lengths of the cables we will use

    # Workaround for Bug where only some of the index have a key value,  maybe it's because fiber.py doesn't use all trenches
    mi = pd.MultiIndex.from_frame(fiber_network.trenches[["min_node_id", "max_node_id", "key"]])
    fiber_network.trenches.index = mi
    fiber_network.trenches.sort_index(inplace=True)

    used_trench_node_ids: Set[Tuple[int, int]] = set()
    for t in CableType:
        costs.fiber_cables_material[t] = DetailedCostLine(0.0, "km", 0.0)
        costs.fiber_cables_installation[t] = DetailedCostLine(0.0, "km", 0.0)
    for fibers in fiber_network.fibers.values():
        for fiber in fibers:
            used_trench_node_ids = used_trench_node_ids.union(set(fiber.trench_node_ids))

            costs.fiber_cables_material[fiber.cable_type].quantity += fiber.length
            costs.fiber_cables_installation[fiber.cable_type].quantity += fiber.length
    for t in costs.fiber_cables_material.keys():
        costs.fiber_cables_material[t].quantity = round(costs.fiber_cables_material[t].quantity / 1000, 2)
    for t in costs.fiber_cables_installation.keys():
        costs.fiber_cables_installation[t].quantity = round(costs.fiber_cables_installation[t].quantity / 1000, 2)

    # For the trenches that we have to dig get the lengths per Type since they have different costs
    for t in TrenchType:
        if t != TrenchType.Road_crossing:
            costs.digging_labour[t] = DetailedCostLine(0.0, "km", 0.0)
        else:
            costs.digging_labour[t] = DetailedCostLine(0.0, "crossings", 0.0)

    for trench_ids in used_trench_node_ids:
        trench: dict = fiber_network.trenches.loc[pd.IndexSlice[trench_ids]].to_dict(orient='records')[0]
        type = TrenchType.Road_crossing if trench["trench_crossing"] else TrenchType.Building if trench["house_trench"]\
            else TrenchType.Road_side
        costs.digging_labour[type].quantity += trench['length'] if type != TrenchType.Road_crossing else 1
    for t in costs.digging_labour.keys():
        if t != TrenchType.Road_crossing:
            costs.digging_labour[t].quantity = round(costs.digging_labour[t].quantity / 1000, 2)
        if t == TrenchType.Road_side:
            costs.digging_labour[t].total_cost = costs.digging_labour[
                                                     t].quantity * cost_parameters.dig_per_road_crossing
        elif t == TrenchType.Building:
            costs.digging_labour[t].total_cost = costs.digging_labour[
                                                     t].quantity * cost_parameters.dig_building_trench_per_km
        else:
            costs.digging_labour[t].total_cost = costs.digging_labour[
                                                     t].quantity * cost_parameters.dig_per_road_crossing

    # Account for the equipment
    for equipmentType, equipments in fiber_network.equipment.items():
        costs.equipment_material[equipmentType] = DetailedCostLine(len(equipments), "units", 0.0)
        costs.equipment_installation[equipmentType] = DetailedCostLine(len(equipments), "units", 0.0)

    # Calculate Cable costs
    for t in CableType:
        costs.fiber_cables_material[t].total_cost = get_cost_for_cable_material(cable_type=t,
            length=costs.fiber_cables_material[t].quantity, cost_parameters=cost_parameters)
        costs.fiber_cables_installation[t].total_cost = get_cost_for_cable_material(cable_type=t,
            length=costs.fiber_cables_installation[t].quantity, cost_parameters=cost_parameters)
        
    # Calculate Equipment costs
    for t in EquipmentType:
        if t in costs.equipment_material:
            costs.equipment_material[t].total_cost = get_cost_for_equipment(equipment_type=t,
                                                                        quantity=costs.equipment_material[
                                                                            t].quantity,
                                                                        cost_parameters=cost_parameters)
        if t in costs.equipment_installation:
            costs.equipment_installation[t].total_cost = get_cost_for_equipment(equipment_type=t,
                                                                        quantity=costs.equipment_installation[
                                                                            t].quantity,
                                                                        cost_parameters=cost_parameters)
    return costs

if __name__ == "__main__":
    fake_network = FiberNetwork()

    fake_network.fibers[CableType.CoreToDS] = list()
    fake_network.fibers[CableType.CoreToDS].append(
        FiberCable(trench_node_ids=[1, 2, 3], length=6, cable_type=CableType.CoreToDS))
    fake_network.fibers[CableType.CoreToDS].append(
        FiberCable(trench_node_ids=[1, 15, 16], length=6, cable_type=CableType.CoreToDS))


    fake_network.fibers[CableType.DSToSplitter96Cores] = list()
    fake_network.fibers[CableType.DSToSplitter96Cores].append(
        FiberCable(trench_node_ids=[16, 21, 22, 23, ], length=6, cable_type=CableType.DSToSplitter96Cores))
    fake_network.fibers[CableType.DSToSplitter96Cores].append(
        FiberCable(trench_node_ids=[16, 24, 25, 26,], length=6, cable_type=CableType.DSToSplitter96Cores))
    fake_network.fibers[CableType.DSToSplitter96Cores].append(
        FiberCable(trench_node_ids=[3, 31, 32, 33, ], length=6, cable_type=CableType.DSToSplitter96Cores))
    fake_network.fibers[CableType.DSToSplitter96Cores].append(
        FiberCable(trench_node_ids=[3, 34, 35, 36], length=6, cable_type=CableType.DSToSplitter96Cores))
    fake_network.equipment[EquipmentType.StreetCabinet] = [StreetCabinet(trench_corner=23, cabinet_id=1),
                                                           StreetCabinet(trench_corner=26, cabinet_id=2),
                                                           StreetCabinet(trench_corner=33, cabinet_id=3),
                                                           StreetCabinet(trench_corner=36, cabinet_id=4),]


    fake_network.equipment[EquipmentType.DecentralLocation] = [DecentralLocation(trench_corner=
                                                                                 TrenchCorner(trench_count=1,
                                                                                              u_node_id=100,
                                                                                              street_ids=set(),
                                                                                              node_for_adding=3,
                                                                                              x=1,
                                                                                              y=1),
                                                                                 street_cabinets=[fake_network.equipment[EquipmentType.StreetCabinet][0],
                                                                                                  fake_network.equipment[EquipmentType.StreetCabinet][1]]),
                                                               DecentralLocation(trench_corner=
                                                                                 TrenchCorner(trench_count=1,
                                                                                              u_node_id=200,
                                                                                              street_ids=set(),
                                                                                              node_for_adding=16,
                                                                                              x=1,
                                                                                              y=1),
                                                                                 street_cabinets=[fake_network.equipment[EquipmentType.StreetCabinet][2],
                                                                                                  fake_network.equipment[EquipmentType.StreetCabinet][3]])
                                                               ]

    fake_network.equipment[EquipmentType.Splitter] = [Splitter(street_cabinet=fake_network.equipment[EquipmentType.StreetCabinet][0]),
                                                      Splitter(street_cabinet=fake_network.equipment[EquipmentType.StreetCabinet][1]),
                                                      Splitter(street_cabinet=fake_network.equipment[EquipmentType.StreetCabinet][2]),
                                                      Splitter(street_cabinet=fake_network.equipment[EquipmentType.StreetCabinet][3]),]

    fake_network.fibers[CableType.SplitterToHouseDropCable] = list()
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[23,41, 42,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[23, 43, 44,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[26, 45, 46,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[26, 47, 48,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[33, 49, 50,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[33, 51, 52,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[36, 53, 54,], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.fibers[CableType.SplitterToHouseDropCable].append(
        FiberCable(trench_node_ids=[36, 55, 56], length=6, cable_type=CableType.SplitterToHouseDropCable))
    fake_network.equipment[EquipmentType.ONT] = [ONT(building_index="way (1)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][0]),
                                                 ONT(building_index="way (2)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][0]),
                                                 ONT(building_index="way (3)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][1]),
                                                 ONT(building_index="way (4)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][1]),
                                                 ONT(building_index="way (5)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][2]),
                                                 ONT(building_index="way (6)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][2]),
                                                 ONT(building_index="way (7)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][3]),
                                                 ONT(building_index="way (8)",
                                                     splitter=fake_network.equipment[EquipmentType.Splitter][3]),
                                                 ]
    trenches: List[Trench] = list()
    for c_type, fiber_cables in fake_network.fibers.items():
        for fiber_cable in fiber_cables:
            used_trenches_unsorted = list(zip(fiber_cable.trench_node_ids[::1], fiber_cable.trench_node_ids[1::1]))
            for pair in used_trenches_unsorted:
                trenches.append(Trench(u_for_edge=pair[0], v_for_edge=pair[1], name="", length=1.0, street_names=set()))

    fake_network.trenches = pd.DataFrame(trenches)
    fake_network.trenches["key"] = 1
    fake_network.trenches["min_node_id"] = fake_network.trenches[['u_for_edge', 'v_for_edge']].min(axis=1)
    fake_network.trenches["max_node_id"] = fake_network.trenches[['u_for_edge', 'v_for_edge']].max(axis=1)
    mi = pd.MultiIndex.from_frame(fake_network.trenches[["min_node_id", "max_node_id", "key"]])
    fake_network.trenches.index = mi

    cost_parameters = CostParameters()
    costs = get_costs(fake_network, cost_parameters)
    print(costs.get_materials_dataframe())
    print(costs.get_labor_dataframe())