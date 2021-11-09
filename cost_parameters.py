import sys
from typing import List, Dict, Any


class CostParameters:
    yaml_tag = u'!CostParameters'

    def __init__(self):
        self.dig_road_side_trench_per_km: float = 1
        self.dig_per_road_crossing: float = 1
        self.dig_building_trench_per_km: float = 1
        self.fiber_drop_pair_per_km: float = 1
        self.fiber_96core_per_km: float = 1
        self.fiber_install_per_km: float = 1
        self.fiber_ds_to_core_per_km: float = 1
        self.placement_of_street_cabinet: float = 1
        self.street_cabinet: float = 1
        self.placement_of_ds: float = 1
        self.ds: float = 1
        self.placement_of_ont: float = 1
        self.ont: float = 1

    def dump_to_file(self):
        yaml = ruamel.yaml.YAML()
        yaml.register_class(CostParameters)
        yaml.dump(self, sys.stdout)

    def get_cost_dataframe(self):
        table_rows: List[Dict[str, Any]] = list()
        table_rows.append({"Parameter": "Digging road-side trench per km", "value": self.dig_road_side_trench_per_km})
        table_rows.append({"Parameter": "Digging a road-crossing", "value": self.dig_per_road_crossing})
        table_rows.append({"Parameter": "Digging building-trench per km", "value": self.dig_building_trench_per_km})
        table_rows.append({"Parameter": "Drop cable per km", "value": self.fiber_drop_pair_per_km})
        table_rows.append({"Parameter": "96core Fiber per km", "value": self.fiber_96core_per_km})
        table_rows.append({"Parameter": "Fiber Installation cost per km", "value": self.fiber_install_per_km})
        table_rows.append({"Parameter": "Decentral central to core fiber per km", "value": self.fiber_ds_to_core_per_km})
        table_rows.append({"Parameter": "Placement cost of street cabinet", "value": self.placement_of_street_cabinet})
        table_rows.append({"Parameter": "Material cost of a street cabinet", "value": self.street_cabinet})
        table_rows.append({"Parameter": "Construction cost of a decentral central building", "value": self.placement_of_ds})
        table_rows.append({"Parameter": "Material cost of a decentral central building", "value": self.ds})
        table_rows.append({"Parameter": "Placement of ONT", "value": self.placement_of_ont})
        table_rows.append({"Parameter": "List price for a ONT", "value": self.ont})





@staticmethod
def load_from_file(file_path: str) -> CostParameters:
    yaml = ruamel.yaml.YAML()
    yaml.register_class(CostParameters)
    return yaml.load(file_path)