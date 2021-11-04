import sys


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



@staticmethod
def load_from_file(file_path: str) -> CostParameters:
    yaml = ruamel.yaml.YAML()
    yaml.register_class(CostParameters)
    return yaml.load(file_path)