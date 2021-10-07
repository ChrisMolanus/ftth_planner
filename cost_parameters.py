import sys
import ruamel.yaml


class CostParameters:
    yaml_tag = u'!CostParameters'

    def __init__(self):
        self.dig_per_km: float = 1
        self.dig_per_road_crossing: float = 1
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

    def dump_to_file(self, file_path):
        yaml = ruamel.yaml.YAML()
        yaml.register_class(CostParameters)
        with open(file_path, 'w') as fp:
            yaml.dump(self, fp)


@staticmethod
def load_from_file(file_path: str) -> CostParameters:
    yaml = ruamel.yaml.YAML()
    yaml.register_class(CostParameters)
    with open(file_path) as fp:
        data = yaml.load(fp)
    return data



