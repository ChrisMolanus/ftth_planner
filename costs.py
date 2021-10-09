import sys

from cost_parameters import CostParameters
from fibers import FiberNetwork
#import ruamel.yaml

class DetailedCost:
    def __init__(self, fiber_network: FiberNetwork, cost_parameters: CostParameters):
        self.fiber_network: FiberNetwork = fiber_network
        self.cost_parameters = cost_parameters


def get_costs(fiber_network: FiberNetwork, cost_parameters: CostParameters) -> DetailedCost:
        return DetailedCost()
