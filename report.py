from typing import Tuple

import geopandas
import osmnx as ox
import networkx
from matplotlib.pyplot import plot
from pandas import DataFrame

from costs import DetailedCost


class DetailedReport:
    def __init__(self, map_plot: plot, costs_table: DataFrame):
        self.plot = map_plot
        self.cost_table = costs_table


def get_detailed_report(detailed_cost: DetailedCost, trench_network: networkx.MultiDiGraph,
                        building_gdf: geopandas.GeoDataFrame) -> DetailedReport:
    # Give different things different colours
    ec = ['yellow' if 'highway' in d else
          'gray' if 'trench_crossing' in d else
          'blue' if 'trench' in d
          else 'red'
          for _, _, _, d in trench_network.edges(keys=True, data=True)]

    # Plot the network
    fig, ax = ox.plot_graph(trench_network, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    # Plot the buildings
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    return None, None
