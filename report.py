from typing import Tuple

import geopandas
from matplotlib.pyplot import plot
from pandas import DataFrame

from costs import DetailedCost

import osmnx as ox


class DetailedReport:
    def __init__(self, map_plot: plot, costs_table: DataFrame):
        self.plot = map_plot
        self.cost_table = costs_table


def get_detailed_report(detailed_cost: DetailedCost, building_gdf: geopandas.GeoDataFrame) -> DetailedReport:
    if detailed_cost.fiber_network.fiber_network is not None:
        g_box = detailed_cost.fiber_network.fiber_network
        # Give different things different colours
        ec = ['yellow' if 'highway' in d else
              'gray' if 'trench_crossing' in d else
              'blue' if 'trench' in d
              else 'red'
              for _, _, _, d in g_box.edges(keys=True, data=True)]

        # Plot the network
        fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                                node_size=0, edge_linewidth=0.5,
                                show=False, close=False)
        # Plot the buildings
        ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    return DetailedReport(None, None)
