import time

import matplotlib.pyplot as plt
import networkx
import osmnx as ox

from cost_parameters import CostParameters
from costs import get_costs
from fibers import get_fiber_network
from trenches import get_trench_network, get_trench_to_network_graph


def plot_network(g_box: networkx.MultiDiGraph, trench_network_graph: networkx.MultiDiGraph, building_gdf):
    ec = ['black' if 'highway' in d else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)

    ec = ["grey" if "trench_crossing" in d and d["trench_crossing"] else
          "blue" if "house_trench" in d else
          'red' for _, _, _, d in trench_network_graph.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(trench_network_graph, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False, ax=ax)

    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    plt.show()


start_time = time.time()

# Get graphs of different infrastructure types, then get trenches
box = (50.78694, 50.77902, 4.48386, 4.49521)
g_box = ox.graph_from_bbox(*box,
                           network_type='drive',
                           simplify=False,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
trench_network = get_trench_network(g_box, building_gdf)

trench_network_graph = get_trench_to_network_graph(trench_network, g_box)
plot_network(g_box, trench_network_graph, building_gdf)

cost_parameters = CostParameters()
fiber_network, fig = get_fiber_network(trench_network, cost_parameters, building_gdf, g_box)

detailed_cost = get_costs(fiber_network, cost_parameters)
print(detailed_cost.get_materials_dataframe())
print(detailed_cost.get_labor_dataframe())

print("--- The job took %s seconds ---" % (time.time() - start_time))

# TODO: convert detailed_report to PDF
