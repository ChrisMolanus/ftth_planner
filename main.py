import networkx
import osmnx as ox
import matplotlib.pyplot as plt

from cost_parameters import CostParameters
from costs import DetailedCost
from fibers import get_fiber_network
from report import get_detailed_report
from trenches2 import get_trench_network, add_trenches_to_network

def plot_network(g_box: networkx.MultiDiGraph):
    ec = ['black' if 'highway' in d else
          "grey" if "trench_crossing" in d and d["trench_crossing"]else
          "blue" if "house_trench" in d else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    plt.show()



# Get graphs of different infrastructure types, then get trenches
g_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                           network_type='drive',
                           simplify=False,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
trench_network = get_trench_network(g_box, building_gdf)
# import pickle
# pickle.dump(trench_network, open("trench_network.p", "wb"))

trench_network_graph = add_trenches_to_network(trench_network, g_box)
ox.plot_graph(trench_network_graph)

cost_parameters = CostParameters()
fiber_network = get_fiber_network(trench_network, cost_parameters)

detailed_cost = DetailedCost(fiber_network, cost_parameters)

detailed_report = get_detailed_report(detailed_cost, building_gdf)

if detailed_report.plot is not None:
    detailed_report.plot.show()

# TODO: convert detailed_report to PDF
