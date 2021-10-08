import osmnx as ox
import matplotlib.pyplot as plt

from cost_parameters import CostParameters
from costs import DetailedCost
from fibers import get_fiber_network
from report import get_detailed_report
from trenches2 import get_trench_network

# Get graphs of different infrastructure types, then get trenches
g_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                           network_type='drive',
                           simplify=True,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
trench_network, g_box = get_trench_network(g_box, building_gdf)

# TODO: make separate network for trenches since fiber planning only needs that network and not the roads

# Add trench corner nodes to network
for intersection_osmid, corners in trench_network.trenchCorners.items():
    for corner in corners:
        # TODO: addes nodes more then ones, but it should be ok since they have the same ID
        g_box.add_node(**corner)

# Add the trenches to the network
osmid = 8945376
for trench in trench_network.trenches:
    osmid += 1
    g_box.add_edge(**trench, key=1, osmid=osmid)

cost_parameters = CostParameters()
fiber_network = get_fiber_network(g_box, cost_parameters)

detailed_cost = DetailedCost(fiber_network, cost_parameters)

detailed_report = get_detailed_report(detailed_cost, g_box, building_gdf)

#detailed_report.plot.show()

# TODO: convert detailed_report to PDF
