from trenches2 import get_trench_network
import matplotlib.pyplot as plt

import osmnx as ox
import networkx as nx

G = nx.MultiDiGraph()

# Get graphs of different infrastructure types, then get trenches
g_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                           network_type='drive',
                           simplify=True,
                           retain_all=False,
                           truncate_by_edge=True)
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})
trench_network, road_network = get_trench_network(g_box, building_gdf)

# Add trench corner nodes to network
for intersection_osmid, corners in trench_network.trenchCorners.items():
    for corner in corners:
        g_box.add_node(**corner)

# Add the trenches to the network
osmid = 8945376
for trench in trench_network.trenches:
    osmid += 1
    g_box.add_edge(**trench, key=1, osmid=osmid)

ec = ['yellow' if 'highway' in d else
      "grey" if "trench_crossing" in d and d["trench_crossing"] else
      "blue" if "house_trench" in d else
      'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
# fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
#                         node_size=0, edge_linewidth=0.5,
#                         show=False, close=False)
# ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
# plt.show()

# nx.draw(G)
# # Set margins for the axes so that nodes aren't clipped
# ax = plt.gca()
# ax.margins(0.20)
# plt.axis("off")
# plt.show()

ox.plot_graph(g_box)
print(list(g_box.nodes(data=True))[1])
print(list(g_box.edges(data=True))[1]['geometry'])
