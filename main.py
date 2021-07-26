import osmnx as ox
import networkx as nx


import matplotlib.pyplot as plt

place_name = "Hoeilaart, Belgium"


# get graphs of different infrastructure types, then combine
G1 = ox.graph_from_place(place_name, custom_filter='["highway"]', buffer_dist=200)
try:
    G2 = ox.graph_from_place(place_name, custom_filter='["railway"~"rail"]')
    G = nx.compose(G1, G2)
except ValueError:
    G = G1
# Get buildings
gdf = ox.geometries_from_place(place_name, tags={'building': True})

# plot highway edges in yellow, railway edges in red
ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G1, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)

# add building footprints in 50% opacity white
ox.plot_footprints(gdf, ax=ax, color="orange", alpha=0.5)
plt.show()



