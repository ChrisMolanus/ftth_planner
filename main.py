from typing import Dict

import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import random as rd

from k_means_constrained import KMeansConstrained
from sklearn.neighbors import KDTree
place_name = "Hoeilaart, Belgium"


# Get graphs of different infrastructure types, then combine
G1 = ox.graph_from_place(place_name, custom_filter='["highway"]', buffer_dist=200)
try:
    G2 = ox.graph_from_place(place_name, custom_filter='["railway"~"rail"]')
    G = nx.compose(G1, G2)
except ValueError:
    G = G1

# Get buildings
building_gdf = ox.geometries_from_place(place_name, tags={'building': True})

# Get Building centroids
building_centroids = list()
for _, building in building_gdf.iterrows():
    centroid = building['geometry'].centroid
    building_centroids.append([centroid.xy[0][0], centroid.xy[1][0]])

# Get max distance of nearest building to get a feel
tree = KDTree(building_centroids, leaf_size=1000)
dist, ind = tree.query(building_centroids[:50], k=3)

# Get Building centroids with street dimension
building_centroids = list()
street_counter: Dict[str, float] = dict()
for _, building in building_gdf.iterrows():
    centroid = building['geometry'].centroid
    street = building["addr:street"]
    if street not in street_counter:
        street_counter[street] = len(street_counter.keys()) * dist.max() * 2
    street_counter[street]
    building_centroids.append([centroid.xy[0][0], centroid.xy[1][0], street_counter[street]])

# Form clusters of buildings
clf = KMeansConstrained(n_clusters=77, size_min=50, size_max=60, random_state=0)
clf.fit_predict(building_centroids)
building_clsuer_centers = clf.cluster_centers_
builing_clsuer_lables = clf.labels_

# Give each building cluster a color
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, len(np.unique(np.unique(builing_clsuer_lables)))))
label_colors = list()
for label in builing_clsuer_lables:
    label_colors.append(colors[label])

# Street DataFrame
street_gdf = ox.graph_to_gdfs(G, nodes=False)
for _, edge in street_gdf.fillna('').iterrows():
    print(edge['name'])

# plot highway edges in yellow, railway edges in red
ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G1, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)

# add building footprints in 50% opacity with cluster colors
ox.plot_footprints(building_gdf, ax=ax, color=label_colors, alpha=0.5)
plt.show()


