import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry import LineString

ox.config(log_console=True, use_cache=True)

# Create map
map_center = (13.743942, 100.570006)

G = ox.graph_from_point(map_center, network_type='drive', simplify=True)

# original point lat/long
pt_latl = (13.744001, 100.570457)

# get nearest node
pt_nearest_node_euc = ox.get_nearest_node(G, pt_latl, method='euclidean')
pt_nearest_node_har = ox.get_nearest_node(G, pt_latl, method='haversine')

# --start-- copied from https://gist.github.com/rochacbruno/2883505
# Calculate distance between latitude longitude pairs with Python

# Haversine formula example in Python
# Author: Wayne Dyck

import math


def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


# --end-- copied from https://gist.github.com/rochacbruno/2883505

# obtain nearest node along nearest edge

# obtain nearest edge for original point
nearest_edge = ox.get_nearest_edge(G, pt_latl)
pt_nearest_edge = nearest_edge[1:3]  # [1], [2] are start/end nodes of the nearest edge

# project points onto neareset edge (a,b)
o_point = Point(pt_latl[1], pt_latl[0])
a_point = Point(G.nodes[pt_nearest_edge[0]]['x'], G.nodes[pt_nearest_edge[0]]['y'])
b_point = Point(G.nodes[pt_nearest_edge[1]]['x'], G.nodes[pt_nearest_edge[1]]['y'])
a_latl = (G.nodes[pt_nearest_edge[0]]['y'], G.nodes[pt_nearest_edge[0]]['x'])
b_latl = (G.nodes[pt_nearest_edge[1]]['y'], G.nodes[pt_nearest_edge[1]]['x'])
dist_ab = LineString([a_point, b_point]).project(o_point)
projected_orig_point = list(LineString([a_point, b_point]).interpolate(dist_ab).coords)
o1_latl = (projected_orig_point[0][1], projected_orig_point[0][0])

# calculate distance o1->a and o1->b
o1a = round(distance(o1_latl, a_latl) * 1000, 0)
o1b = round(distance(o1_latl, b_latl) * 1000, 0)

# choose nearer node
if (o1a < o1b):
    nearest_node_along_edge = a_latl
else:
    nearest_node_along_edge = b_latl

# print(nearest_node_along_edge)

# plot graph
fig, ax = ox.plot_graph(G, fig_height=10, fig_width=7, node_size=50,
                        show=False, close=False)

# original point in red
ax.scatter(pt_latl[1], pt_latl[0], c='r', s=100)

# nearest node by euclidean distance in blue
ax.scatter(G.nodes[pt_nearest_node_euc]['x'], G.nodes[pt_nearest_node_euc]['y'], c='b', s=100)

# nodes of nearest edge in green and lime
ax.scatter(G.nodes[pt_nearest_edge[0]]['x'], G.nodes[pt_nearest_edge[0]]['y'], c='g', s=100)
ax.scatter(G.nodes[pt_nearest_edge[1]]['x'], G.nodes[spt_nearest_edge[1]]['y'], c='lime', s=200)

# origial point projected onto nearest edge in violet
ax.scatter(o1_latl[1], o1_latl[0], c='violet', s=100)

# nearest node along nearest edge in black
ax.scatter(nearest_node_along_edge[1], nearest_node_along_edge[0], c='black', s=50)

plt.show()