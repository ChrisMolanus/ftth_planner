from typing import Dict

import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math

distance_from_center_of_road = 5 #
#roads_gdf = ox.geometries_from_place(place_name, tags={'highway': True})
G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=False)


def get_intersection_point(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # raise Exception('lines do not intersect')
        print('lines do not intersect')
        return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_trench_line(u, v, key, d, distance_from_center_of_road, last_line):
    """
    return: u, v, key, d
    """
    new_u_node = None
    new_v_node = None
    if 'geometry' in d:
        x = 0
        y = 0
        for sub_x, sub_y in d['geometry'].coords:
            new_u_node, new_v_node = getparellel_line_points(last_v, {'x': sub_x, 'y': sub_y},
                                                             distance_from_center_of_road)
            line = [(new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y'])]  # (new_u_node, new_v_node)
            if last_line is not None:
                x, y = get_intersection_point(line, last_line)
                if x is not None:
                    line = [(x, y), (new_v_node['x'], new_v_node['y'])]
            last_line = line
    else:
        new_u_node, new_v_node = getparellel_line_points(u, v, distance_from_center_of_road)
        new_edges.append((new_u_node, new_v_node))
    return new_u_node, new_v_node, 1, {**new_u_node, 'street_count': 1}, last_line


def getparellel_line_points(u_node, v_node, distance_from_center_of_road):
    dx = u_node['x'] - v_node['x']
    dy = u_node['y'] - v_node['y']

    road_length = math.sqrt(dx ** 2 + dy ** 2)
    if road_length == 0:
        road_length = 0.00001
    t = distance_from_center_of_road / road_length

    # Perpendicular line
    dx1 = -1 * dy
    dy1 = dx

    # Point distance_from_center_of_road form u_node on Perpendicular line from u_node
    xu1 = u_node['x'] + dx1
    yu1 = u_node['y'] + dy1
    xu1t = ((1 - t) * u_node['x'] + t * xu1)
    yu1t = ((1 - t) * u_node['y'] + t * yu1)

    # Point distance_from_center_of_road form v_node on Perpendicular line from v_node
    xv1 = v_node['x'] + dx1
    yv1 = v_node['y'] + dy1
    xv1t = ((1 - t) * v_node['x'] + t * xv1)
    yv1t = ((1 - t) * v_node['y'] + t * yv1)

    new_u_node = {'x': xu1t, 'y': yu1t}
    new_v_node = {'x': xv1t, 'y': yv1t}

    return new_u_node, new_v_node


distance_from_center_of_road = 0.0001
G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=False)
# Get point 5 meter from u point along line        u1
# Get point 5 meter from u point along other line  u2
# circumcenter for u, u1, u2


new_edges = list()
G_trenches = nx.Graph()
node_id = G_box.number_of_nodes()
last_line = None
for u, v, key, d in G_box.edges(keys=True, data=True):
    print(u)
    new_u_node, new_v_node, new_key, new_d, last_line = get_trench_line(G_box.nodes[u], G_box.nodes[v], key, d,
                                                                        distance_from_center_of_road, last_line)
    new_edges.append((new_u_node, new_v_node))

for new_u_node, new_v_node in new_edges:
    node_id += 1
    u = node_id
    node_id += 1
    v = node_id
    # G_trenches.add_node(u, **new_u_node, street_count=1)
    # G_trenches.add_node(v, **new_v_node, street_count=1)

    G_box.add_node(u, **new_u_node, street_count=1)
    G_box.add_node(v, **new_v_node, street_count=1)

    # print(f"u={u} v={v}")
    # G_trenches.add_edge(u, v, key=1, d={**new_u_node, 'street_count':1})
    G_box.add_edge(u, v, key=1, d={**new_u_node, 'street_count': 1})

ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G_box.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G_box, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)
plt.show()