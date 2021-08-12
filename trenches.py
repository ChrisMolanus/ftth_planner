from typing import Dict

import osmnx as ox
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from shapely.geometry.linestring import LineString

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
        raise Exception('lines do not intersect')
        # print('lines do not intersect')
        # return None, None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_trench_line(u, v, key, d, distance_from_center_of_road, osmid):
    """
    return: u, v, key, d
    """
    new_d = {'osmid': osmid, 'name': f"{d['name']} Trench", 'trench': "roadside", 'oneway': d['oneway'], 'length': d['length']}
    linestring = list()
    if 'geometry' in d:
        x = 0
        y = 0
        last_road_point = None
        last_trench_point = None
        last_line = None
        for sub_x, sub_y in d['geometry'].coords:
            if last_road_point is not None:
                new_u_node, new_v_node = getparellel_line_points({'x': last_road_point[0], 'y': last_road_point[1], 'street_count': 1},
                                                                 {'x': sub_x, 'y': sub_y, 'street_count': 1},
                                                                 distance_from_center_of_road)

                if last_line is not None:
                    # Not first line
                    x, y = get_intersection_point(last_line,
                                                  [(new_u_node['x'], new_u_node['y']),
                                                   (new_v_node['x'], new_v_node['y'])])
                    x = round(x, 7)
                    y = round(y, 7)
                    new_u_node = {'x': x, 'y': y, 'street_count': 1}
                else:
                    # Fist line
                    x = new_u_node['x']
                    y = new_u_node['y']

                linestring.append((new_u_node['x'], new_u_node['y']))
                line = [(new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y'])]
                last_line = line

                dx = x - new_v_node['x']
                dy = y - new_v_node['y']
                road_length = math.sqrt(dx ** 2 + dy ** 2)
                new_d['length'] += road_length

                last_trench_point = (new_v_node['x'], new_v_node['y'])

            last_road_point = (sub_x, sub_y)

        linestring.append(last_trench_point)

        new_d['geometry'] = LineString(linestring)
        new_u_node = {'x': linestring[0][0], 'y': linestring[0][1], 'street_count': u['street_count']}
        new_v_node = {'x': last_trench_point[0], 'y': last_trench_point[1], 'street_count': v['street_count']}

    else:
        new_u_node, new_v_node = getparellel_line_points(u, v, distance_from_center_of_road)

        dx = new_u_node['x'] - new_v_node['x']
        dy = new_u_node['y'] - new_v_node['y']
        road_length = math.sqrt(dx ** 2 + dy ** 2)
        new_d['length'] = road_length

    return new_u_node, new_v_node, key, new_d


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
    xu1t = round((1 - t) * u_node['x'] + t * xu1, 7)
    yu1t = round((1 - t) * u_node['y'] + t * yu1, 7)

    # Point distance_from_center_of_road form v_node on Perpendicular line from v_node
    xv1 = v_node['x'] + dx1
    yv1 = v_node['y'] + dy1
    xv1t = round((1 - t) * v_node['x'] + t * xv1, 7)
    yv1t = round((1 - t) * v_node['y'] + t * yv1, 7)

    new_u_node = {'x': xu1t, 'y': yu1t, 'street_count': u_node['street_count']}
    new_v_node = {'x': xv1t, 'y': yv1t, 'street_count': v_node['street_count']}

    return new_u_node, new_v_node


def point_distance(node1, point):
    return ((((point[0] - node1['x']) ** 2) + ((point[1] - node1['y']) ** 2)) ** 0.5)


def fix_intersecting_trenches(trench1, trench2):
    trench1_u_node = trench1[0]
    trench1_v_node = trench1[1]

    trench2_u_node = trench2[0]
    trench2_v_node = trench2[1]
    try:
        x, y = get_intersection_point([(trench1_u_node['x'], trench1_u_node['y']),
                                       (trench1_v_node['x'], trench1_v_node['y'])],
                                      [(trench2_u_node['x'], trench2_u_node['y']),
                                       (trench2_v_node['x'], trench2_v_node['y'])])
        if point_distance(trench1_u_node, (x, y)) < point_distance(trench1_v_node, (x, y)):
            trench1_u_node['x'] = x
            trench1_u_node['y'] = y
        else:
            trench1_v_node['x'] = x
            trench1_v_node['y'] = y

        if point_distance(trench2_u_node, (x, y)) < point_distance(trench2_v_node, (x, y)):
            trench2_u_node['x'] = x
            trench2_u_node['y'] = y
        else:
            trench2_v_node['x'] = x
            trench2_v_node['y'] = y

        return True
    except Exception:
        return False

def get_edges(g_box, distance_from_center_of_road = 0.0001):
    new_edges = dict()
    osmid = 8945376
    road_node_to_trench_nodes = dict()
    node_as_u = dict()
    node_as_v = dict()
    last_d = dict()
    for u, v, key, d in G_box.edges(keys=True, data=True):
        line_key = "_".join([str(u), str(v)])
        road_node_to_trench_nodes[line_key] = {'u': u, 'v': v}
        if u not in node_as_u:
            node_as_u[u] = list()
        node_as_u[u].append(line_key)
        if v not in node_as_v:
            node_as_v[v] = list()
        node_as_v[v].append(line_key)



        new_u_node, new_v_node, new_key, new_d = get_trench_line(g_box.nodes[u], g_box.nodes[v], key, d,
                                                                 distance_from_center_of_road, osmid)
        road_node_to_trench_nodes[line_key]['trench_u'] = new_u_node
        road_node_to_trench_nodes[line_key]['trench_v'] = new_v_node

        new_edges[line_key] = (new_u_node, new_v_node, new_key, new_d)
        road_node_to_trench_nodes[line_key]['trench'] = new_edges[line_key]
        d1 = new_d.copy()
        if 'geometry' in d1:
            del d1['geometry']
        last_d[u] = d1
        last_d[v] = d1
        osmid += 1

    for node_id, node in g_box.nodes.items():
        for v_line_keys in node_as_v[node_id]:
            v_trench = road_node_to_trench_nodes[v_line_keys]
            for u_line_keys in node_as_u[node_id]:
                u_trench = road_node_to_trench_nodes[u_line_keys]
                if u_trench['v'] != v_trench['u']:
                    if not fix_intersecting_trenches(u_trench['trench'], v_trench['trench']):
                        new_edges.append((v_trench['trench_v'], u_trench['trench_u'], 1, last_d[node_id]))


    return list(new_edges.values())





G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=False)
# Get point 5 meter from u point along line        u1
# Get point 5 meter from u point along other line  u2

new_edges = get_edges(G_box)

node_id = G_box.number_of_nodes()
for new_u_node, new_v_node, new_key, new_d in new_edges:
    node_id += 1
    u = node_id
    node_id += 1
    v = node_id

    G_box.add_node(u, **new_u_node)
    G_box.add_node(v, **new_v_node)

    G_box.add_edge(u, v, key=new_key, **new_d)

ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G_box.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G_box, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)
plt.show()