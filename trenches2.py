from typing import Dict

import osmnx as ox
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from shapely.geometry.linestring import LineString

distance_from_center_of_road = 0.0001


def point_on_circle(center, radius, radian):
    angle = math.degrees(radian)
    x = center['x'] + (radius * math.cos(angle))
    y = center['y'] + (radius * math.sin(angle))

    return x, y


class TrenchCorner(dict):
    def __init__(self, x, y, trench_count, *args, **kw):
        super(TrenchCorner, self).__init__(*args, **kw)
        self['x'] = x
        self['y'] = y
        self['trench_count'] = trench_count
        self['street_count'] = 1

    def __cmp__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']

    def __hash__(self):
        return hash((self['x'], self['y']))

    def __eq__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']


def get_trench_corners(G_box):
    nodes = dict()
    trench_corners = dict()
    road_crossing = dict()
    #node_id = 362555986
    node_id = 400000000#G_box.number_of_nodes()
    for u, current_none in G_box.nodes(data=True):
        neighbors = dict()
        for v in G_box.neighbors(u):
            neighbor = G_box.nodes[v]
            radian = math.atan2(neighbor['y'] - current_none['y'], neighbor['x'] - current_none['x'])
            neighbors[radian] = v

        sorted_vs = list(neighbors.keys())
        sorted_vs.sort()

        first_radian = None
        first_street = None
        last_radian = None
        last_v = None
        last_street = None
        last_node_id = None
        first_node_id = None
        for radian in sorted_vs:
            v = neighbors[radian]
            streets = G_box.get_edge_data(u, v)
            if len(streets) > 1:
                print("Crap len(streets) > 1")
            else:
                print(streets[0])
            street = streets[0]
            street_id = str(street['osmid'])
            if last_radian is not None:
                between_radian = radian + (radian - last_radian)
                x, y = point_on_circle(current_none, distance_from_center_of_road, between_radian)
                node = TrenchCorner(x, y, 2)
                if street_id not in trench_corners:
                    trench_corners[street_id] = set()
                if node not in trench_corners[first_street] and node not in trench_corners[last_street]:
                    node_id += 1
                    node['node_for_adding'] = node_id
                    trench_corners[street_id].add(node)
                    trench_corners[last_street].add(node)
                    nodes[node.__hash__()] = node
                else:
                    node_id = nodes[node.__hash__()]['node_for_adding']

                if last_node_id is not None:
                    road_crossing[last_street] = (last_node_id, node_id)
                    if last_node_id == 400000011 or node_id == 400000011:
                        print(f"In loop {last_node_id} {node_id}")
                else:
                    first_node_id = node_id
                last_node_id = node_id
            else:
                first_radian = radian
                first_street = street_id
                if street_id not in trench_corners:
                    trench_corners[street_id] = set()
            last_radian = radian
            last_v = v
            last_street = street_id

        if len(sorted_vs) > 1:
            between_radian = last_radian + (last_radian - first_radian)
            x, y = point_on_circle(current_none, distance_from_center_of_road, between_radian)
            node = TrenchCorner(x, y, 2)
            if node not in trench_corners[first_street] and node not in trench_corners[last_street]:
                node_id += 1
                node['node_for_adding'] = node_id
                trench_corners[first_street].add(node)
                trench_corners[last_street].add(node)
                road_crossing[last_street] = (node_id, first_node_id)
                if last_node_id == 400000011 or node_id == 400000011:
                    print(f"Out of loop {first_node_id} {node_id}")




        # TODO: add road cross trench in order of sorted_vs
    return trench_corners, road_crossing

def isBetween(a, b, c):
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    # # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 0.00000005:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True


def intersection_between_points(l1, l2):
    # TODO: use 'x' and 'y' instead of 0 and 1
    line1 = (l1[0]['x'], l1[0]['y']),(l1[1]['x'], l1[1]['y'])
    line2 = (l2[0]['x'], l2[0]['y']), (l2[1]['x'], l2[1]['y'])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        # raise Exception('lines do not intersect')
        print('lines do not intersect')
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    if isBetween((line1[0][0], line1[0][1]),
                 (line1[1][0], line1[1][1]),
                 (x, y)):
        return True
    else:
        return False

G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=True)

trench_corners, road_crossing = get_trench_corners(G_box)

for osmid, corners in trench_corners.items():
    for corner in corners:
        G_box.add_node(**corner)
        print(f"Corner {corner}")

for u, current_none in G_box.nodes(data=True):
    print(f"{u} {current_none}")

new_edges = list()
for u, v, key, street in G_box.edges(keys=True, data=True):
    print(f"{u} {v} {key} {street}")
    street_id = str(street['osmid'])
    if street_id in trench_corners:
        corners = trench_corners[street_id]
        for point_pair in list(itertools.combinations(corners, 2)):
            u_node = G_box.nodes[u]
            v_node = G_box.nodes[v]
            if not intersection_between_points((u_node, v_node), point_pair):
                new_edges.append({'u_for_edge': point_pair[0]['node_for_adding'],
                                  'v_for_edge': point_pair[1]['node_for_adding'],
                                  'key': 1, 'osmid': 8945376,
                                  'oneway': False,
                                  'name': f"trench {street_id}",
                                  'length': 225.493})
    else:
        print(f"Street {street} not in trench_corners")

for street_id, crossing in road_crossing. items():
    new_edges.append({'u_for_edge': crossing[0],
                      'v_for_edge': crossing[1],
                      'key': 1, 'osmid': 8945376,
                      'oneway': False,
                      'name': f"trench {street_id}",
                      'length': 225.493})

for edge in new_edges:
    G_box.add_edge(**edge)

# TODO: Add corners as node
# TODO: Add all trench lines as edges

# Get buildings
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721,  tags={'building': True})

# Get Building centroids
building_centroids = list()
for _, building in building_gdf.iterrows():
    centroid = building['geometry'].centroid
    building_centroids.append([centroid.xy[0][0], centroid.xy[1][0]])



ec = ['y' if 'highway' in d else 'r' for _, _, _, d in G_box.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G_box, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)
ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
plt.show()