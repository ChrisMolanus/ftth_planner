from typing import Dict

import osmnx as ox
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from shapely.geometry.linestring import LineString

distance_from_center_of_road = 5


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

    def __cmp__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']

    def __hash__(self):
        return hash((self['x'], self['y']))

    def __eq__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']


def get_trench_corners(G_box):
    trench_corners = dict()
    for u, current_none in G_box.nodes(data=True):
        print(current_none)
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
        for radian in sorted_vs:
            neighbor = G_box.nodes[v]
            v = neighbors[radian]
            streets = G_box.get_edge_data(u, v)
            if len(streets) > 1:
                print("Crap len(streets) > 1")
            street = streets[0]
            street_id = str(street['osmid'])
            print(street)
            if last_radian is not None:
                trench_corner_radian = radian - last_radian
                between_radian = radian + (radian - last_radian)
                x, y = point_on_circle(current_none, distance_from_center_of_road, between_radian)
                node = TrenchCorner(x, y, 2)
                if street_id not in trench_corners:
                    trench_corners[street_id] = set()
                trench_corners[street_id].add(node)
                trench_corners[last_street].add(node)
            else:
                first_radian = radian
                first_street = street_id
                trench_corners[street_id] = set()
            last_radian = radian
            last_v = v
            last_street = street_id

        between_radian = last_radian + (last_radian - first_radian)
        x, y = point_on_circle(current_none, distance_from_center_of_road, between_radian)
        node = TrenchCorner(x, y, 2)
        trench_corners[first_street].add(node)
        trench_corners[last_street].add(node)

        # TODO: add road cross trench in order of sorted_vs
    return trench_corners

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


def intersection_between_points(line1, line2):
    # TODO: use 'x' and 'y' instead of 0 and 1
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
    if isBetween(line1[0], line1[1], (x, y)):
        return True
    else:
        return False

G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=True)

trench_corners = get_trench_corners(G_box)

for u, v, key, street in G_box.edges(keys=True, data=True):
    print(street)
    street_id = str(street['osmid'])
    if street_id in trench_corners:
        corners = trench_corners[street_id]
        for point_pair in list(itertools.combinations(corners, 2)):
            if not intersection_between_points((u, v), point_pair):
                pass
                # TODO: Create trench line

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