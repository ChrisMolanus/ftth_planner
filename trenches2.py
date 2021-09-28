from typing import Dict, List, Tuple

import osmnx as ox
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math
from shapely.geometry.linestring import LineString

distance_from_center_of_road = 0.0001

def node_distance(node1, node2):
    return ((((node2['x'] - node1['x']) ** 2) + ((node2['y'] - node1['y']) ** 2)) ** 0.5)

def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    if y2 < y1:
        return math.pi - math.acos(inner_product/(len1*len2)) + math.pi
    else:
        return math.acos(inner_product/(len1*len2))

# center = {'x':4.4861903, 'y':50.7819225}
# radius = 0.0001
# radian = 0.03847

def point_on_circle(center, radius, radian):
    x = center['x'] + (radius * math.cos(radian))
    y = center['y'] + (radius * math.sin(radian))
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
    for u, current_node in G_box.nodes(data=True):
        print(current_node)
        # if current_node['x'] == 4.4879159 and current_node['y'] == 50.7845063:
        #     print("Test crossing 1")
        neighbors = dict()
        for v in G_box.neighbors(u):
            neighbor = G_box.nodes[v]
            #radian = math.atan2(neighbor['y'] - current_node['y'], neighbor['x'] - current_node['x'])
            # radian between horizontal line though current_node and line though current_node and v
            street = G_box.get_edge_data(u, v)[0]
            if 'geometry' not in street:
                #print(f"neighbor:{neighbor}")
                #radian = angle((current_node['x'] + 1, current_node['y']), (neighbor['x']-current_node['x'], neighbor['y']-current_node['y']))
                radian = angle((1, 0), (neighbor['x']-current_node['x'], neighbor['y']-current_node['y']))
            else:
                # Street is not a simple line so we have to look at the geometry
                l: List[Tuple[float, float]] = list(street['geometry'].coords)
                if l[0] == (current_node['x'], current_node['y']):
                    # l[0] is the "u"
                    # l[1] is one point away from "u"
                    v1 = l[1]
                else:
                    # "u" is the last coordinate in l
                    # l[-2] is one point away from "u"
                    v1 = l[-2]
                #print(f"v1:{v1}")
                # radian = angle((current_node['x'] + 1, current_node['y']),
                #                (v1[0] - current_node['x'], v1[1] - current_node['y']))
                radian = angle((1, 0), (v1[0] - current_node['x'], v1[1] - current_node['y']))
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
            # if current_node['x'] == 4.4879159 and current_node['y'] == 50.7845063:
            #     print("Test crossing 2")
            v = neighbors[radian]
            streets = G_box.get_edge_data(u, v)
            if len(streets) > 1:
                print("Crap len(streets) > 1")
            # else:
            #     print(streets[0])
            street = streets[0]
            street_id = str(street['osmid'])
            #print(f"radian:{radian}")
            #print(f"last_radian:{last_radian}")

            if last_radian is not None:
                between_radian = radian - (abs(radian - last_radian) / 2)
                #print(f"between_radian:{between_radian}")
                x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
                #print((x, y))
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
                    if last_street not in road_crossing:
                        road_crossing[last_street] = list()
                    road_crossing[last_street].append((last_node_id, node_id))
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

        # if current_node['x'] == 4.4879159 and current_node['y'] == 50.7845063:
        #     print("Test crossing 3")
        if len(sorted_vs) > 1:
            first_radian = first_radian + (2 * math.pi)
            between_radian = first_radian - (abs(first_radian - last_radian) / 2)
            x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
            #print((x, y))
            node = TrenchCorner(x, y, 2)
            if node not in trench_corners[first_street] and node not in trench_corners[last_street]:
                node_id += 1
                node['node_for_adding'] = node_id
                trench_corners[first_street].add(node)
                trench_corners[last_street].add(node)
                if last_street not in road_crossing:
                    road_crossing[last_street] = list()
                road_crossing[last_street].append((last_node_id, node_id))
                if first_street not in road_crossing:
                    road_crossing[first_street] = list()
                road_crossing[first_street].append((node_id, first_node_id))


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

G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521, network_type='drive', simplify=True, retain_all=False)

trench_corners, road_crossing = get_trench_corners(G_box)

for osmid, corners in trench_corners.items():
    for corner in corners:
        # TODO: addes nodes more then ones
        #print(corner)
        G_box.add_node(**corner)

# for u, current_none in G_box.nodes(data=True):
#     print(f"{u} {current_none}")

new_edges = list()
new_pp = list()
point_edges = dict()
processed_streets = set()
for u, v, key, street in G_box.edges(keys=True, data=True):
    added_trenches = set()
    print(f"{u} {v} {key} {street}")
    street_id = str(street['osmid'])
    if street_id not in processed_streets:
        processed_streets.add(street_id)
        if 'geometry' not in street or True:
            if street_id in trench_corners:
                corners = trench_corners[street_id]
                for point_pair1 in list(itertools.combinations(corners, 2)):
                    point_pair = [point_pair1[0], point_pair1[1]] # because tuples are immutable
                    u_node = G_box.nodes[u]
                    v_node = G_box.nodes[v]
                    xs = [point_pair[0]['x'], point_pair[1]['x']]
                    ys = [point_pair[0]['y'], point_pair[1]['y']]
                    # xs.sort()
                    # ys.sort()
                    h1 = hash((xs[0], xs[1], ys[0], ys[1]))
                    h2 = hash((xs[1], xs[0], ys[1], ys[0]))
                    if not intersection_between_points((u_node, v_node), point_pair) \
                            and h1 not in added_trenches\
                            and h2 not in added_trenches:
                        added_trenches.add(h1)
                        added_trenches.add(h2)
                        if point_pair[0]["node_for_adding"] not in point_edges:
                            point_edges[point_pair[0]["node_for_adding"]] = list()
                        point_edges[point_pair[0]["node_for_adding"]].append(point_pair)
                        if point_pair[1]["node_for_adding"] not in point_edges:
                            point_edges[point_pair[1]["node_for_adding"]] = list()
                        point_edges[point_pair[1]["node_for_adding"]].append(point_pair)
                        new_pp.append(point_pair)

for node_id, point_pairs in point_edges.items():
    if len(point_pairs) > 1:
        shortest_pair = None
        shortest_distance = 1000000
        for i in range(1, len(point_pairs)):
            if point_pairs[i][0] is not None:
                new_dist = node_distance(*point_pairs[i])
                if new_dist < shortest_distance:
                    if shortest_pair is not None:
                        shortest_pair[0] = None # Mark trench to not be used
                        shortest_pair[1] = None
                    shortest_pair = point_pairs[i]
                    shortest_distance = new_dist

for point_pair in new_pp:
    if point_pair[0] is not None:
        new_edges.append({'u_for_edge': point_pair[0]['node_for_adding'],
                          'v_for_edge': point_pair[1]['node_for_adding'],
                          'key': 1, 'osmid': 8945376,
                          'oneway': False,
                          'name': f"trench {street_id}",
                          'length': 225.493})
    # else:
    #     print("Removed")

for street_id, crossings in road_crossing. items():
    for crossing in crossings:
        new_edges.append({'u_for_edge': crossing[0],
                          'v_for_edge': crossing[1],
                          'key': 1, 'osmid': 8945376,
                          'oneway': False,
                          'name': f"trench {street_id}",
                          'length': 225.493,
                          'trench crossing': True})

for edge in new_edges:
    G_box.add_edge(**edge)

# Get buildings
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721,  tags={'building': True})

# Get Building centroids
building_centroids = list()
for _, building in building_gdf.iterrows():
    centroid = building['geometry'].centroid
    building_centroids.append([centroid.xy[0][0], centroid.xy[1][0]])



ec = ['y' if 'highway' in d else 'gray' if 'trench crossing' in d else 'r' for _, _, _, d in G_box.edges(keys=True, data=True)]
fig, ax = ox.plot_graph(G_box, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)
ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
plt.show()