from typing import Dict

import osmnx as ox
import networkx as nx
import itertools

import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math

from shapely.geometry import Point
from shapely.geometry.linestring import LineString

import shapely
from shapely.geometry import LineString, Point


distance_from_center_of_road = 5 #
#roads_gdf = ox.geometries_from_place(place_name, tags={'highway': True})
G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=False, retain_all=True)
#G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521, network_type='drive', simplify=True, retain_all=False)

def get_intersection_point2(line1, line2):
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


def get_intersection_point(line1, line2):
    # xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    # ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    #
    # def det(a, b):
    #     return a[0] * b[1] - a[1] * b[0]
    #
    # div = det(xdiff, ydiff)
    # if div == 0:
    #     raise Exception('lines do not intersect')
    #     # print('lines do not intersect')
    #     # return None, None
    #
    # d = (det(*line1), det(*line2))
    # x = det(d, xdiff) / div
    # y = det(d, ydiff) / div
    # return x, y
    line1 = LineString([line1[0], line1[1]])
    line2 = LineString([line2[0], line2[1]])

    int_pt = line1.intersection(line2)
    x = int_pt.x
    y = int_pt.y
    return x, y

def get_nearest_road(G, point):
    """
    Return the nearest road to a pair of coordinates.
    Pass in the graph representing the road network and a tuple with the
    coordinates. We first get all the roads in the graph. Secondly we
    compute the distance from the coordinates to the segments determined
    by each road. The last step is to sort the road segments in ascending
    order based on the distance from the coordinates to the road.
    In the end, the first element in the list of roads will be the closest
    road that we will return as a tuple containing the shapely geometry and
    the u, v nodes.
    Parameters
    ----------
    G : networkx multidigraph
    point : tuple
        The (lat, lng) or (y, x) point for which we will find the nearest node
        in the graph
    Returns
    -------
    closest_road_to_point : tuple
        A geometry object representing the segment and the coordinates of the two
        nodes that determine the road section, u and v, the OSM ids of the nodes.
    """
    gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    print(gdf.keys())
    graph_roads = gdf[["geometry", "u", "v"]].values.tolist()

    roads_with_distances = [
        (
            graph_road,
            Point(tuple(reversed(point))).distance(graph_road[0])
        )
        for graph_road in graph_roads
    ]

    roads_with_distances = sorted(roads_with_distances, key=lambda x: x[1])
    closest_road_to_point = roads_with_distances[0][0]
    return closest_road_to_point


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
    # xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    # ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    #
    # def det(a, b):
    #     return a[0] * b[1] - a[1] * b[0]
    #
    # div = det(xdiff, ydiff)
    # if div == 0:
    #     # raise Exception('lines do not intersect')
    #     print('lines do not intersect')
    #     return False
    #
    # d = (det(*line1), det(*line2))
    # x = det(d, xdiff) / div
    # y = det(d, ydiff) / div
    # if isBetween(line1[0], line1[1], (x, y)):
    #     return True
    # else:
    #     return False
    line1 = LineString([line1[0], line1[1]])
    line2 = LineString([line2[0], line2[1]])

    int_pt = line1.intersection(line2)
    return not int_pt.is_empty


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

                # x, y = get_intersection_point([(last_road_point[0],last_road_point[1]),(sub_x, sub_y)],
                #                               [(new_u_node['x'], new_u_node['y']),
                #                               (new_v_node['x'], new_v_node['y'])])




                if last_line is not None:
                    # Not first line

                    x, y = get_intersection_point2(last_line,
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

def node_distance(node1, node2):
    return ((((node2['x'] - node1['x']) ** 2) + ((node2['y'] - node1['y']) ** 2)) ** 0.5)


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
    for u, v, key, d in g_box.edges(keys=True, data=True):
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
        # for w in G_box.neighbors(u):
        #     neighbor = G_box.nodes[w]
        #     if intersection_between_points(((g_box.nodes[u]['x'], g_box.nodes[u]['y']), (neighbor['x'], neighbor['y']))
        #             , ((new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y']))):
        #         x, y = get_intersection_point(((g_box.nodes[u]['x'], g_box.nodes[u]['y']), (neighbor['x'], neighbor['y']))
        #             , ((new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y'])))
        #         if point_distance(new_v_node, (x, y)) < point_distance(new_u_node, (x, y)):
        #             new_v_node['x'] = x
        #             new_v_node['y'] = y
        #         else:
        #             new_u_node['x'] = x
        #             new_u_node['y'] = y
        #
        # for w in G_box.neighbors(v):
        #     neighbor = G_box.nodes[w]
        #     if intersection_between_points(((g_box.nodes[v]['x'], g_box.nodes[v]['y']), (neighbor['x'],neighbor['y']))
        #             , ((new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y']))):
        #         x, y = get_intersection_point(((g_box.nodes[v]['x'], g_box.nodes[v]['y']), (neighbor['x'],neighbor['y']))
        #             , ((new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y'])))
        #         if point_distance(new_v_node, (x, y)) < point_distance(new_u_node, (x, y)):
        #             new_v_node['x'] = x
        #             new_v_node['y'] = y
        #
        #         else:
        #             new_u_node['x'] = x
        #             new_u_node['y'] = y



        # tmp = ox.nearest_edges(g_box, new_u_node['x'], new_u_node['y'])
        # print(tmp)
        # new_u_node['x'] = tmp[1]
        # new_u_node['y'] = tmp[0]
        #
        # tmp = ox.nearest_edges(g_box, new_v_node['x'], new_v_node['y'], interpolate=1)
        # print(tmp)
        # new_v_node['x'] = tmp[1]
        # new_v_node['y'] = tmp[0]

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
    for point_pair in list(itertools.combinations(new_edges.keys(), 2)):
        edge1 = new_edges[point_pair[0]]
        edge2 = new_edges[point_pair[1]]

        if intersection_between_points(((edge1[0]['x'], edge1[0]['y']), (edge1[1]['x'], edge1[1]['y'])),
                                       ((edge2[0]['x'], edge2[0]['y']), (edge2[1]['x'], edge2[1]['y']))):
            x, y = get_intersection_point(((edge1[0]['x'], edge1[0]['y']), (edge1[1]['x'], edge1[1]['y'])),
                                          ((edge2[0]['x'], edge2[0]['y']), (edge2[1]['x'], edge2[1]['y'])))
            if point_distance(edge1[0], (x, y)) < point_distance(edge1[1], (x, y)):
                edge1[0]['x'] = x
                edge1[0]['y'] = y
            else:
                edge1[1]['x'] = x
                edge1[1]['y'] = y
            if point_distance(edge2[0], (x, y)) < point_distance(edge2[1], (x, y)):
                edge2[0]['x'] = x
                edge2[0]['y'] = y
            else:
                edge2[1]['x'] = x
                edge2[1]['y'] = y
            new_edges[point_pair[0]] = edge1
            new_edges[point_pair[1]] = edge2


        #print(get_nearest_road(G_box, new_u_node))
    # TODO: just connect all point and remove intersecting lines (both both of them)
    # candidate_edges = dict()
    # remove_candidate_edges = set()
    # for node_id, node in g_box.nodes.items():
    #     points = list()
    #     for u_line_keys in node_as_u[node_id]:
    #         u_trench = road_node_to_trench_nodes[u_line_keys]
    #         points.append(u_trench['trench_u'])
    #     for v_line_keys in node_as_v[node_id]:
    #         v_trench = road_node_to_trench_nodes[v_line_keys]
    #         points.append(v_trench['trench_v'])
    #     for point_pair in list(itertools.combinations(points, 2)):
    #         line_key = "_".join([str(point_pair[0]), str(point_pair[1])])
    #         d = last_d[node_id]
    #         d['length'] = node_distance(point_pair[0], point_pair[1])
    #         candidate_edges[line_key] = (point_pair[0], point_pair[1], 1, d)
    #
    #     for line_key_pair in candidate_edges.keys():
    #         edge = candidate_edges[line_key_pair]
    #
    #     for line_key_pair in list(itertools.combinations(candidate_edges.keys(), 2)):
    #         edge1 = candidate_edges[line_key_pair[0]]
    #         edge2 = candidate_edges[line_key_pair[1]]
    #         if intersection_between_points(((edge1[0]['x'], edge1[0]['y']), (edge1[1]['x'], edge1[1]['y'])),
    #                                        ((edge2[0]['x'], edge2[0]['y']), (edge2[1]['x'], edge2[1]['y']))):
    #             if line_key_pair[0] in remove_candidate_edges or line_key_pair[1] in remove_candidate_edges:
    #                 pass
    #             else:
    #                 if node_distance(edge1[0], edge1[1]) < node_distance(edge2[0], edge2[1]):
    #                     remove_candidate_edges.add(line_key_pair[0])
    #                 else:
    #                     remove_candidate_edges.add(line_key_pair[1])
    #     for line_key, edge in candidate_edges.items():
    #         if line_key not in remove_candidate_edges:
    #             new_edges[line_key] = edge
    #             print(edge)
    #         else:
    #             print("Removed")
    #
    #
    #
    # for node_id, node in g_box.nodes.items():
    #     if node['street_count'] != 1:
    #         for u_line_keys in node_as_u[node_id]:
    #             u_trench = road_node_to_trench_nodes[u_line_keys]
    #             for v_line_keys in node_as_v[node_id]:
    #                 v_trench = road_node_to_trench_nodes[v_line_keys]
    #                 if not fix_intersecting_trenches(u_trench['trench'], v_trench['trench']):
    #                     if u_trench['v'] != v_trench['u']:
    #                         line_key = "_".join([str(u_line_keys), str(v_line_keys)])
    #                         #new_edges.append((v_trench['trench_v'], u_trench['trench_u'], 1, last_d[node_id]))
    #                         new_edges[line_key] = (v_trench['trench_v'], u_trench['trench_u'], 1, last_d[node_id])

    return list(new_edges.values())


G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, network_type='drive', simplify=True, retain_all=False)
#G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521, network_type='drive', simplify=True, retain_all=False)

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