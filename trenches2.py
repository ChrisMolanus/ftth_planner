from enum import Enum
from typing import Dict, List, Tuple, Set, Any, Hashable

import geopandas
import networkx
import numpy as np
import osmnx as ox
import itertools

import matplotlib.pyplot as plt
import math

import pandas as pd
import geopandas as gpd
import pyproj
from shapely.geometry import LineString

distance_from_center_of_road = 0.0001
geod = pyproj.Geod(ellps='WGS84')
# zone 31 for benelux
P = pyproj.Proj(proj='utm', zone=31, ellps='WGS84', preserve_units=True)

def point_distance_from_line(line: Tuple[dict, dict], point: dict) -> float:
    """
    The distance between a point and a line
    :param line: The line
    :param point: The point
    :return: The distance between the point and the line
    """
    return (((point['x'] - line[0]['x'])*(line[1]['y']-line[0]['y']))
            - ((point['y'] - line[0]['y']) * (line[1]['x'] - line[0]['x'])))


def node_distance(node1: dict, node2: dict) -> float:
    """
    The distance between two points
    :param node1: A point
    :param node2: A point
    :return: The distance between the two points
    """
    azimuth1, azimuth2, distance = geod.inv(node1['y'], node1['x'], node2['y'], node2['x'])
    return distance


def angle(vector1: Tuple[float, float], vector2: Tuple[float, float]) -> float:
    """
    Returns the clockwise angle between two vectors in radian
    :param vector1: A vector
    :param vector2: A vector
    :return: The angle between the two vectors
    """
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    if y2 < y1:
        return math.pi - math.acos(inner_product/(len1*len2)) + math.pi
    else:
        return math.acos(inner_product/(len1*len2))


def point_on_circle(center: dict, radius: float, radian: float) -> Tuple[float, float]:
    """
    Returns a point on a circle with the center "center" and a radius "radius" at the angle "radian"
    :param center: The center of the circle
    :param radius: The radius of the circle
    :param radian: The angle of the point need relative to (0,0)(1,0) in radian not degrees
    :return: a point on a circle
    """
    x = center['x'] + (radius * math.cos(radian))
    y = center['y'] + (radius * math.sin(radian))
    return x, y

def _LatLon_To_XY(Lat,Lon):
  return P(Lat,Lon)

def _XY_To_LatLon(x,y):
  return P(x,y,inverse=True)

def get_perpendicular_line(u_node: dict, v_node: dict, ref_point: dict) -> Tuple[dict, dict]:
    """
    Returns the projection of a point over a line, corresponding to the perpendicular line.
    :param u_node: The start node of the vector
    :param v_node: The end node of the vector
    :param ref_point: The point we would like to project
    :return: The projected point over the vector
    """
    x1, y1 = _LatLon_To_XY(u_node['x'], u_node['y'])
    x2, y2 = _LatLon_To_XY(v_node['x'], v_node['y'])
    x3, y3 = _LatLon_To_XY(ref_point['x'], ref_point['y'])
    dx = x2 - x1
    dy = y2 - y1

    if (dy ** 2 + dx ** 2) != 0:
        k = (dy * (x3 - x1) - (dx) * (y3 - y1)) / (dy ** 2 + dx ** 2)
        x4 = x3 - k * dy
        y4 = y3 + k * dx
    px4, py4 = _XY_To_LatLon(x4, y4)
    return {'x': ref_point['x'], 'y': ref_point['y']}, {'x': px4, 'y': py4}


def point_on_line(u, v, c, return_distance=False):
    p1 = np.array([u['x'], u['y']])
    p2 = np.array([v['x'], v['y']])
    p3 = np.array([c['x'], c['y']])
    l2 = np.sum((p1 - p2) ** 2)
    t = np.sum((p3 - p1) * (p2 - p1)) / l2

    projection = p1 + t * (p2 - p1)

    dist = np.sum((p3 - projection) ** 2)
    if return_distance:
        return projection, dist
    else:
        return projection


class TrenchCorner(dict):
    def __init__(self, x: float, y: float, trench_count: int, u_node_id: int, street_ids: Set,
                 node_for_adding: int = None, *args, **kw):
        """
        A FttH planner trench corner
        :param x: The OSMnx x coordinate of the node
        :param y: The OSMnx y coordinate of the node
        :param trench_count:
        :param u_node_id: The OSMnx node ID of the intersection this corner is on
        :param street_ids: A SET of the string- representation of the sorted list of node IDs
        :param node_for_adding: The OSMX node id, default is None if not yet known
        :param args: Dict args
        :param kw: Dict kw
        """
        super(TrenchCorner, self).__init__(*args, **kw)
        self['x'] = x
        self['y'] = y
        self['trench_count'] = trench_count
        self['u'] = u_node_id
        self['street_count'] = 1
        self['street_ids'] = street_ids
        self['node_for_adding'] = node_for_adding


    def __cmp__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']

    def __hash__(self):
        return hash((self['x'], self['y']))

    def __eq__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']


class TrenchType(Enum):
    Road_side = "Road-Side Trench"
    Road_crossing = "Road-Crossing Trench"
    Building = "To Building Trench"


class Trench(dict):
    def __init__(self, u_for_edge: int, v_for_edge: int, name: str, length: float, street_names: Set[str], trench: bool = True,
                 trench_crossing: bool = False, geometry: LineString = None, house_trench: bool = False,  *args, **kw):
        super(Trench, self).__init__(*args, **kw)
        self['u_for_edge'] = u_for_edge
        self['v_for_edge'] = v_for_edge
        self['name'] = name
        self['length'] = length
        self.street_names = street_names
        self['trench'] = trench
        self['trench_crossing'] = trench_crossing
        self['house_trench'] = house_trench
        if geometry is not None:
            self['geometry'] = geometry
            self.has_geometry = True
        else:
            self.has_geometry = False

        if self['trench_crossing']:
            self.type = TrenchType.Road_crossing
        elif self['house_trench']:
            self.type = TrenchType.Building
        else:
            self.type = TrenchType.Road_side

    def has_geometry(self) -> bool:
        return self.has_geometry

    def __eq__(self, other):
        return (self['u_for_edge'] == other['u_for_edge'] and self['v_for_edge'] == other['v_for_edge']) \
               or (self['u_for_edge'] == other['v_for_edge'] and self['v_for_edge'] == other['u_for_edge'])

    def __hash__(self):
        if self['u_for_edge'] > self['v_for_edge']:
            id_t = (self['u_for_edge'], self['v_for_edge'])
        else:
            id_t = (self['v_for_edge'], self['u_for_edge'])
        return hash(id_t)

    def __str__(self):
        if not self['trench_crossing']:
            if self.has_geometry:
                return "Curved trench " + self['name']
            else:
                return "Trench " + self['name']
        else:
            return "Road crossing Trench " + self['name']


def get_parallel_line_points(u_node: dict, v_node: dict, vector_distance: float, side_id: int) -> Tuple[dict, dict]:
    """
    Returns a vector parallel to the vector (u_node, v_node) on one side of the vector.
    :param u_node: The start node of the vector
    :param v_node: The end node of the vector
    :param vector_distance: The distance between the vector and the parallel vector
    :param side_id: The side of the vector to create a parallel vector on, 0 or 1
    :return: A vector parallel to the vector (u_node, v_node) on one side of the vector.
    """
    dx = u_node['x'] - v_node['x']
    dy = u_node['y'] - v_node['y']

    road_length = math.sqrt(dx ** 2 + dy ** 2)
    if road_length == 0:
        road_length = 0.00001
    t = vector_distance / road_length

    # Perpendicular vector
    # TODO: flipping dx and dy is probably not correct but the point looks worse if we correct it
    if side_id == 1:
        dx1 = -1 * dy
        dy1 = dx
    else:
        dx1 = dy
        dy1 = -1 * dx

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


def get_intersection_point2(line1: Tuple[dict, dict],
                            line2: Tuple[dict, dict]) -> dict:
    """
    Returns the point where the two lines intersect
    :param line1: Fist line
    :param line2: Second line
    :return: The point where the two lines intersect
    """
    l1 = ((line1[0]['x'], line1[0]['y']), (line1[1]['x'], line1[1]['y']))
    l2 = ((line2[0]['x'], line2[0]['y']), (line2[1]['x'], line2[1]['y']))
    p = get_intersection_point(l1, l2)
    return {'x': p[0], 'y': p[1]}


def get_intersection_point(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                           line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    """
    Returns the point where the two lines intersect
    :param line1: Fist line
    :param line2: Second line
    :return: The point where the two lines intersect
    """
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


def get_trench_linestring(u_side_corners: List[TrenchCorner], v_side_corners: List[TrenchCorner],
                          street, ref_distance_from_center_of_road: float, side_id: int) -> dict:
    """
    Returns a curved trench parallel to the road on one side of the road.
    :param u_side_corners: A set of trench corners around the first point in geometry of this road.
    :param v_side_corners: A set of trench corners around the last point in geometry of this road.
    :param street: The data of the street
    :param ref_distance_from_center_of_road: The distance the trench should be from the road
    :param side_id: The side of the road the trench should be on 0 or 1
    :return: A curved trench parallel to the road on one side of the road.
    """
    linestring = list()
    total_road_length = 0
    last_road_point = None
    last_trench_point = None
    last_line = None
    closest_u_for_trench = None
    for sub_x, sub_y in street['geometry'].coords:
        if last_road_point is not None:
            new_u_node, new_v_node = get_parallel_line_points({'x': last_road_point[0], 'y': last_road_point[1],
                                                               'street_count': 1},
                                                              {'x': sub_x, 'y': sub_y, 'street_count': 1},
                                                              ref_distance_from_center_of_road, side_id)

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
                # Find Trench corner that is closest to this point
                # TODO: check if corner is closer to new_v_node
                u_trench_shortest_distance = 10000000
                for v_corner in u_side_corners:
                    current_distance = node_distance(v_corner, new_u_node)
                    if current_distance < u_trench_shortest_distance:
                        u_trench_shortest_distance = current_distance
                        closest_u_for_trench = v_corner
                new_u_node = closest_u_for_trench
                x = new_u_node['x']
                y = new_u_node['y']

            linestring.append((new_u_node['x'], new_u_node['y']))
            line = [(new_u_node['x'], new_u_node['y']), (new_v_node['x'], new_v_node['y'])]
            last_line = line

            dx = x - new_v_node['x']
            dy = y - new_v_node['y']
            # TODO: Correct for last point that will change to V Node
            total_road_length += math.sqrt(dx ** 2 + dy ** 2)

            last_trench_point = (new_v_node['x'], new_v_node['y'])

        last_road_point = (sub_x, sub_y)

    closest_v_for_trench = None
    u_trench_shortest_distance = 10000000
    # TODO: check if corner is closer to point before last_trench_point
    for v_corner in v_side_corners:
        current_distance = node_distance(v_corner, {'x': last_trench_point[0], 'y': last_trench_point[0]})
        if current_distance < u_trench_shortest_distance:
            u_trench_shortest_distance = current_distance
            closest_v_for_trench = v_corner
    linestring.append((closest_v_for_trench['x'], closest_v_for_trench['y']))

    if isinstance(street['name'], str):
        street_names = {street['name']}
    else:
        street_names = set(street['name'])

    return {'u_for_edge': closest_u_for_trench['node_for_adding'],
            'v_for_edge': closest_v_for_trench['node_for_adding'],
            'geometry': LineString(linestring),
            'length': total_road_length,
            'name': f"Curved Trench {street['name']}",
            'street_names': street_names}


def get_trench_corners(road_network: networkx.MultiDiGraph,
                       ref_distance_from_center_of_road: float) -> Tuple[Dict[str, Set[TrenchCorner]],
                                                                         Dict[str, List[Trench]]]:
    """
    Create TrenchCorners (Nodes) for every intersection in the network.
    The TrenchCorners will be places between each of the roads of the intersection.
    It also creates the trenches between those points to connect road trenches to each other (road_crossing Trenches)
    :param road_network: The road network
    :param ref_distance_from_center_of_road: The distance the trenches should be from the center of the road
    :return: trench_corners, road_crossing
    """
    # make network undirected so that one way street nodes have two neighbors
    network = road_network.to_undirected()
    nodes = dict()
    output_trench_corners = dict()
    output_road_crossing = dict()
    # Pick some very large number so it will likely not overlap with an existing OSMID
    node_id = 400000000

    # Loop though every intersection
    for u, current_node in network.nodes(data=True):
        neighbors = dict()
        # Then look for all of the other intersections what it is connected to by a road (street)
        # we will create vectors from this intersection "u" to the neighbors "v"s
        for v in network.neighbors(u):
            neighbor = network.nodes[v]
            street = network.get_edge_data(u, v)[0]
            if 'geometry' not in street:
                # Its' a simple straight line so that the other intersection as point ot form the vector
                radian = angle((1.0, 0.0), (neighbor['x']-current_node['x'], neighbor['y']-current_node['y']))
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

                # Find the angel between a horizontal line ( (1,0) ) the road vector (in radian, not degrees)
                radian = angle((1.0, 0.0), (v1[0] - current_node['x'], v1[1] - current_node['y']))
            neighbors[radian] = v

        # Sort the angles since if ther are more that 3 we want to only put corners between adjacent vectors
        sorted_vs = list(neighbors.keys())
        sorted_vs.sort()

        first_radian = None
        first_street_id = None
        last_radian = None
        last_street_id = None
        last_node_id = None
        first_node_id = None
        radian_street_segment_id: str = ""
        # Loop though the street vectors in a clockwise order (sorted_vs.sort())
        for radian in sorted_vs:
            v = neighbors[radian]
            streets = network.get_edge_data(u, v)
            street_names = [u, v]
            # we can get this segment twice to sorting the node ids make sure they have the same street_segment_id
            street_names.sort()
            radian_street_segment_id = str(street_names)
            if len(streets) > 1:
                print("Warning len(streets) > 1, This can happen if the GBox hacked the street into multiple segments")
                print(streets)

            # We need two vectors to find a trench corner between them
            if last_radian is not None:
                # Find a angle between the two other angles
                between_radian = radian - (abs(radian - last_radian) / 2)
                # Find a point on a circle with the radius of distance_from_center_of_road at that angle
                x, y = point_on_circle(current_node, ref_distance_from_center_of_road, between_radian)
                # Create a Trench Corner at that point
                node = TrenchCorner(x, y, 2, u, {radian_street_segment_id, last_street_id})
                if radian_street_segment_id not in output_trench_corners:
                    output_trench_corners[radian_street_segment_id] = set()
                if node not in output_trench_corners[first_street_id] \
                        and node not in output_trench_corners[last_street_id]:
                    node_id += 1
                    node['node_for_adding'] = node_id
                    output_trench_corners[radian_street_segment_id].add(node)
                    output_trench_corners[last_street_id].add(node)
                    nodes[node.__hash__()] = node
                else:
                    node_id = nodes[node.__hash__()]['node_for_adding']

                if last_node_id is not None:
                    if last_street_id not in output_road_crossing:
                        output_road_crossing[last_street_id] = list()
                    output_road_crossing[last_street_id].append(Trench(u_for_edge=last_node_id,
                                                                       v_for_edge=node_id,
                                                                       name=last_street_id,
                                                                       length=2 * ref_distance_from_center_of_road,
                                                                       street_names=last_street_names,
                                                                       trench_crossing=True
                                                                       )
                                                                )
                else:
                    first_node_id = node_id
                last_node_id = node_id
            else:
                first_radian = radian
                first_street_id = radian_street_segment_id
                if isinstance(street_names, str):
                    first_street_names = {street_names}
                else:
                    first_street_names = set(street_names)
                if radian_street_segment_id not in output_trench_corners:
                    output_trench_corners[radian_street_segment_id] = set()
            last_radian = radian
            last_street_id = radian_street_segment_id
            if isinstance(street_names, str):
                last_street_names = {street_names}
            else:
                last_street_names = set(street_names)

        # Now all we have left if to create a trench corner between the last vector and the first vector
        if len(sorted_vs) > 1:
            first_radian = first_radian + (2 * math.pi)
            between_radian = first_radian - (abs(first_radian - last_radian) / 2)
            x, y = point_on_circle(current_node, ref_distance_from_center_of_road, between_radian)
            node = TrenchCorner(x, y, 2, u, {first_street_id, last_street_id})
            if node not in output_trench_corners[first_street_id] and node not in output_trench_corners[last_street_id]:
                node_id += 1
                node['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node)
                output_trench_corners[last_street_id].add(node)
                if last_street_id not in output_road_crossing:
                    output_road_crossing[last_street_id] = list()
                output_road_crossing[last_street_id].append(Trench(u_for_edge=last_node_id,
                                                                   v_for_edge=node_id,
                                                                   name=last_street_id,
                                                                   length=2*ref_distance_from_center_of_road,
                                                                   street_names=last_street_names,
                                                                   trench_crossing=True
                                                                   )
                                                            )

                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()
                output_road_crossing[first_street_id].append(Trench(u_for_edge=node_id,
                                                                    v_for_edge=first_node_id,
                                                                    name=first_street_id,
                                                                    length=2*ref_distance_from_center_of_road,
                                                                    street_names=first_street_names,
                                                                    trench_crossing=True
                                                                    )
                                                            )
        elif len(sorted_vs) == 1:
            # This is a Dead end road, there was only 1 neighbor
            # So we make a "T" shape with a road crossing trench at the top
            between_radian = first_radian + math.pi * 0.5
            x, y = point_on_circle(current_node, ref_distance_from_center_of_road, between_radian)
            node1 = TrenchCorner(x, y, 2, u, {radian_street_segment_id})
            if node1 not in output_trench_corners[first_street_id] \
                    and node1 not in output_trench_corners[last_street_id]:
                node_id += 1
                node1['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node1)
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()

            between_radian = first_radian + math.pi * 1.5
            x, y = point_on_circle(current_node, ref_distance_from_center_of_road, between_radian)
            node2 = TrenchCorner(x, y, 2, u, {radian_street_segment_id})
            if node2 not in output_trench_corners[first_street_id] \
                    and node2 not in output_trench_corners[last_street_id]:
                node_id += 1
                node2['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node2)
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()

            output_road_crossing[first_street_id].append(Trench(u_for_edge=node1['node_for_adding'],
                                                                v_for_edge=node2['node_for_adding'],
                                                                name=first_street_id,
                                                                length=node_distance(node1, node2),
                                                                street_names=first_street_names,
                                                                trench_crossing=True
                                                                ),
                                                         )

    return output_trench_corners, output_road_crossing


def is_between2(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any]) -> bool:
    """
    Is point c between points a and b, but where the points are dicts instead of tuples
    :param a: A point
    :param b: A point
    :param c: The point that might be between "a" and "b"
    :return: True is "c" is between "a" and "b" (or close enough for floating point precision)
    """
    a1 = (a["x"], a["y"])
    b1 = (b["x"], b["y"])
    c1 = (c["x"], c["y"])
    return is_between(a1, b1, c1)


def is_between(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """
    Is point c between points a and b, but where the points are tuples instead of dicts
    :param a: A point
    :param b: A point
    :param c: The point that might be between "a" and "b"
    :return: True is "c" is between "a" and "b" (or close enough for floating point precision)
    """
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


def intersection_between_points(l1: List[dict], l2: List[dict]) -> bool:
    """
    Returns True if two line intersect at a point on both lines
    :param l1: A line
    :param l2: A line
    :return: True if two line intersect at a point on both lines
    """
    line1 = (l1[0]['x'], l1[0]['y']), (l1[1]['x'], l1[1]['y'])
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
    if is_between((line1[0][0], line1[0][1]),
                  (line1[1][0], line1[1][1]),
                  (x, y)):
        return True
    else:
        return False


class TrenchNetwork:
    def __init__(self, trench_corners: Dict[str, TrenchCorner], trenches: List[Trench],
                 building_trenches_lookup: Dict[str, Tuple[int, int]], corner_by_id: Dict[int, TrenchCorner]):
        """
        A Cognizant FttH Trench Network
        :param trench_corners: The nodes of the network
        :param trenches: The edges of the network
        :param building_trenches_lookup: the building's centroid and trenchcorner of building trench
        :param corner_by_id: The nodes of the network keyed by id (should replace trench_corners)
        """
        self.trenchCorners = trench_corners
        self.building_trenches_lookup = building_trenches_lookup
        self.trenches = trenches
        self.corner_by_id = corner_by_id


class TrenchInfo:
    def __init__(self, building_centroid_node: Dict[str, Hashable], ref_new_v_node: dict, closest_trench: int,
                 geometry: bool, segment_index: int, ref_corner_u: TrenchCorner):
        """
        An object that is ment to hold information about how a building should be connected to teh closest road trench
        :param building_centroid_node: A point that represents the termination point of the building trench
        :param ref_new_v_node: The point where we should create a new Node for the road sub-trenches
        :param closest_trench: The index in the "trenches" list of the closest trench
        :param geometry: Is the road trench a cured road i.e LineString
        :param segment_index: The index in the curved trench's LineString
        that is the point that is closest to the building
        :param ref_corner_u: The "u" point of the original road trench
        """
        self.building_centroid_node = building_centroid_node
        self.new_v_node = ref_new_v_node
        self.closest_trench = closest_trench
        self.geometry = geometry
        self.segment_index = segment_index
        self.corner_u = ref_corner_u

    def __eq__(self, other):
        if self.geometry != other.geometry:
            print("Warning comparing trenches of different types (curved vs straight)")
            return False
        if self.geometry and other.geometry:
            return self.segment_index == other.segment_index
        else:
            return node_distance(other.new_v_node, other.corner_u) == node_distance(self.new_v_node, self.corner_u)

    def __gt__(self, other):
        if self.geometry and other.geometry:
            return self.segment_index > other.segment_index
        else:
            return node_distance(other.new_v_node, other.corner_u) < node_distance(self.new_v_node, self.corner_u)


def get_building_by_closest_trench(building_gdf: geopandas.GeoDataFrame,
                                   trench_corners: Dict[int, TrenchCorner],
                                   trenches: List[Trench]) -> Dict[int, List[TrenchInfo]]:
    """
    Return the a dictionary where the keys are the index of a road trench in the trenches List
    and the value is a list of all building trench information for each building on that road trench
    :param building_gdf: The GeoPandas dataFrame of the building
    :param trench_corners: The current trench corners
    :param trenches: The current trenches
    :return: The Trench Info dictionary
    """
    # Create dictionary of Trench corners by their Node ID
    corner_by_id: Dict[int, TrenchCorner] = dict()
    for intersection_osmid, corners in trench_corners.items():
        for corner in corners:
            corner_by_id[corner['node_for_adding']] = corner

    # Create dictionary of Trenches that are candidates for a street address
    street_trenches: Dict[str, Dict[int, Trench]] = dict()
    streets = building_gdf['addr:street'].unique()
    for street_name in streets:
        street_trenches[street_name] = dict()
        for i in range(0, len(trenches)):
            trench = trenches[i]
            if street_name in trench.street_names:
                street_trenches[street_name][i] = trench

    # Loop over every building and try and find the road trench that is closest to it
    # and find the intersection point between the road trench and a perpendicular line
    # of the road trench that goes through the building centroid
    building_by_closest_trench: Dict[int, List[TrenchInfo]] = dict()
    for building_index, building in building_gdf.iterrows():
        closest_trench_info = None
        street_name = building['addr:street']
        centroid = building['geometry'].centroid
        distance = float('inf')
        building_centroid_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0], 'building_index': building_index}
        # There might be buildings in the box that are on roads that are not in the box, geo fencing problem
        # or just buildings with no address
        if street_name in street_trenches and len(street_trenches[street_name]) > 0:
            # Loop over every trench for this street and find the closest one
            for trench_index, trench in street_trenches[street_name].items():
                if trench_index not in building_by_closest_trench:
                    building_by_closest_trench[trench_index] = list()
                corner_u: TrenchCorner = corner_by_id[trench['u_for_edge']]
                corner_v: TrenchCorner = corner_by_id[trench['v_for_edge']]
                if 'geometry' not in trenches[trench_index]:
                    # Get the intersection point between the road trench and a perpendicular line of the building
                    perpendicular_line = get_perpendicular_line(corner_u, corner_v, building_centroid_node)
                    projected = get_intersection_point2(perpendicular_line, (corner_u, corner_v))
                    # Extra check to make sure we are not doing something wrong, might be a bug in the code
                    if is_between2(corner_u, corner_v, projected):
                        new_distance = node_distance(projected, building_centroid_node)
                        # Check if this trench is the closest one so far
                        if new_distance < distance:
                            new_v_node = projected
                            distance = new_distance
                            closest_trench = trench_index
                            closest_trench_info = {'building_centroid_node': building_centroid_node,
                                                   'ref_new_v_node': new_v_node,
                                                   'closest_trench': closest_trench,
                                                   'geometry': False,
                                                   'ref_corner_u': corner_u,
                                                   'segment_index': None}

                else:
                    # This is an attempt of finding a closest trench but for trenches that have geometry
                    # it is not used if the g_box is no simplified
                    trench = trenches[trench_index]
                    coords = list(trench['geometry'].coords)
                    last_node = None
                    for segment_index in range(0, len(coords)):
                        sub_x, sub_y = coords[segment_index]
                        if last_node is None:
                            last_node = {'x': sub_x, 'y': sub_y}
                        else:
                            sub_u_node = {'x': sub_x, 'y': sub_y}
                            perpendicular_line = get_perpendicular_line(last_node, sub_u_node, building_centroid_node)
                            projected = get_intersection_point2(perpendicular_line, (last_node, sub_u_node))
                            if is_between2(last_node, sub_u_node, projected):
                                new_distance = node_distance(projected, building_centroid_node)
                                last_node = sub_u_node
                                if new_distance < distance:
                                    new_v_node = projected
                                    distance = new_distance
                                    shortest_i = segment_index
                                    closest_trench = trench_index
                                    closest_trench_info = {'building_centroid_node': building_centroid_node,
                                                           'ref_new_v_node': new_v_node,
                                                           'closest_trench': closest_trench,
                                                           'geometry': True,
                                                           'segment_index': shortest_i,
                                                           'ref_corner_u': corner_u}

            # It is possible we could not find a road trench for this building, geo fencing problem
            if closest_trench_info is not None:
                building_by_closest_trench[closest_trench_info['closest_trench']].append(
                    TrenchInfo(**closest_trench_info))
    return building_by_closest_trench


def get_sub_trenches_for_buildings(building_by_closest_trench: Dict[int, List[TrenchInfo]],
                                   trenches: List[Trench],
                                   trench_corners: Dict[str, Set[TrenchCorner]]
                                   ) -> Tuple[Dict[str, Set[TrenchCorner]], List[Trench], List[int]]:
    """
    Returns the:
    - new_trench_corners: The building Nodes and the new road sub-trench Nodes as TrenchCorner
    - new_trenches: The Building Trenches and the new road sub-trenches
    - trench_indexes_to_remove: A list if indexes of the "trenches" list that should be removed
      since we have replaced them with sub-trenches
    :param building_by_closest_trench: The Trench Info dictionary
    :param trenches: The current list of trenches
    :param trench_corners: The dict of trench corners
    :return: new_trench_corners, new_trenches, trench_indexes_to_remove
    """
    # TODO: This is duplicate code from get_building_by_closest_trench, we could pass the object or separate methode
    # Create dictionary of Trench corners by their Node ID
    corner_by_id: Dict[int, TrenchCorner] = dict()
    for intersection_osmid, corners in trench_corners.items():
        for corner in corners:
            corner_by_id[corner['node_for_adding']] = corner

    new_trenches: List[Trench] = list()
    new_trench_corners: Dict[str, Set[TrenchCorner]] = dict()
    node_id = 500000000
    trench_indexes_to_remove = list()
    # object used by fiber network to find road trench nodes for street cabinets
    building_trenches_lookup: Dict[str, Tuple[int, int]] = dict()

    # Loop over all the buildings that we found closest trenches for and create the building trenches
    # and the new sub-trenches that should replace the current road trenches
    for trench_index, building_trench_info in building_by_closest_trench.items():
        # It's possible that this trench has no buildings that should be connected to it
        if len(building_trench_info) > 0:
            trench_indexes_to_remove.append(trench_index)
            last_shortest_i = 0
            trench = trenches[trench_index]
            last_node_id = trench['u_for_edge']
            last_node = corner_by_id[last_node_id]
            # Since we are creating the sub-trenches from teh "u" node the the "v" node
            # We have to order the buildings so we chain the sub-trenches correctly
            # The Trench Info object order them selves based on how far they are from the "u" node of the road trench
            building_trench_info.sort()
            # Loop over the now sorted building infos and create a new building trench
            # and the next sub-trench in teh chain
            for closest_trench_info1 in building_trench_info:
                # Create the new road sub-trench node
                node_id += 1
                new_v_node_id = node_id
                new_v_node = TrenchCorner(x=closest_trench_info1.new_v_node["x"],
                                          y=closest_trench_info1.new_v_node["y"],
                                          trench_count=3,
                                          u_node_id=closest_trench_info1.corner_u,
                                          street_ids=set(),
                                          node_for_adding=new_v_node_id,
                                          )
                if str(trench_index) not in new_trench_corners:
                    new_trench_corners[str(trench_index)] = set()
                new_trench_corners[str(trench_index)].add(new_v_node)

                # Create the building Node
                node_id += 1
                building_node_id = node_id
                building_node = TrenchCorner(x=closest_trench_info1.building_centroid_node["x"],
                                             y=closest_trench_info1.building_centroid_node["y"],
                                             trench_count=1,
                                             u_node_id=closest_trench_info1.corner_u,
                                             street_ids=set(),
                                             node_for_adding=building_node_id,
                                             building_index=closest_trench_info1.building_centroid_node[
                                                 'building_index'])
                # Since we are looping the buildings the building_index is unique,
                # so we don't have to check if the key already exists
                new_trench_corners[str(closest_trench_info1.building_centroid_node['building_index'])] = {building_node}

                # Create teh building trench and the next road sub-trench in the chain
                if closest_trench_info1.geometry:
                    coords = list(trench['geometry'].coords)
                    t = coords[last_shortest_i:closest_trench_info1.segment_index]
                    last_shortest_i = closest_trench_info1.segment_index
                    trench_length = node_distance(last_node, new_v_node)
                    if len(t) > 1:
                        line_string = LineString(t)
                        sub_trench = Trench(last_node_id, new_v_node_id, "sub " + trench["name"], trench_length,
                                            trench.street_names, True, False,
                                            line_string)
                        new_trenches.append(sub_trench)
                    else:
                        sub_trench = Trench(last_node_id, new_v_node_id, "sub " + trench["name"], trench_length,
                                            trench.street_names, True, False)
                        new_trenches.append(sub_trench)
                    trench_length = node_distance(new_v_node, building_node)
                    building_trench = Trench(new_v_node_id, building_node_id, "House Trench", trench_length,
                                             trench.street_names,
                                             True,
                                             False, None, house_trench=True)
                    new_trenches.append(building_trench)
                else:
                    trench_length = node_distance(last_node, new_v_node)
                    sub_trench = Trench(last_node_id, new_v_node_id, "sub " + trench["name"], trench_length,
                                        trench.street_names, True, False)
                    new_trenches.append(sub_trench)
                    trench_length = node_distance(new_v_node, building_node)
                    building_trench = Trench(new_v_node_id, building_node_id, "House Trench", trench_length,
                                             trench.street_names,
                                             True,
                                             False, None, house_trench=True)
                    new_trenches.append(building_trench)
                building_trenches_lookup[closest_trench_info1.building_centroid_node['building_index']] = \
                    (building_node_id, new_v_node_id)
                last_node_id = new_v_node_id
                last_node = new_v_node

            # Add the last sub-trench in the chain to connect to the "v" node of the original road trench
            v_node = corner_by_id[trench["v_for_edge"]]
            trench_length = node_distance(last_node, v_node)
            sub_trench = Trench(last_node_id, trench["v_for_edge"], "sub " + trench["name"], trench_length,
                                trench.street_names, True, False)
            new_trenches.append(sub_trench)
    return new_trench_corners, new_trenches, trench_indexes_to_remove, building_trenches_lookup


def get_trench_network(road_network: networkx.MultiDiGraph,
                       building_gdf: geopandas.GeoDataFrame) -> TrenchNetwork:
    """
    Creates a Trench Network based on the roads and buildings
    :param road_network: The Open Street map road network
    :param building_gdf: The GeoPandas DataFrame of the buildings that should be connected in the trench network
    :return: The TrenchNetwork
    """
    trench_corners, road_crossing = get_trench_corners(road_network, distance_from_center_of_road)

    new_pp = list()
    new_curved_pp = list()
    point_edges = dict()
    for u, v, key, street in road_network.edges(keys=True, data=True):
        added_trenches = set()

        u_node = road_network.nodes[u]
        v_node = road_network.nodes[v]
        s = [u, v]
        s.sort()
        street_segment_id = str(s)
        # Make sure we have trench corners that re on this street
        if street_segment_id in trench_corners:
            corners = trench_corners[street_segment_id]

            # Check if this is a curved road
            if 'geometry' not in street:

                # Since that same street can have multiple segments between intersections
                # make sure we have the trench corners of the intersections of this street segment
                filtered_corners = set()
                for corner in corners:
                    if corner['u'] == u or corner['u'] == v:
                        filtered_corners.add(corner)

                # To try and prevent getting trenches that cross streets
                # we split the trench corners be the side of the road
                street_sides = [[], []]
                for corner in filtered_corners:
                    if point_distance_from_line((u_node, v_node), corner) > 0:
                        street_sides[1].append(corner)
                    else:
                        street_sides[0].append(corner)

                # We look for the shortest possible trench on each side of the road separately
                for side_id in range(0, len(street_sides)):
                    side_corners = street_sides[side_id]
                    # Create possible trench corner pairs but looking for all possible combinations of corner points
                    for point_pair1 in list(itertools.combinations(side_corners, 2)):
                        # Only consider corner point pairs of points on different intersections
                        # Otherwize they are on the same intersections and that is a road crossing
                        if point_pair1[0]['u'] != point_pair1[1]['u']:
                            # trench_candidate as to be a list because tuples are immutable,
                            # and we might invalidate it later whe we chose from the candidates
                            if "name" in street:
                                trench_candidate = [point_pair1[0], point_pair1[1], street['name']]
                            else:
                                trench_candidate = [point_pair1[0], point_pair1[1], "Unknown"]
                            # There is no need to have multiple trenches between the same two points
                            # So only process a pair ones
                            xs = [trench_candidate[0]['x'], trench_candidate[1]['x']]
                            ys = [trench_candidate[0]['y'], trench_candidate[1]['y']]
                            xs.sort()
                            ys.sort()
                            trench_candidate_hash = hash((xs[0], xs[1], ys[0], ys[1]))
                            if not intersection_between_points([u_node, v_node], trench_candidate) \
                                    and trench_candidate_hash not in added_trenches:
                                added_trenches.add(trench_candidate_hash)

                                # Because it is possible that their are more than one pair that could from a trench
                                # on one side of a street segment, we collect them all and find the shortest one later
                                if street_segment_id not in point_edges:
                                    point_edges[street_segment_id] = dict()
                                if trench_candidate_hash not in point_edges[street_segment_id]:
                                    point_edges[street_segment_id][trench_candidate_hash] = [[], []]

                                point_edges[street_segment_id][trench_candidate_hash][side_id].append(trench_candidate)

                                new_pp.append(trench_candidate)

            else:
                # Since that same street can have multiple segments between intersections
                # make sure we have the trench corners of the intersections of this street segment
                u_filtered_corners = set()
                for corner in corners:
                    if corner['u'] == u:
                        u_filtered_corners.add(corner)

                v_filtered_corners = set()
                for corner in corners:
                    if corner['u'] == v:
                        v_filtered_corners.add(corner)

                # To try and prevent getting trenches that cross streets
                # we split the trench corners be the side of the road
                # Since the road is curved we get the side by only considering the segments
                # of the road that the "u" and "v" nodes are connected to, so teh first and last segment respectively
                # Sine we are determining side, the direction of the vector is never important ("u" to "v" was chosen)
                curved_line = list(street['geometry'].coords)
                u_node = road_network.nodes[u]
                v_node = road_network.nodes[v]
                if curved_line[0][0] == u_node['x'] and curved_line[0][1] == u_node['y']:
                    first_segment = (u_node, {'x': curved_line[1][0], 'y': curved_line[1][1]})
                    last_segment = ({'x': curved_line[-2][0], 'y': curved_line[-2][1]}, v_node)
                else:
                    first_segment = (u_node, {'x': curved_line[-2][0], 'y': curved_line[-2][1]})
                    last_segment = ({'x': curved_line[1][0], 'y': curved_line[1][1]}, v_node)

                u_street_sides = [[], []]
                for corner in u_filtered_corners:
                    if point_distance_from_line(first_segment, corner) > 0:
                        u_street_sides[1].append(corner)
                    else:
                        u_street_sides[0].append(corner)

                v_street_sides = [[], []]
                for corner in v_filtered_corners:
                    if point_distance_from_line(last_segment, corner) > 0:
                        v_street_sides[1].append(corner)
                    else:
                        v_street_sides[0].append(corner)

                # We look for the shortest possible trench on each side of the road separately
                for side_id in range(0, 2):
                    u_side_corners = u_street_sides[side_id]
                    v_side_corners = v_street_sides[side_id]
                    if len(u_side_corners) > 0 and len(v_side_corners) > 0:
                        curved_trench = get_trench_linestring(u_side_corners, v_side_corners, street,
                                                              distance_from_center_of_road, side_id)
                        new_curved_pp.append(curved_trench)
                    else:
                        print(f"Can't find side corners {street}")
        else:
            print(f"Warning: street_segment_id {street_segment_id} not in trench_corners")
    # If there are more than one possibility to have a trench on one side of the street segment,
    # Find the shortest one and make the others as invalid i.e. (None, None)
    # TODO: split by side of road and find shortest on ether side separately
    for street_segment_id, streets in point_edges.items():
        for trench_candidate_hash, sides in streets.items():
            for side in sides:
                if len(side) > 1:
                    shortest_pair = None
                    shortest_distance = 1000000
                    for trench_candidate in side:
                        if trench_candidate[0] is not None:
                            new_dist = node_distance(*trench_candidate)
                            if new_dist < shortest_distance:
                                if shortest_pair is not None:
                                    # Mark trench to not be used
                                    shortest_pair[0] = None
                                    shortest_pair[1] = None
                                shortest_pair = trench_candidate
                                shortest_distance = new_dist

    trenches: List[Trench] = list()
    # Add the trenches to the network
    for trench_candidate in new_pp:
        # (None, None) pairs were marked as invalid trenches above
        if trench_candidate[0] is not None:
            if isinstance(trench_candidate[2], str):
                street_names = {trench_candidate[2]}
            else:
                street_names = set(trench_candidate[2])
            trenches.append(Trench(u_for_edge=trench_candidate[0]['node_for_adding'],
                                   v_for_edge=trench_candidate[1]['node_for_adding'],
                                   name=f"trench {trench_candidate[2]}",
                                   street_names=street_names,
                                   length=node_distance(trench_candidate[0], trench_candidate[1])))

    # Add the curved trenches to the network
    for curved_trench in new_curved_pp:
        trenches.append(Trench(**curved_trench))

    # Add the crossings, trenches connecting corners around an intersection
    for street_segment_id, crossings in road_crossing.items():
        for crossing in crossings:
            if isinstance(crossing, Trench):
                trenches.append(crossing)

    # Workaround to remove duplicate trenches
    seen = set()
    for trench in trenches:
        if trench not in seen:
            seen.add(trench)
    trenches = list(seen)

    # Get the building trench info objects
    building_by_closest_trench = get_building_by_closest_trench(building_gdf, trench_corners, trenches)

    # Get new road trenches that are connected to the building trenches
    new_trench_corners, new_trenches,  trench_indexes_to_remove, building_trenches_lookup = \
        get_sub_trenches_for_buildings(
        building_by_closest_trench, trenches, trench_corners)

    # new_trench_corners has different keys than what is currently in trench_corners
    # So we can safely add them to the trench corner dict
    for street_index, corner in new_trench_corners.items():
        trench_corners[street_index] = corner

    # Add the road sub-trenches that are connected to the building trenches and the building trenches
    for sub_trenches in new_trenches:
        trenches.append(sub_trenches)

    # Remove the original road trenches that have been replaced by the sub-trenches
    trench_indexes_to_remove.sort(reverse=True)
    for trench_index in trench_indexes_to_remove:
        del trenches[trench_index]

    corner_by_id: Dict[int, TrenchCorner] = dict()
    for intersection_osmid, corners in trench_corners.items():
        for corner in corners:
            corner_by_id[corner['node_for_adding']] = corner

    return TrenchNetwork(trench_corners, trenches, building_trenches_lookup, corner_by_id)


def get_trench_to_network_graph(trench_network: TrenchNetwork,
                                road_network: networkx.MultiDiGraph) -> networkx.MultiDiGraph:
    """
    Adds the trenches and nodes in the trench_network to an existing OSM Network
    :param trench_network: The Trench network
    :param road_network: The OSM street MultiDiGraph
    :return: A combined MultiDiGraph
    """
    # Add trench corner nodes to network
    building_fiber_graph = ox.graph_from_gdfs(gpd.GeoDataFrame(columns=["x", "y"]), gpd.GeoDataFrame(), graph_attrs=road_network.graph)

    for intersection_osmid, corners in trench_network.trenchCorners.items():
        for corner in corners:
            # TODO: addes nodes more then ones, but it should be ok since they have the same ID
            building_fiber_graph.add_node(**corner)

    # Add the trenches to the network
    osmid = 8945376
    for trench in trench_network.trenches:
        osmid += 1
        building_fiber_graph.add_edge(**trench, key=1, osmid=osmid)

    return building_fiber_graph


if __name__ == "__main__":
    box = (50.843217, 50.833949, 4.439903, 4.461962)
    g_box = ox.graph_from_bbox(*box,
                               network_type='drive',
                               simplify=False,
                               retain_all=False,
                               truncate_by_edge=True)
    building_gdf = ox.geometries_from_bbox(*box, tags={'building': True})
    trench_network = get_trench_network(g_box, building_gdf)

    trench_graph = get_trench_to_network_graph(trench_network, g_box)

    ec = ['black' if 'highway' in d else
          'red' for _, _, _, d in g_box.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(g_box, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False)

    ec = ["grey" if "trench_crossing" in d and d["trench_crossing"]else
          "blue" if "house_trench" in d else
          'red' for _, _, _, d in trench_graph.edges(keys=True, data=True)]
    fig, ax = ox.plot_graph(trench_graph, bgcolor='white', edge_color=ec,
                            node_size=0, edge_linewidth=0.5,
                            show=False, close=False, ax=ax)
    ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
    plt.show()
