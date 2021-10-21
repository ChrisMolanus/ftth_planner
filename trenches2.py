from typing import Dict, List, Tuple, Set

import geopandas
import networkx
import numpy as np
import osmnx as ox
import itertools

import matplotlib.pyplot as plt
import math

from shapely.geometry import Point, LineString, point

from shapely.geometry import LineString

distance_from_center_of_road = 0.0001


def point_distance_from_line(line: Tuple[dict, dict], point: dict) -> float:
    """
    The distance between a point and a line
    :param line: The line
    :param point: The point
    :return: The distance between the point and the line
    """
    return (((point['x'] - line[0]['x']) * (line[1]['y'] - line[0]['y']))
            - ((point['y'] - line[0]['y']) * (line[1]['x'] - line[0]['x'])))


def node_distance(node1: dict, node2: dict) -> float:
    """
    The distance between two points
    :param node1: A point
    :param node2: A point
    :return: The distance between the two points
    """
    return (((node2['x'] - node1['x']) ** 2) + ((node2['y'] - node1['y']) ** 2)) ** 0.5


def angle(vector1: Tuple[float, float], vector2: Tuple[float, float]) -> float:
    """
    Returns the clockwise angle between two vectors in radian
    :param vector1: A vector
    :param vector2: A vector
    :return: The angle between the two vectors
    """
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    if y2 < y1:
        return math.pi - math.acos(inner_product / (len1 * len2)) + math.pi
    else:
        return math.acos(inner_product / (len1 * len2))


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


def get_perpendicular_line(u_node, v_node, point) -> Tuple[dict, dict]:
    dx = u_node['x'] - v_node['x']
    dy = u_node['y'] - v_node['y']

    road_length = math.sqrt(dx ** 2 + dy ** 2)
    if road_length == 0:
        road_length = 0.00001
    t = distance_from_center_of_road / road_length

    # Perpendicular line
    dx1 = -1 * dy
    dy1 = dx

    return {'x': point['x'], 'y': point['y']}, {'x': point['x'] * dx1, 'y': point['y'] * dy1}


def point_on_line(u, v, c, return_distance=False):
    p1 = np.array([u['x'], u['y']])
    p2 = np.array([v['x'], v['y']])
    p3 = np.array([c['x'], c['y']])
    l2 = np.sum((p1 - p2) ** 2)
    t = np.sum((p3 - p1) * (p2 - p1)) / l2
    # if t > 1 or t < 0:
    #     print('p3 does not project onto p1-p2 line segment')

    # if you need the point to project on line segment between p1 and p2 or closest point of the line segment
    # t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))

    projection = p1 + t * (p2 - p1)

    # a = np.array([u['x'], u['y']])
    # b = np.array([v['x'], v['y']])
    # p = np.array([c['x'], c['y']])
    # ap = p - a
    # ab = b - a
    # result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    dist = np.sum((p3 - projection) ** 2)
    if return_distance:
        return projection, dist
    else:
        return projection


class TrenchCorner(dict):
    def __init__(self, x: float, y: float, trench_count: int, u_node_id: int, street_ids: Set, *args, **kw):
        """
        A FttH planner trench corner
        :param x: The OSMnx x coordinate of the node
        :param y: The OSMnx y coordinate of the node
        :param trench_count:
        :param u_node_id: The OSMnx node ID of the intersection this corner is on
        :param street_ids: A SET of the string- representation of the sorted list of node IDs
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

    def __cmp__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']

    def __hash__(self):
        return hash((self['x'], self['y']))

    def __eq__(self, other):
        return self['x'] == other['x'] and self['y'] == other['y']


class Trench(dict):
    def __init__(self, u_for_edge: int, v_for_edge: int, name: str, length: float, trench: bool = True,
                 trench_crossing: bool = False, geometry: LineString = None, *args, **kw):
        super(Trench, self).__init__(*args, **kw)
        self['u_for_edge'] = u_for_edge
        self['v_for_edge'] = v_for_edge
        self['name'] = name
        self['length'] = length
        self['trench'] = trench
        self['trench_crossing'] = trench_crossing
        if geometry is not None:
            self['geometry'] = geometry
            self.has_geometry = True
        else:
            self.has_geometry = False

    def has_geometry(self) -> bool:
        return self.has_geometry


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
                          street, distance_from_center_of_road: float, side_id: int) -> dict:
    """
    Returns a curved trench parallel to the road on one side of the road.
    :param u_side_corners: A set of trench corners around the first point in geometry of this road.
    :param v_side_corners: A set of trench corners around the last point in geometry of this road.
    :param street: The data of the street
    :param distance_from_center_of_road: The distance the trench should be from the road
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
                                                              distance_from_center_of_road, side_id)

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

    return {'u_for_edge': closest_u_for_trench,
            'v_for_edge': closest_v_for_trench,
            'geometry': LineString(linestring),
            'length': total_road_length,
            'name': f"Curved Trench {street['name']}"}


def get_trench_corners(network: networkx.MultiDiGraph) -> Tuple[Dict[str, Set[TrenchCorner]], Dict[str, List[Trench]]]:
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
                radian = angle((1.0, 0.0), (neighbor['x'] - current_node['x'], neighbor['y'] - current_node['y']))
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
            s = [u, v]
            # we can get this segment twice to sorting the node ids make sure they have the same street_segment_id
            s.sort()
            radian_street_segment_id = str(s)
            if len(streets) > 1:
                print("Warning len(streets) > 1, This can happen if the GBox hacked the street into multiple segments")
                print(streets)

            # We need two vectors to find a trench corner between them
            if last_radian is not None:
                # Find a angle between the two other angles
                between_radian = radian - (abs(radian - last_radian) / 2)
                # Find a point on a circle with the radius of distance_from_center_of_road at that angle
                x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
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
                    output_road_crossing[last_street_id].append((last_node_id, node_id))
                else:
                    first_node_id = node_id
                last_node_id = node_id
            else:
                first_radian = radian
                first_street_id = radian_street_segment_id
                if radian_street_segment_id not in output_trench_corners:
                    output_trench_corners[radian_street_segment_id] = set()
            last_radian = radian
            last_street_id = radian_street_segment_id

        # Now all we have left if to create a trench corner between the last vector and the first vector
        if len(sorted_vs) > 1:
            first_radian = first_radian + (2 * math.pi)
            between_radian = first_radian - (abs(first_radian - last_radian) / 2)
            x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
            node = TrenchCorner(x, y, 2, u, {first_street_id, last_street_id})
            if node not in output_trench_corners[first_street_id] and node not in output_trench_corners[last_street_id]:
                node_id += 1
                node['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node)
                output_trench_corners[last_street_id].add(node)
                if last_street_id not in output_road_crossing:
                    output_road_crossing[last_street_id] = list()
                output_road_crossing[last_street_id].append((last_node_id, node_id))
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()
                output_road_crossing[first_street_id].append((node_id, first_node_id))
        elif len(sorted_vs) == 1:
            # This is a Dead end road, there was only 1 neighbor
            # So we make a "T" shape with a road crossing trench at the top
            between_radian = first_radian + math.pi * 0.5
            x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
            node1 = TrenchCorner(x, y, 2, u, {radian_street_segment_id})
            if node1 not in output_trench_corners[first_street_id] \
                    and node1 not in output_trench_corners[last_street_id]:
                node_id += 1
                node1['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node1)
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()

            between_radian = first_radian + math.pi * 1.5
            x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
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
                                                                trench_crossing=True
                                                                ),
                                                         )

    return output_trench_corners, output_road_crossing


def is_between(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """
    Is point c between points a and b
    :param a: A point
    :param b: A point
    :param c: The point that might be between "a" and "b"
    :return: True is "c" is between "a" and "b" (or close enough for floating point precision)
    """
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    # # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 0.00000005:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
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
    def __init__(self, trench_corners: Dict[int, TrenchCorner], trenches: List[Trench]):
        self.trenchCorners = trench_corners
        self.trenches = trenches


def get_trench_network(road_network: networkx.MultiDiGraph,
                       building_gdf: geopandas.GeoDataFrame) -> TrenchNetwork:
    trench_corners, road_crossing = get_trench_corners(road_network)

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

                street_sides = [[], []]
                for corner in filtered_corners:
                    if point_distance_from_line((u_node, v_node), corner) > 0:
                        street_sides[1].append(corner)
                    else:
                        street_sides[0].append(corner)

                for side_id in range(0, len(street_sides)):
                    side_corners = street_sides[side_id]
                    # Create possible trench corner pairs but looking for all possible combinations of corner points
                    for point_pair1 in list(itertools.combinations(side_corners, 2)):
                        # Only consider corner point pairs of points on different intersections
                        # Otherwize they are on the same intersections and that is a road crossing
                        if point_pair1[0]['u'] != point_pair1[1]['u']:
                            # trench_candidate as to be a list because tuples are immutable,
                            # and we might invalidate it later whe we chose from the candidates
                            trench_candidate = [point_pair1[0], point_pair1[1], street['name']]

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

                for side_id in range(0, 2):
                    u_side_corners = u_street_sides[side_id]
                    v_side_corners = v_street_sides[side_id]
                    if len(u_side_corners) > 0 and len(v_side_corners) > 0:
                        curved_trench = get_trench_linestring(u_side_corners, v_side_corners, street,
                                                              distance_from_center_of_road, side_id)
                        new_curved_pp.append(curved_trench)
                    else:
                        print(f"Can't find side corners {street}")

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
            trenches.append(Trench(u_for_edge=trench_candidate[0]['node_for_adding'],
                                   v_for_edge=trench_candidate[1]['node_for_adding'],
                                   name=f"trench {trench_candidate[2]}",
                                   length=node_distance(trench_candidate[0], trench_candidate[1])))

    # Add the curved trenches to the network
    for curved_trench in new_curved_pp:
        trenches.append(Trench(**curved_trench))

    # Add the crossings, trenches connecting corners around an intersection
    for street_segment_id, crossings in road_crossing.items():
        for crossing in crossings:
            if isinstance(crossing, Trench):
                trenches.append(crossing)

    # Get Building centroids
    # TODO: Add trenches from building centroid to nearest trench
    building_centroids = list()
    node_id = 500000000
    for _, building in building_gdf.iterrows():
        street_name = building['addr:street']
        centroid = building['geometry'].centroid
        building_centroids.append([centroid.xy[0][0], centroid.xy[1][0]])
        node_id += 1
        u_id = node_id
        node_id += 1
        v_id = node_id
        distance = float('inf')
        new_u_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0]}
        for trench in trenches:
            if str(street_name) in trench['name']:
                if 'geometry' not in trench:
                    for intersection_osmidR, corners in trench_corners.items():
                        for corner in corners:
                            if corner['node_for_adding'] == trench['u_for_edge']:
                                u_node = corner
                            if corner['node_for_adding'] == trench['v_for_edge']:
                                v_node = corner
                    projected, new_distance = point_on_line(u_node, v_node, new_u_node, return_distance=True)
                    if new_distance < distance:
                        new_v_node = {'x': projected[0], 'y': projected[1]}
                        distance = new_distance
                    if distance != float('inf'):
                        # trenches.remove(trench)
                        # road_network.add_node(u_id, **new_u_node)
                        # road_network.add_node(v_id, **new_v_node)
                        # trenches.append(Trench(u_for_edge=trench['u_for_edge'],
                        #                        v_for_edge=v_id,
                        #                        name=f"trench {street_name}",
                        #                        length=node_distance(trench_candidate[0], trench_candidate[1])))
                        # trenches.append(Trench(u_for_edge=v_id,
                        #                        v_for_edge=trench['v_for_edge'],
                        #                        name=f"trench {street_name}",
                        #                        length=node_distance(trench_candidate[0], trench_candidate[1])))

                        # # trench_corners[v_id] = new_v_node
                        road_network.add_node(u_id, **new_u_node)
                        road_network.add_node(v_id, **new_v_node)
                        road_network.add_edge(u_for_edge=u_id,
                                              v_for_edge=v_id,
                                              name=f"trench {u_id}")
                else:
                    for intersection_osmidR, corners in trench_corners.items():
                        for corner in corners:
                            if corner['node_for_adding'] == trench['u_for_edge']['node_for_adding']:
                                u_node = corner
                            if corner['node_for_adding'] == trench['v_for_edge']['node_for_adding']:
                                v_node = corner
                    #projection on curved road
                    last_node = ''
                    # for sub_x, sub_y in trench['geometry'].coords:
                    coords = list(trench['geometry'].coords)
                    shortest_i = None
                    for i in range(0, len(coords)):
                        sub_x, sub_y = coords[i]
                        if last_node == '':
                            last_node = {'x': sub_x, 'y': sub_y}
                        else:
                            sub_u_node = {'x': sub_x, 'y': sub_y}
                            # projected, new_distance = point_on_line(sub_u_node, last_node, new_u_node, return_distance=True)
                            perpendicular_line = get_perpendicular_line(last_node, sub_u_node, new_u_node)
                            projected = get_intersection_point2(perpendicular_line, (last_node, sub_u_node))
                            new_distance = node_distance(projected, new_u_node)
                            last_node = sub_u_node
                            if new_distance < distance:
                                new_v_node = projected
                                distance = new_distance
                                shortest_i = i
                    if shortest_i is not None:
                        # trench_corners[v_id] = new_v_node
                        coords.insert(shortest_i, (new_v_node['x'], new_v_node['y']))
                        # trench['geometry'] = LineString(coords)

                        road_network.add_node(u_id, **new_u_node)
                        road_network.add_node(v_id, **new_v_node)
                        road_network.add_edge(u_for_edge=u_id,
                                              v_for_edge=v_id,
                                              name=f"trench {u_id}")

    return TrenchNetwork(trench_corners, trenches), road_network
