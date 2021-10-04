from typing import Dict, List, Tuple, Set

import numpy as np
import osmnx as ox
import itertools

import matplotlib.pyplot as plt
import math

from shapely.geometry import Point, LineString

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

def point_on_line(u, v, c, return_distance=False):
    a = np.array([u['x'], u['y']])
    b = np.array([v['x'], v['y']])
    p = np.array([c['x'], c['y']])
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
    dist = np.sum((p - result) ** 2)
    if return_distance:
        return result, dist
    else:
        return result



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


def get_trench_corners(network):
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
        street_segment_id: str = ""
        # Loop though the street vectors in a clockwise order (sorted_vs.sort())
        for radian in sorted_vs:
            v = neighbors[radian]
            streets = network.get_edge_data(u, v)
            s = [u, v]
            # we can get this segment twice to sorting the node ids make sure they have the same street_segment_id
            s.sort()
            street_segment_id = str(s)
            if len(streets) > 1:
                # This can happen if the GBox hacked the street into multiple segments, I think
                print("Crap len(streets) > 1")
                print(streets)

            # We need two vectors to find a trench corner between them
            if last_radian is not None:
                # Find a angle between the two other angles
                between_radian = radian - (abs(radian - last_radian) / 2)
                # Find a point on a circle with the radius of distance_from_center_of_road at that angle
                x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
                # Create a Trench Corner at that point
                node = TrenchCorner(x, y, 2, u, {street_segment_id, last_street_id})
                if street_segment_id not in output_trench_corners:
                    output_trench_corners[street_segment_id] = set()
                if node not in output_trench_corners[first_street_id] \
                        and node not in output_trench_corners[last_street_id]:
                    node_id += 1
                    node['node_for_adding'] = node_id
                    output_trench_corners[street_segment_id].add(node)
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
                first_street_id = street_segment_id
                if street_segment_id not in output_trench_corners:
                    output_trench_corners[street_segment_id] = set()
            last_radian = radian
            last_street_id = street_segment_id

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
            node1 = TrenchCorner(x, y, 2, u, {street_segment_id})
            if node1 not in output_trench_corners[first_street_id] \
                    and node1 not in output_trench_corners[last_street_id]:
                node_id += 1
                node1['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node1)
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()

            between_radian = first_radian + math.pi * 1.5
            x, y = point_on_circle(current_node, distance_from_center_of_road, between_radian)
            node2 = TrenchCorner(x, y, 2, u, {street_segment_id})
            if node2 not in output_trench_corners[first_street_id] \
                    and node2 not in output_trench_corners[last_street_id]:
                node_id += 1
                node2['node_for_adding'] = node_id
                output_trench_corners[first_street_id].add(node2)
                if first_street_id not in output_road_crossing:
                    output_road_crossing[first_street_id] = list()

            output_road_crossing[first_street_id].append((node1['node_for_adding'], node2['node_for_adding']))

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


def get_nearest_edge(G, point):
    """
    Return the nearest edge to a pair of coordinates. Pass in a graph and a tuple
    with the coordinates. We first get all the edges in the graph. Secondly we compute
    the euclidean distance from the coordinates to the segments determined by each edge.
    The last step is to sort the edge segments in ascending order based on the distance
    from the coordinates to the edge. In the end, the first element in the list of edges
    will be the closest edge that we will return as a tuple containing the shapely
    geometry and the u, v nodes.
    Parameters
    ----------
    G : networkx multidigraph
    point : tuple
        The (lat, lng) or (y, x) point for which we will find the nearest edge
        in the graph
    Returns
    -------
    closest_edge_to_point : tuple (shapely.geometry, u, v)
        A geometry object representing the segment and the coordinates of the two
        nodes that determine the edge section, u and v, the OSM ids of the nodes.
    """
    gdf = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
    graph_edges = gdf[["geometry", "u", "v"]].values.tolist()

    edges_with_distances = [
        (
            graph_edge,
            Point(tuple(reversed(point))).distance(graph_edge[0])
        )
        for graph_edge in graph_edges
    ]

    edges_with_distances = sorted(edges_with_distances, key=lambda x: x[1])
    closest_edge_to_point = edges_with_distances[0][0]

    geometry, u, v = closest_edge_to_point

    # log('Found nearest edge ({}) to point {} in {:,.2f} seconds'.format((u, v), point, time.time() - start_time))

    return u, v


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


G_box = ox.graph_from_bbox(50.78694, 50.77902, 4.48386, 4.49521,
                           network_type='drive',
                           simplify=True,
                           retain_all=False,
                           truncate_by_edge=True)

trench_corners, road_crossing = get_trench_corners(G_box)

for osmid, corners in trench_corners.items():
    for corner in corners:
        # TODO: addes nodes more then ones, but it should be ok since they have the same ID
        G_box.add_node(**corner)

new_edges = list()
new_pp = list()
point_edges = dict()
processed_streets = set()
for u, v, key, street in G_box.edges(keys=True, data=True):
    added_trenches = set()

    u_node = G_box.nodes[u]
    v_node = G_box.nodes[v]
    s = [u, v]
    s.sort()
    street_segment_id = str(s)
    # Check if this is a curved road
    if 'geometry' not in street or True:
        # Not a curved road

        # Mke sure we have trench corners that re on this street
        if street_segment_id in trench_corners:
            corners = trench_corners[street_segment_id]

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
        # A curved road
        pass

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
                        new_dist = node_distance(trench_candidate[0], trench_candidate[1])
                        if new_dist < shortest_distance:
                            if shortest_pair is not None:
                                # Mark trench to not be used
                                shortest_pair[0] = None
                                shortest_pair[1] = None
                            shortest_pair = trench_candidate
                            shortest_distance = new_dist

# Add the trenches to the network
for trench_candidate in new_pp:
    # (None, None) pairs were marked as invalid trenches above
    if trench_candidate[0] is not None:
        G_box.add_edge(u_for_edge=trench_candidate[0]['node_for_adding'],
                       v_for_edge=trench_candidate[1]['node_for_adding'],
                       key=1, osmid=8945376,
                       oneway=False,
                       name=f"trench {trench_candidate[2]}",
                       length=225.493)

# Add the crossings, trenches connecting corners around an intersection
for street_segment_id, crossings in road_crossing.items():
    for crossing in crossings:
        G_box.add_edge(u_for_edge=crossing[0],
                       v_for_edge=crossing[1],
                       key=1, osmid=8945376,
                       oneway=False,
                       name=f"trench {street_segment_id}",
                       length=225.493,
                       trench_crossing=True)

# Get buildings
building_gdf = ox.geometries_from_bbox(50.78694, 50.77902, 4.48586, 4.49721, tags={'building': True})

# Get Building centroids
# TODO: Add trenches from building centroid to nearest trench
building_centroids = list()
node_id = G_box.number_of_nodes()
for _, building in building_gdf.iterrows():
    street_name = building['addr:street']
    centroid = building['geometry'].centroid
    building_centroids.append([centroid.xy[0][0], centroid.xy[1][0]])
    node_id += 1
    u_id = node_id
    node_id += 1
    v_id = node_id
    tmp = False
    distance = float('inf')
    for u, v, key, street in G_box.edges(keys=True, data=True):
        if str(street_name) in street['name']:
            u_node = G_box.nodes[u]
            v_node = G_box.nodes[v]
            #a_point = Point(u_node['x'], u_node['y'])
            #b_point = Point(v_node['x'], v_node['y'])
            new_u_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0]}
            print(street_name)
            projected, new_distance = point_on_line(u_node, v_node, new_u_node, return_distance=True)
            if new_distance < distance:
                new_v_node = {'x': projected[0], 'y': projected[1]}
                distance = new_distance
                tmp = True
    if tmp:
        G_box.add_node(v_id, **new_v_node)
        G_box.add_node(u_id, **new_u_node)
        G_box.add_edge(u_for_edge=u_id,
                       v_for_edge=v_id,
                       key=1, osmid=8945376,
                       oneway=False,
                       name=f"trench {u}",
                       length=225.493)

# node_id += 1
# u = node_id
# node_id += 1
# v = node_id
# print(v)
# new_u_node = {'x': centroid.xy[0][0], 'y': centroid.xy[1][0]}
#
# pt_nearest_edge = edge = ox.distance.nearest_edges(G_box, centroid.xy[0][0], centroid.xy[1][0], interpolate=0.1)  # [1], [2] are start/end nodes of the nearest edge
#
# # project points onto neareset edge (a,b)
# o_point = Point(centroid.xy[0][0], centroid.xy[1][0])
# a_point = Point(G_box.nodes[pt_nearest_edge[0]]['x'], G_box.nodes[pt_nearest_edge[0]]['y'])
# b_point = Point(G_box.nodes[pt_nearest_edge[1]]['x'], G_box.nodes[pt_nearest_edge[1]]['y'])
# a_latl = (G_box.nodes[pt_nearest_edge[0]]['y'], G_box.nodes[pt_nearest_edge[0]]['x'])
# b_latl = (G_box.nodes[pt_nearest_edge[1]]['y'], G_box.nodes[pt_nearest_edge[1]]['x'])
# dist_ab = LineString([a_point, b_point]).project(o_point)
# projected_orig_point = list(LineString([a_point, b_point]).interpolate(dist_ab).coords)
# o1_latl = (projected_orig_point[0][1], projected_orig_point[0][0])
#
# print(o1_latl)
#
# new_v_node = {'x': o1_latl[0], 'y': o1_latl[1]}
#
# G_box.add_node(u, **new_u_node)
# G_box.add_node(v, **new_v_node)
#
# G_box.add_edge(u_for_edge=u,
#                v_for_edge=v,
#                key=1, osmid=8945376,
#                oneway=False,
#                name=f"trench {u}",
#                length=225.493)


# Give different things different colours
ec = ['y' if 'highway' in d else
      'gray' if 'trench_crossing' in d
      else 'r'
      for _, _, _, d in G_box.edges(keys=True, data=True)]

# Plot the network
fig, ax = ox.plot_graph(G_box, bgcolor='white', edge_color=ec,
                        node_size=0, edge_linewidth=0.5,
                        show=False, close=False)
# Plot the buildings
ox.plot_footprints(building_gdf, ax=ax, color="orange", alpha=0.5)
plt.show()
