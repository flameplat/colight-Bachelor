"""
Generates realistic OD-based traffic flows for a 6x6 CityFlow grid.

Each flow picks a random origin and destination intersection (boundary or inner),
computes the shortest path via BFS, and converts it to a road sequence.

Mix of trip types:
  - boundary → boundary (long cross-city trips)
  - boundary → inner    (drop-off inside city)
  - inner → boundary    (pick-up from inside city)
  - inner → inner       (short inner-city trips)
"""

import json
import random
from collections import deque

GRID_COLS = 6
GRID_ROWS = 6
SIM_DURATION = 14400

VEHICLE = {
    "length": 5.0, "width": 2.0,
    "maxPosAcc": 2.0, "maxNegAcc": 4.5,
    "usualPosAcc": 2.0, "usualNegAcc": 4.5,
    "minGap": 2.5, "maxSpeed": 11.111, "headwayTime": 2
}


def build_graph(net):
    """Build directed graph: node → list of (neighbor_node, road_id)."""
    graph = {}
    road_map = {}  # (from, to) → road_id

    for inter in net['intersections']:
        graph[inter['id']] = []

    for road in net['roads']:
        src = road['startIntersection']
        dst = road['endIntersection']
        graph[src].append((dst, road['id']))
        road_map[(src, dst)] = road['id']

    return graph, road_map


def bfs_shortest_path(graph, src, dst):
    """Return list of intersection IDs from src to dst, or None if unreachable."""
    if src == dst:
        return [src]
    visited = {src: None}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                visited[neighbor] = node
                if neighbor == dst:
                    path = []
                    cur = dst
                    while cur is not None:
                        path.append(cur)
                        cur = visited[cur]
                    return list(reversed(path))
                queue.append(neighbor)
    return None


def path_to_roads(path, road_map):
    """Convert intersection path to list of road IDs."""
    roads = []
    for i in range(len(path) - 1):
        key = (path[i], path[i + 1])
        if key not in road_map:
            return None
        roads.append(road_map[key])
    return roads


def make_flow(route, interval):
    return {
        "vehicle": VEHICLE,
        "route": route,
        "interval": float(interval),
        "startTime": 0,
        "endTime": SIM_DURATION
    }


def get_intersection_sets(net):
    """Return sets of real (inner) and virtual (boundary) intersection IDs."""
    real, virtual = [], []
    for inter in net['intersections']:
        if inter['virtual']:
            virtual.append(inter['id'])
        else:
            real.append(inter['id'])
    return real, virtual


if __name__ == "__main__":
    random.seed(42)

    net = json.load(open('data/template_lsr/6_6/roadnet_6_6.json'))
    graph, road_map = build_graph(net)
    real_nodes, virtual_nodes = get_intersection_sets(net)

    base_flows = json.load(open('data/template_lsr/6_6/anon_6_6_600_0.3_bi.json'))

    # Adjust base flow intervals to contribute 30,000 vehicles (maintaining EW:NS ratio)
    for f in base_flows:
        if f['interval'] == 6.0:
            f['interval'] = 7.5   # EW arterials
        elif f['interval'] == 20.0:
            f['interval'] = 25.0  # NS arterials

    # OD flows: 1,200 flows × 25 vehicles each = 30,000 vehicles
    # interval = 14400 / 25 = 576s
    OD_INTERVAL = 576.0

    trip_types = [
        ("boundary→boundary", virtual_nodes, virtual_nodes, 200, OD_INTERVAL),
        ("boundary→inner",    virtual_nodes, real_nodes,    300, OD_INTERVAL),
        ("inner→boundary",    real_nodes,    virtual_nodes, 300, OD_INTERVAL),
        ("inner→inner",       real_nodes,    real_nodes,    400, OD_INTERVAL),
    ]

    new_flows = []
    skipped = 0

    for label, origins, destinations, count, interval in trip_types:
        generated = 0
        attempts = 0
        while generated < count and attempts < count * 20:
            attempts += 1
            src = random.choice(origins)
            dst = random.choice(destinations)
            if src == dst:
                continue
            path = bfs_shortest_path(graph, src, dst)
            if path is None or len(path) < 2:
                skipped += 1
                continue
            roads = path_to_roads(path, road_map)
            if roads is None or len(roads) == 0:
                skipped += 1
                continue
            new_flows.append(make_flow(roads, interval))
            generated += 1
        print(f"{label:<25}: {generated} flows generated")

    all_flows = base_flows + new_flows

    # verify total vehicle count
    total_vehicles = sum(int(14400 / f['interval']) for f in all_flows)

    out = 'data/template_lsr/6_6/anon_6_6_900_0.3_turn.json'
    json.dump(all_flows, open(out, 'w'), indent=2)

    print(f"\nBase flows     : {len(base_flows)}")
    print(f"New OD flows   : {len(new_flows)}")
    print(f"Skipped        : {skipped}")
    print(f"Total flows    : {len(all_flows)}")
    print(f"Total vehicles : {total_vehicles}")
    print(f"Output         : {out}")
