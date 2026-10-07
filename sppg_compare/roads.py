"""Directed OpenStreetMap road primitives and fastest-path routing."""
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class RoadNode:
    id: str
    x: float
    y: float


@dataclass(frozen=True)
class RoadEdge:
    source: str
    target: str
    speed_kph: float
    one_way: bool = False
    road_class: str = ""
    name: str = ""
    speed_source: str = "custom"


def route_tables(nodes, edges, kitchen_nodes, school_nodes, metric="travel_time"):
    if metric not in ("travel_time", "distance"):
        raise ValueError("Unknown shortest-path metric.")
    lookup = {node.id: node for node in nodes}
    if len(lookup) != len(nodes) or any(not math.isfinite(v) for n in nodes for v in (n.x, n.y)):
        raise ValueError("Road nodes need unique IDs and finite coordinates.")
    graph = {key: [] for key in lookup}
    for edge in edges:
        if edge.source not in lookup or edge.target not in lookup or not math.isfinite(edge.speed_kph) or edge.speed_kph <= 0:
            raise ValueError("Road edges need valid endpoints and positive finite speeds.")
        a, b = lookup[edge.source], lookup[edge.target]
        distance = math.hypot(a.x-b.x, a.y-b.y)
        directions = ((a.id, b.id),) if edge.one_way else ((a.id, b.id), (b.id, a.id))
        for source, target in directions:
            graph[source].append((target, distance * 60 / edge.speed_kph, distance))
    if any(key not in lookup for key in (*kitchen_nodes, *school_nodes)):
        raise ValueError("Every kitchen and school must connect to a road node.")
    tables = []
    for source in kitchen_nodes:
        best = {source: (0., 0.)}
        route_values = {source: (0., 0.)}
        previous = {}
        queue = [(0., 0., source)]
        while queue:
            primary, secondary, node = heapq.heappop(queue)
            if best[node] != (primary, secondary):
                continue
            time, distance = route_values[node]
            for target, minutes, km in graph[node]:
                actual = (time + minutes, distance + km)
                candidate = actual if metric == "travel_time" else (actual[1], actual[0])
                if candidate < best.get(target, (math.inf, math.inf)):
                    best[target] = candidate
                    route_values[target] = actual
                    previous[target] = node
                    heapq.heappush(queue, (*candidate, target))
        rows = []
        for target in school_nodes:
            if target not in best:
                raise ValueError("Road network must connect every kitchen to every school.")
            path = [target]
            while path[-1] != source:
                path.append(previous[path[-1]])
            rows.append((*route_values[target], tuple(reversed(path))))
        tables.append(rows)
    return tuple(tuple(tables[j][i] for j in range(len(kitchen_nodes)))
                 for i in range(len(school_nodes)))
