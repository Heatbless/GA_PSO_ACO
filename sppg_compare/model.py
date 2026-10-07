"""Shared assignment model, validation, scoring, and deterministic repair."""

from dataclasses import asdict, dataclass, field
import math
import random
import hashlib
import json
import copy
from functools import cached_property, lru_cache
from .roads import RoadNode, RoadEdge, route_tables


@dataclass(frozen=True)
class Kitchen:
    id: str
    x: float
    y: float
    capacity: int


@dataclass(frozen=True)
class School:
    id: str
    x: float
    y: float
    demand: int


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    radius: float
    kitchens: tuple[Kitchen, ...]
    schools: tuple[School, ...]
    capacity_slack: float | None = None
    difficulty: str = "Custom"
    generator_version: str = "custom"
    road_nodes: tuple[RoadNode, ...] = ()
    road_edges: tuple[RoadEdge, ...] = ()
    kitchen_nodes: tuple[str, ...] = ()
    school_nodes: tuple[str, ...] = ()
    objective_metric: str = "distance"
    road_source: dict = field(default_factory=dict, hash=False)
    max_active_kitchens: int | None = None
    routing_mode: str = "allocation"
    unloading_minutes: float = 5.0
    opening_weight: float = 10.0
    distance_weight: float = 1.0
    school_routes: tuple = field(init=False, repr=False)
    return_routes: tuple = field(init=False, repr=False)
    costs: tuple = field(init=False, repr=False)
    travel_times: tuple = field(init=False, repr=False)
    road_distances: tuple = field(init=False, repr=False)
    paths: tuple = field(init=False, repr=False)
    distances: tuple[tuple[float, ...], ...] = field(init=False, repr=False)
    eligible: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "road_source", copy.deepcopy(self.road_source))
        if any(not math.isfinite(w) or w < 0 for w in (self.opening_weight, self.distance_weight)):
            raise ValueError("Objective weights must be finite and nonnegative.")
        if self.objective_metric == "facility_distance" and self.opening_weight == self.distance_weight == 0:
            raise ValueError("At least one objective weight must be positive.")
        if self.routing_mode not in ("allocation", "multi_stop") or not math.isfinite(self.unloading_minutes) or self.unloading_minutes < 0:
            raise ValueError("Invalid routing mode or unloading time.")
        if self.routing_mode == "multi_stop" and (not self.road_nodes or self.objective_metric != "travel_time"):
            raise ValueError("Multi-stop routing requires a road graph and travel-time objective.")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Service radius must be a positive finite number.")
        if not self.kitchens or not self.schools:
            raise ValueError("A scenario needs kitchens and schools.")
        if self.max_active_kitchens is not None and (type(self.max_active_kitchens) is not int or
                not 1 <= self.max_active_kitchens <= len(self.kitchens)):
            raise ValueError("Active SPPG limit must be between one and the candidate count.")
        for items, quantity in ((self.kitchens, "capacity"), (self.schools, "demand")):
            if len({item.id for item in items}) != len(items):
                raise ValueError("Kitchen IDs and school IDs must each be unique.")
            for item in items:
                if not all(math.isfinite(v) for v in (item.x, item.y)):
                    raise ValueError("Coordinates must be finite.")
                value = getattr(item, quantity)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"Every {quantity} must be a positive integer.")
        distances = tuple(tuple(math.hypot(s.x - k.x, s.y - k.y)
                                for k in self.kitchens) for s in self.schools)
        eligible = tuple(tuple(j for j, distance in enumerate(row)
                               if distance <= self.radius + 1e-9) for row in distances)
        if any(not row for row in eligible):
            missing = ", ".join(s.id for s, row in zip(self.schools, eligible) if not row)
            raise ValueError(f"Schools outside every kitchen's service radius: {missing}")
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "eligible", eligible)
        if self.objective_metric not in ("distance", "travel_time", "facility_distance"):
            raise ValueError("Objective metric must be distance or travel_time.")
        if self.road_nodes:
            if len(self.kitchen_nodes) != len(self.kitchens) or len(self.school_nodes) != len(self.schools):
                raise ValueError("Road endpoint bindings must match the number of sites.")
            lookup = {n.id: n for n in self.road_nodes}
            for sites, bindings in ((self.kitchens, self.kitchen_nodes), (self.schools, self.school_nodes)):
                for site, key in zip(sites, bindings):
                    if key not in lookup or math.hypot(site.x-lookup[key].x, site.y-lookup[key].y) > 1e-9:
                        raise ValueError("Road endpoints must coincide with their sites.")
            routes = route_tables(self.road_nodes, self.road_edges, self.kitchen_nodes, self.school_nodes,
                                  metric="distance" if self.objective_metric == "facility_distance" else "travel_time")
            times = tuple(tuple(r[0] for r in row) for row in routes)
            road_distances = tuple(tuple(r[1] for r in row) for row in routes)
            paths = tuple(tuple(r[2] for r in row) for row in routes)
        else:
            if self.objective_metric == "travel_time" or self.road_edges or self.kitchen_nodes or self.school_nodes:
                raise ValueError("Travel-time scenarios require an explicit road network.")
            times, road_distances, paths = (), distances, ()
        object.__setattr__(self, "travel_times", times)
        object.__setattr__(self, "road_distances", road_distances)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "costs", times if self.objective_metric == "travel_time" else road_distances)
        if self.routing_mode == "multi_stop":
            connections = route_tables(self.road_nodes, self.road_edges, self.school_nodes,
                                       self.school_nodes + self.kitchen_nodes)
            object.__setattr__(self, "school_routes", connections[:len(self.schools)])
            object.__setattr__(self, "return_routes", connections[len(self.schools):])
        else:
            object.__setattr__(self, "school_routes", ())
            object.__setattr__(self, "return_routes", ())

    @property
    def objective_unit(self):
        if self.objective_metric == "facility_distance":
            return "weighted score"
        return "portion-min" if self.objective_metric == "travel_time" else "portion-km"

    @property
    def total_demand(self):
        return sum(s.demand for s in self.schools)

    @cached_property
    def instance_id(self):
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        return f"{self.name}-{self.seed}-{digest}"

    @property
    def school_order(self):
        return tuple(sorted(range(len(self.schools)),
                            key=lambda i: (-self.schools[i].demand, i)))

    def to_dict(self):
        return {"name": self.name, "seed": self.seed, "radius": self.radius,
                "kitchens": [asdict(k) for k in self.kitchens],
                "schools": [asdict(s) for s in self.schools],
                "capacity_slack": self.capacity_slack, "difficulty": self.difficulty,
                "generator_version": self.generator_version,
                "objective_metric": self.objective_metric,
                "road_nodes": [asdict(n) for n in self.road_nodes],
                "road_edges": [asdict(e) for e in self.road_edges],
                "kitchen_nodes": list(self.kitchen_nodes), "school_nodes": list(self.school_nodes),
                "road_source": copy.deepcopy(self.road_source),
                "max_active_kitchens": self.max_active_kitchens,
                "routing_mode": self.routing_mode, "unloading_minutes": self.unloading_minutes,
                "opening_weight": self.opening_weight, "distance_weight": self.distance_weight}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["seed"], data["radius"],
                   tuple(Kitchen(**k) for k in data["kitchens"]),
                   tuple(School(**s) for s in data["schools"]),
                   data.get("capacity_slack"), data.get("difficulty", "Custom"),
                   data.get("generator_version", "legacy-v1"),
                   tuple(RoadNode(**n) for n in data.get("road_nodes", [])),
                   tuple(RoadEdge(**e) for e in data.get("road_edges", [])),
                   tuple(data.get("kitchen_nodes", [])), tuple(data.get("school_nodes", [])),
                   data.get("objective_metric", "distance"), data.get("road_source", {}),
                   data.get("max_active_kitchens"), data.get("routing_mode", "allocation"),
                   data.get("unloading_minutes", 5.0), data.get("opening_weight", 10.), data.get("distance_weight", 1.))


@dataclass(frozen=True)
class Evaluation:
    assignment: tuple[int, ...]
    loads: tuple[int, ...]
    objective: float
    overflow: int
    radius_violation: int
    active_site_violation: int = 0
    visit_order: tuple[int, ...] = ()
    school_arrivals: tuple[float, ...] = ()
    route_durations: tuple[float, ...] = ()

    @property
    def feasible(self):
        return self.overflow == 0 and self.radius_violation == 0 and self.active_site_violation == 0

    @property
    def rank(self):
        return self.radius_violation, self.active_site_violation, self.overflow, self.objective


def evaluate(scenario, assignment, visit_order=None):
    assignment = tuple(assignment)
    if len(assignment) != len(scenario.schools):
        raise ValueError("Each school needs exactly one assignment.")
    loads = [0] * len(scenario.kitchens)
    objective = 0.0
    radius_violation = 0
    for i, kitchen in enumerate(assignment):
        if type(kitchen) is not int or not 0 <= kitchen < len(loads):
            raise ValueError("Assignments must contain valid integer kitchen indices.")
        demand = scenario.schools[i].demand
        loads[kitchen] += demand
        objective += demand * scenario.costs[i][kitchen]
        if kitchen not in scenario.eligible[i]:
            radius_violation += demand
    overflow = sum(max(0, load - k.capacity)
                   for load, k in zip(loads, scenario.kitchens))
    excess = max(0, len(set(assignment)) - (scenario.max_active_kitchens or len(loads)))
    if scenario.objective_metric == "facility_distance":
        objective = scenario.opening_weight*len(set(assignment)) + scenario.distance_weight*sum(
            scenario.road_distances[i][j] for i, j in enumerate(assignment))
    arrivals, durations, order = (), (), ()
    if scenario.routing_mode == "multi_stop":
        order = tuple(scenario.school_order if visit_order is None else visit_order)
        if len(order) != len(assignment) or any(type(i) is not int for i in order) or set(order) != set(range(len(assignment))):
            raise ValueError("Visit order must be a permutation of all school indices.")
        arrivals, durations = [0.]*len(assignment), [0.]*len(loads)
        for j in range(len(loads)):
            stops = [i for i in order if assignment[i] == j]
            elapsed, previous = 0., None
            for i in stops:
                elapsed += scenario.travel_times[i][j] if previous is None else scenario.school_routes[i][previous][0]
                arrivals[i] = elapsed
                elapsed += scenario.unloading_minutes
                previous = i
            if stops:
                elapsed += scenario.return_routes[j][stops[-1]][0]
                durations[j] = elapsed
        objective = sum(s.demand*t for s, t in zip(scenario.schools, arrivals))
    return Evaluation(assignment, tuple(loads), objective, overflow, radius_violation, excess,
                      order, tuple(arrivals), tuple(durations))


def repair(scenario, assignment, initial=None, visit_order=None):
    """Project excess sites, then relocate overloads without opening excess sites."""
    assignment = list(assignment)
    current = initial if initial is not None else evaluate(scenario, assignment, visit_order)
    order = current.visit_order or visit_order
    if current.active_site_violation:
        # Close one used candidate at a time. Every algorithm uses this same
        # deterministic projection; evaluate every raw candidate in diagnostics.
        while len(set(assignment)) > scenario.max_active_kitchens:
            alternatives = []
            active = set(assignment)
            for closing in sorted(active):
                trial = list(assignment)
                loads = list(evaluate(scenario, trial, order).loads)
                possible = True
                for i in scenario.school_order:
                    if trial[i] != closing:
                        continue
                    choices = [j for j in scenario.eligible[i] if j in active and j != closing]
                    if not choices:
                        possible = False
                        break
                    demand = scenario.schools[i].demand
                    target = min(choices, key=lambda j: (max(0, loads[j]+demand-scenario.kitchens[j].capacity)
                                  - max(0, loads[j]-scenario.kitchens[j].capacity), scenario.costs[i][j], j))
                    loads[closing] -= demand
                    loads[target] += demand
                    trial[i] = target
                if possible:
                    alternatives.append((evaluate(scenario, trial, order).rank, tuple(trial)))
            if not alternatives:
                break
            assignment = list(min(alternatives)[1])
        current = evaluate(scenario, assignment, order)
    if current.overflow == 0:
        return tuple(assignment)
    loads = list(current.loads)
    while True:
        best_move = None
        for i, source in enumerate(assignment):
            if loads[source] <= scenario.kitchens[source].capacity:
                continue
            demand = scenario.schools[i].demand
            for target in scenario.eligible[i]:
                if target == source or loads[target] + demand > scenario.kitchens[target].capacity:
                    continue
                if scenario.max_active_kitchens and loads[target] == 0 and loads[source] != demand and sum(v > 0 for v in loads) >= scenario.max_active_kitchens:
                    continue
                delta = demand * (scenario.costs[i][target] - scenario.costs[i][source])
                move = (delta, i, target)
                if best_move is None or move < best_move:
                    best_move = move
        if best_move is None:
            return tuple(assignment)
        _, i, target = best_move
        source = assignment[i]
        loads[source] -= scenario.schools[i].demand
        loads[target] += scenario.schools[i].demand
        assignment[i] = target


def delivery_metrics(scenario, score, include_routes=True):
    """Report one-way independent deliveries, separate from optimizer runtime."""
    weighted_km = sum(s.demand * scenario.road_distances[i][j]
                      for i, (s, j) in enumerate(zip(scenario.schools, score.assignment)))
    if scenario.routing_mode == "multi_stop":
        weighted_km = 0.
        for j in set(score.assignment):
            elapsed_km, previous = 0., None
            for i in score.visit_order:
                if score.assignment[i] != j:
                    continue
                elapsed_km += scenario.road_distances[i][j] if previous is None else scenario.school_routes[i][previous][1]
                weighted_km += scenario.schools[i].demand*elapsed_km
                previous = i
    times = list(score.school_arrivals) if scenario.routing_mode == "multi_stop" else [scenario.travel_times[i][j] for i, j in enumerate(score.assignment)] if scenario.travel_times else []
    data = {"objective_metric": scenario.objective_metric, "objective_unit": scenario.objective_unit,
            "total_allocation_distance_km": sum(scenario.road_distances[i][j] for i, j in enumerate(score.assignment)) if score.feasible else None,
            "opening_weight": scenario.opening_weight, "distance_weight": scenario.distance_weight,
            "capacity_count_lower_bound": math.ceil(scenario.total_demand/max(k.capacity for k in scenario.kitchens)),
            "maximum_schools_per_sppg": [k.capacity//min(s.demand for s in scenario.schools) for k in scenario.kitchens],
            "candidate_sppg_count": len(scenario.kitchens),
            "max_active_sppg": scenario.max_active_kitchens,
            "active_sppg_count": len(set(score.assignment)),
            "selected_sppg_ids": [k.id for k, load in zip(scenario.kitchens, score.loads) if load],
            "active_site_violation": score.active_site_violation,
            "routing_mode": scenario.routing_mode,
            "total_tour_minutes": sum(score.route_durations) if score.feasible and score.route_durations else None,
            "maximum_tour_minutes": max(score.route_durations) if score.feasible and score.route_durations else None,
            "selected_capacity": sum(k.capacity for k, load in zip(scenario.kitchens, score.loads) if load),
            "sppg_load_utilization": [{"id": k.id, "selected": load > 0, "load": load,
                                       "school_count": score.assignment.count(j),
                                       "capacity": k.capacity, "utilization": load/k.capacity}
                                      for j, (k, load) in enumerate(zip(scenario.kitchens, score.loads))],
            "weighted_average_km": weighted_km / scenario.total_demand if score.feasible else None,
            "weighted_average_minutes": sum(s.demand * t for s, t in zip(scenario.schools, times)) / scenario.total_demand if times and score.feasible else None,
            "mean_school_eta_minutes": sum(times) / len(times) if times and score.feasible else None,
            "max_school_eta_minutes": max(times) if times and score.feasible else None}
    if include_routes:
        data["deliveries"] = [{"school": s.id, "kitchen": scenario.kitchens[j].id,
                               "demand": s.demand, "road_km": scenario.road_distances[i][j],
                               "eta_minutes": times[i] if times else None,
                               "path": list(scenario.paths[i][j]) if scenario.paths else []}
                              for i, (s, j) in enumerate(zip(scenario.schools, score.assignment))]
        data["vehicle_tours"] = vehicle_tours(scenario, score) if scenario.routing_mode == "multi_stop" else []
        for tour in data["vehicle_tours"]:
            cumulative_km = 0.
            for position, leg in enumerate(tour["legs"], 1):
                i = leg["school_index"]
                if i is None:
                    continue
                cumulative_km += leg["road_km"]
                data["deliveries"][i].update(path=leg["path"], path_origin=leg["from"],
                                             road_km=cumulative_km, incoming_leg_km=leg["road_km"],
                                             leg_travel_minutes=leg["travel_minutes"], stop_number=position)
    return data


def vehicle_tours(scenario, score):
    """One directed road tour per active SPPG, including unloading and return."""
    tours = []
    for j, kitchen in enumerate(scenario.kitchens):
        stops = [i for i in score.visit_order if score.assignment[i] == j]
        if not stops:
            continue
        legs, elapsed, previous = [], 0., None
        for i in stops:
            t, km, path = ((scenario.travel_times[i][j], scenario.road_distances[i][j], scenario.paths[i][j])
                           if previous is None else scenario.school_routes[i][previous])
            legs.append({"from": kitchen.id if previous is None else scenario.schools[previous].id,
                         "to": scenario.schools[i].id, "school_index": i, "departure_minutes": elapsed,
                         "arrival_minutes": elapsed+t, "travel_minutes": t, "road_km": km,
                         "path": list(path), "unloading_minutes": scenario.unloading_minutes})
            elapsed += t+scenario.unloading_minutes
            previous = i
        t, km, path = scenario.return_routes[j][previous]
        legs.append({"from": scenario.schools[previous].id, "to": kitchen.id, "school_index": None,
                     "departure_minutes": elapsed, "arrival_minutes": elapsed+t,
                     "travel_minutes": t, "road_km": km, "path": list(path), "unloading_minutes": 0.})
        tours.append({"kitchen": kitchen.id, "kitchen_index": j, "vehicle_capacity": kitchen.capacity,
                      "initial_load": score.loads[j], "school_order": stops,
                      "duration_minutes": elapsed+t, "legs": legs})
    return tours


def greedy_baseline(scenario):
    loads = [0] * len(scenario.kitchens)
    assignment = [0] * len(scenario.schools)
    for i in scenario.school_order:
        demand = scenario.schools[i].demand
        fits = [j for j in scenario.eligible[i]
                if loads[j] + demand <= scenario.kitchens[j].capacity]
        choices = fits or list(scenario.eligible[i])
        j = min(choices, key=lambda j: (scenario.costs[i][j], j))
        assignment[i] = j
        loads[j] += demand
    return evaluate(scenario, repair(scenario, assignment))


SIZES = {"Small": (3, None, 101), "Medium": (5, None, 202), "Large": (8, None, 303)}
DIFFICULTIES = {"Easy": 0.25, "Standard": 0.05, "Tight": 0.0, "Overloaded": -0.05}


@lru_cache(maxsize=12)
def _fixed_reference(size, network):
    from .osm import generate_osm_scenario
    return generate_osm_scenario(size, SIZES[size][2], 6.0, .05, "Standard", "Ciledug", network=network)


def generate_synthetic_schools(size, seed, radius, count, school_seed, demand, capacity,
                              opening_weight, distance_weight, network):
    """Nested, fixed synthetic schools on real roads; independent candidate-site seed."""
    from .basemap import VIEW_BBOX, VIEW_URL
    for label, value in (("School count", count), ("Meals per school", demand), ("SPPG capacity", capacity)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
    if count > 500:
        raise ValueError("School count must not exceed 500.")
    if type(school_seed) is not int:
        raise ValueError("School location seed must be an integer.")
    lat0, lon0 = network.metadata["projection_origin_lat_lon"]
    def inside(node):
        lat = lat0 + math.degrees(node.y/6371.0088)
        lon = lon0 + math.degrees(node.x/(6371.0088*math.cos(math.radians(lat0))))
        return VIEW_BBOX[0] <= lat <= VIEW_BBOX[2] and VIEW_BBOX[1] <= lon <= VIEW_BBOX[3]
    pool = sorted((node for node in network.nodes if inside(node)), key=lambda node: node.id)
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} road locations in this area; reduce school count.")
    random.Random(school_seed).shuffle(pool)
    school_nodes = pool[:count]
    schools = tuple(School(f"S{i+1}", node.x, node.y, demand) for i, node in enumerate(school_nodes))
    candidate_pool = sorted(pool[count:], key=lambda node: node.id)
    candidate_count = 2*SIZES[size][0]
    if len(candidate_pool) < candidate_count:
        raise ValueError("Too few separate road locations for the candidate SPPGs.")
    sites = random.Random(seed).sample(candidate_pool, candidate_count)
    kitchens = tuple(Kitchen(f"K{i+1}", node.x, node.y, capacity) for i, node in enumerate(sites))
    source = dict(network.metadata)
    source.update(focus_bbox_south_west_north_east=list(VIEW_BBOX), focus_osm_url=VIEW_URL,
                  school_locations="Synthetic fixed road-node locations; not real schools",
                  school_location_seed=school_seed, school_demand=demand,
                  school_features=[{"school_id": school.id, "name": f"Synthetic school {i+1}"}
                                   for i, school in enumerate(schools)],
                  scenario_seed_role="candidate SPPG sites only; school seed independent",
                  distance_definition="sum of one-way shortest road distances SPPG to assigned school; not vehicle mileage",
                  capacity_model="fixed editable meals per SPPG; all schools served indivisibly",
                  objective_formula="opening_weight * active_sppg_count + distance_weight * total_allocation_distance_km")
    return Scenario(size, seed, radius, kitchens, schools, None, "Fixed capacity",
                    "ciledug-synthetic-facility-v4", network.nodes, network.edges,
                    tuple(n.id for n in sites), tuple(n.id for n in school_nodes),
                    "facility_distance", source, None, "allocation", 5., opening_weight, distance_weight)


def generate_scenario(size="Small", seed=None, radius=6.0, capacity_slack=None, difficulty="Standard", *,
                      map_source="OpenStreetMap", osm_city="Ciledug", school_count=None, school_seed=101,
                      school_demand=200, sppg_capacity=3000, opening_weight=10., distance_weight=1.):
    """Fixed schools/demands/roads; all 2*K candidate SPPG sites drawn from the seed.

    Every candidate has equal capacity, so opening at most K has the stated
    capacity slack (apart from integer rounding). School assignments implicitly
    select sites; unused candidates contribute no capacity to feasibility.
    """
    if size not in SIZES or difficulty not in DIFFICULTIES:
        raise ValueError("Unknown scenario size or difficulty.")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Service radius must be a positive finite number.")
    limit, _, default_seed = SIZES[size]
    seed = default_seed if seed is None else seed
    slack = DIFFICULTIES[difficulty] if capacity_slack is None else capacity_slack
    if not math.isfinite(slack) or slack <= -1:
        raise ValueError("Capacity slack must be finite and greater than -1.")
    if map_source != "OpenStreetMap" or osm_city != "Ciledug":
        raise ValueError("This application uses only OpenStreetMap Ciledug, Tangerang.")
    from .osm import load_network
    if school_count is not None:
        return generate_synthetic_schools(size, seed, radius, school_count, school_seed, school_demand,
                                          sppg_capacity, opening_weight, distance_weight, load_network("Ciledug", planning=True))
    reference = _fixed_reference(size, load_network("Ciledug"))
    capacity = max(1, math.ceil(reference.total_demand * (1 + slack) / limit))
    lookup = {n.id: n for n in reference.road_nodes}
    xs = [s.x for s in reference.schools]
    ys = [s.y for s in reference.schools]
    pad = 0.5
    pool = sorted((n for n in reference.road_nodes if n.id not in reference.school_nodes
                   and min(xs) - pad <= n.x <= max(xs) + pad and min(ys) - pad <= n.y <= max(ys) + pad),
                  key=lambda n: n.id)
    if len(pool) < 2 * limit:
        pool = sorted((n for n in reference.road_nodes if n.id not in reference.school_nodes),
                      key=lambda n: n.id)
    bindings = [n.id for n in random.Random(seed).sample(pool, 2 * limit)]
    kitchens = tuple(Kitchen(f"K{j+1}", lookup[key].x, lookup[key].y, capacity)
                     for j, key in enumerate(bindings))
    source = dict(reference.road_source)
    source.update(school_dataset_seed=101,
                  average_speed_kph={"main_road": 50., "local_street": 25.},
                  scenario_seed_role="dynamic candidate SPPG locations; schools, demands and roads fixed",
                  capacity_model="equal per-site capacity; slack applies to active-site limit, rounded up per site")
    level = difficulty if slack == DIFFICULTIES[difficulty] else "Custom"
    return Scenario(size, seed, radius, kitchens, reference.schools, slack, level,
                    "ciledug-all-schools-routing-v3", reference.road_nodes, reference.road_edges,
                    tuple(bindings), reference.school_nodes, "travel_time", source, limit, "multi_stop", 5.)
