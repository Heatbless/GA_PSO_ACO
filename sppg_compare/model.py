"""Shared assignment model, validation, scoring, and deterministic repair."""

from dataclasses import asdict, dataclass, field
import math
import random


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
    distances: tuple[tuple[float, ...], ...] = field(init=False, repr=False)
    eligible: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self):
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Service radius must be a positive finite number.")
        if not self.kitchens or not self.schools:
            raise ValueError("A scenario needs kitchens and schools.")
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

    @property
    def total_demand(self):
        return sum(s.demand for s in self.schools)

    @property
    def school_order(self):
        return tuple(sorted(range(len(self.schools)),
                            key=lambda i: (-self.schools[i].demand, i)))

    def to_dict(self):
        return {"name": self.name, "seed": self.seed, "radius": self.radius,
                "kitchens": [asdict(k) for k in self.kitchens],
                "schools": [asdict(s) for s in self.schools]}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["seed"], data["radius"],
                   tuple(Kitchen(**k) for k in data["kitchens"]),
                   tuple(School(**s) for s in data["schools"]))


@dataclass(frozen=True)
class Evaluation:
    assignment: tuple[int, ...]
    loads: tuple[int, ...]
    objective: float
    overflow: int
    radius_violation: int

    @property
    def feasible(self):
        return self.overflow == 0 and self.radius_violation == 0

    @property
    def rank(self):
        return self.radius_violation, self.overflow, self.objective


def evaluate(scenario, assignment):
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
        objective += demand * scenario.distances[i][kitchen]
        if kitchen not in scenario.eligible[i]:
            radius_violation += demand
    overflow = sum(max(0, load - k.capacity)
                   for load, k in zip(loads, scenario.kitchens))
    return Evaluation(assignment, tuple(loads), objective, overflow, radius_violation)


def repair(scenario, assignment):
    """Move schools out of overloaded kitchens; never overload a destination."""
    assignment = list(assignment)
    current = evaluate(scenario, assignment)
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
                delta = demand * (scenario.distances[i][target] - scenario.distances[i][source])
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


def greedy_baseline(scenario):
    loads = [0] * len(scenario.kitchens)
    assignment = [0] * len(scenario.schools)
    for i in scenario.school_order:
        demand = scenario.schools[i].demand
        fits = [j for j in scenario.eligible[i]
                if loads[j] + demand <= scenario.kitchens[j].capacity]
        choices = fits or list(scenario.eligible[i])
        j = min(choices, key=lambda j: (scenario.distances[i][j], j))
        assignment[i] = j
        loads[j] += demand
    return evaluate(scenario, repair(scenario, assignment))


SIZES = {"Small": (3, 15, 101), "Medium": (5, 40, 202), "Large": (8, 80, 303)}


def generate_scenario(size="Small", seed=None, radius=6.0):
    """Generate an overlapping, capacity-constrained instance with a private witness.

    Coordinates scale with radius so regenerating at another radius is well defined.
    The witness is used only here to verify feasibility, never returned to solvers.
    """
    if size not in SIZES:
        raise ValueError(f"Unknown scenario size: {size}")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Service radius must be a positive finite number.")
    count_k, count_s, default_seed = SIZES[size]
    seed = default_seed if seed is None else seed
    rng = random.Random(seed)
    for _ in range(100):
        # Nearby kitchen clusters produce multiple eligible choices per school.
        coords = [(rng.uniform(0.4, 2.0) * radius, rng.uniform(0.4, 2.0) * radius)
                  for _ in range(count_k)]
        schools, witness, loads = [], [], [0] * count_k
        for i in range(count_s):
            owner = i % count_k if i < count_k else rng.randrange(count_k)
            angle = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(1 / 12, 0.75) * radius
            x, y = coords[owner]
            demand = rng.randint(80, 300)
            schools.append(School(f"S{i + 1:02d}", x + distance * math.cos(angle),
                                  y + distance * math.sin(angle), demand))
            witness.append(owner)
            loads[owner] += demand
        kitchens = tuple(Kitchen(f"K{j + 1}", x, y, math.ceil(loads[j] * 1.05))
                         for j, (x, y) in enumerate(coords))
        scenario = Scenario(size, seed, radius, kitchens, tuple(schools))
        nearest = tuple(min(choices, key=lambda j: (scenario.distances[i][j], j))
                        for i, choices in enumerate(scenario.eligible))
        overlap = sum(len(choices) > 1 for choices in scenario.eligible)
        if overlap >= count_s // 3 and evaluate(scenario, nearest).overflow > 0:
            if not evaluate(scenario, witness).feasible:
                raise RuntimeError("Generator produced an invalid feasibility witness.")
            return scenario
    raise ValueError("Could not generate a challenging scenario after 100 attempts; change the seed.")
