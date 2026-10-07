"""GA, preference-score PSO, and assignment Ant System with a common budget."""

from dataclasses import asdict, dataclass, field
import math
import random
import time

from .model import Evaluation, evaluate, repair
from .diagnostics import SearchDiagnostics
from .oracle import OracleResult, optimality_gap, lower_bound_gap, solve_exact


ALGORITHMS = ("GA", "PSO", "ACO")
ALGORITHM_LABELS = {"GA": "GA", "PSO": "PSO (preference-score/discrete)", "ACO": "ACO"}
CONSTRAINT_POLICIES = ("shared_repair", "aco_capacity_aware", "no_repair")


@dataclass(frozen=True)
class SolverSettings:
    budget: int = 5000
    population: int = 30
    crossover: float = 0.9
    mutation: float | None = None
    tournament: int = 3
    elites: int = 2
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5
    velocity_limit: float = 0.2
    alpha: float = 1.0
    beta: float = 2.0
    evaporation: float = 0.2
    constraint_policy: str = "shared_repair"
    oracle_time_limit: float = 60.0

    def validate(self):
        if self.constraint_policy not in CONSTRAINT_POLICIES:
            raise ValueError(f"Unknown constraint policy: {self.constraint_policy}")
        if not math.isfinite(self.oracle_time_limit) or self.oracle_time_limit <= 0:
            raise ValueError("Oracle time limit must be positive and finite.")
        for name in ("budget", "population", "tournament", "elites"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer.")
        if self.budget < 1 or self.population < 3:
            raise ValueError("Budget must be positive and population must be at least 3.")
        if not 0 <= self.elites < self.population or self.tournament < 1:
            raise ValueError("Elites must be below population; tournament size must be positive.")
        numeric = ("crossover", "inertia", "cognitive", "social", "velocity_limit",
                   "alpha", "beta", "evaporation")
        if any(not math.isfinite(getattr(self, name)) for name in numeric):
            raise ValueError("Algorithm parameters must be finite.")
        if not 0 <= self.crossover <= 1 or not 0 < self.evaporation < 1:
            raise ValueError("Crossover must be in [0,1] and evaporation in (0,1).")
        if self.mutation is not None and (not math.isfinite(self.mutation)
                                         or not 0 <= self.mutation <= 1):
            raise ValueError("Mutation must be in [0,1], or automatic.")
        if any(getattr(self, name) < 0 for name in ("inertia", "cognitive", "social", "alpha", "beta")):
            raise ValueError("PSO coefficients and ACO exponents cannot be negative.")
        if self.alpha > 100 or self.beta > 100:
            raise ValueError("ACO exponents must not exceed 100.")
        if not 0 < self.velocity_limit <= 1:
            raise ValueError("Velocity limit must be in (0,1].")


@dataclass(frozen=True)
class ProgressSnapshot:
    algorithm: str
    seed: int
    iteration: int
    evaluations: int
    best: Evaluation
    solver_seconds: float
    diagnostics: dict = field(default_factory=dict)
    optimal_objective: float | None = None
    optimality_gap_percent: float | None = None
    oracle_status: str = "not_solved"
    search_state: dict = field(default_factory=dict)
    oracle_lower_bound: float | None = None
    lower_bound_gap_percent: float | None = None
    optimal_active_sppg_count: int | None = None


@dataclass
class RunResult:
    algorithm: str
    seed: int
    settings: SolverSettings
    scenario_name: str
    scenario_seed: int
    best: Evaluation
    evaluations: int
    solver_seconds: float
    history: list[dict] = field(default_factory=list)
    snapshots: list[ProgressSnapshot] = field(default_factory=list)
    completed: bool = True
    oracle: OracleResult | None = None
    diagnostics: dict = field(default_factory=dict)
    scenario_id: str = ""

    @property
    def optimal_objective(self):
        return self.oracle.optimal_objective if self.oracle else None

    @property
    def optimality_gap_percent(self):
        return optimality_gap(self.best, self.oracle)[1]

    def to_dict(self):
        data = asdict(self)
        data.update(optimizer_seed=self.seed, algorithm_label=ALGORITHM_LABELS[self.algorithm],
                    optimal_objective=self.optimal_objective,
                    oracle_lower_bound=self.oracle.lower_bound if self.oracle else None,
                    lower_bound_gap_percent=lower_bound_gap(self.best, self.oracle),
                    absolute_optimality_gap=optimality_gap(self.best, self.oracle)[0],
                    optimality_gap_percent=self.optimality_gap_percent)
        return data


class Optimizer:
    """One advance() is an interruptible population-sized batch of work.

    Every evaluated candidate passes through common repair and scoring, including
    repeats. Stored evaluations are reused for elites and personal bests. Runtime
    measures advance() work only, excluding pauses, queue waits, and animation.
    """

    def __init__(self, scenario, algorithm, seed=1, settings=None, *, capture_search=False):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.scenario = scenario
        self.algorithm = algorithm
        self.seed = seed
        self.settings = settings or SolverSettings()
        self.settings.validate()
        self.oracle = solve_exact(scenario, self.settings.oracle_time_limit)
        self.diagnostics = SearchDiagnostics()
        self.rng = random.Random(seed)
        self.evaluations = 0
        self.iteration = 0
        self.solver_seconds = 0.0
        self.best = None
        self.history = []
        self.snapshots = []
        self.population = []
        self.positions = []
        self.velocities = []
        self.personal_positions = []
        self.personal_scores = []
        self.global_position = None
        self.last_decoded = []
        self.last_phenotypes = []
        self.last_orders = []
        self.last_phenotype_orders = []
        self.pheromone = [[1.0] * len(scenario.kitchens) for _ in scenario.schools]
        self.heuristic = [[1 / (1 + d) for d in row] for row in scenario.costs]
        self.order = scenario.school_order
        self.capture_search = capture_search
        self.ant_trace = []
        self.route_pheromone = [[1.] * len(scenario.schools) for _ in range(len(scenario.schools)+1)]
        self.capture_seconds = 0.

    @property
    def done(self):
        return self.evaluations >= self.settings.budget

    def _score(self, assignment, previous=None, previous_phenotype=None, position_changed=None, visit_order=None, previous_order=None, previous_phenotype_order=None):
        raw = evaluate(self.scenario, assignment, visit_order)
        repair_enabled = self.settings.constraint_policy != "no_repair"
        score = evaluate(self.scenario, repair(self.scenario, assignment, raw), raw.visit_order or None) if repair_enabled and (raw.overflow or raw.active_site_violation) else raw
        self.diagnostics.record(raw, score, repair_enabled, previous, previous_phenotype, position_changed, previous_order, previous_phenotype_order)
        self.evaluations += 1
        if self.best is None or score.rank < self.best.rank:
            self.best = score
        if self.evaluations == 1 or self.evaluations % 100 == 0 or self.done:
            self.history.append({"evaluations": self.evaluations,
                                 "objective": self.best.objective if self.best.feasible else None,
                                 "overflow": self.best.overflow,
                                 "active_site_violation": self.best.active_site_violation,
                                 "active_sppg_count": len(set(self.best.assignment)),
                                 "radius_violation": self.best.radius_violation,
                                 "optimality_gap_percent": optimality_gap(self.best, self.oracle)[1]})
            self.history[-1]["lower_bound_gap_percent"] = lower_bound_gap(self.best, self.oracle)
        return score

    def _random_order(self):
        if self.scenario.routing_mode != "multi_stop":
            return None
        order = list(range(len(self.scenario.schools)))
        self.rng.shuffle(order)
        return order

    def _random_assignment(self):
        return tuple(self.rng.choice(row) for row in self.scenario.eligible)

    def _tournament(self):
        return min((self.rng.choice(self.population) for _ in range(self.settings.tournament)),
                   key=lambda score: score.rank)

    def _ga(self, limit):
        if not self.population:
            self.population = [self._score(self._random_assignment(), visit_order=self._random_order()) for _ in range(limit)]
            return
        parents = self.population
        elite_count = min(self.settings.elites, len(parents) - 1)
        next_population = sorted(parents, key=lambda score: score.rank)[:elite_count]
        mutation = self.settings.mutation
        if mutation is None:
            mutation = 1 / len(self.scenario.schools)
        for _ in range(min(limit, self.settings.population - elite_count)):
            a, b = self._tournament(), self._tournament()
            crosses = self.rng.random() < self.settings.crossover
            child = [b.assignment[i] if crosses and self.rng.random() < 0.5 else gene
                     for i, gene in enumerate(a.assignment)]
            for i, choices in enumerate(self.scenario.eligible):
                if self.rng.random() < mutation:
                    alternatives = [j for j in choices if j != child[i]]
                    if alternatives:
                        child[i] = self.rng.choice(alternatives)
            order = None
            if self.scenario.routing_mode == "multi_stop":
                cut = self.rng.randrange(len(child)+1) if crosses else len(child)
                order = list(a.visit_order[:cut])
                order.extend(i for i in b.visit_order if i not in order)
                if len(order) > 1 and self.rng.random() < min(1., mutation*len(child)):
                    left, right = self.rng.sample(range(len(order)), 2)
                    order[left], order[right] = order[right], order[left]
            next_population.append(self._score(child, visit_order=order))
        self.population = next_population

    def _decode(self, position):
        return tuple(max(choices, key=lambda j: (position[i][j], -j))
                     for i, choices in enumerate(self.scenario.eligible))

    def _pso(self, limit):
        n, k = len(self.scenario.schools), len(self.scenario.kitchens)
        dimensions = k + int(self.scenario.routing_mode == "multi_stop")
        if not self.positions:
            for _ in range(limit):
                position = [[self.rng.random() for _ in range(dimensions)] for _ in range(n)]
                velocity = [[self.rng.uniform(-self.settings.velocity_limit,
                                             self.settings.velocity_limit)
                             for _ in range(dimensions)] for _ in range(n)]
                decoded = self._decode(position)
                score = self._score(decoded, visit_order=sorted(range(n), key=lambda i: (position[i][k], i)) if dimensions > k else None)
                self.last_decoded.append(decoded)
                self.last_phenotypes.append(score.assignment)
                self.last_orders.append(tuple(sorted(range(n), key=lambda i: (position[i][k], i))) if dimensions > k else ())
                self.last_phenotype_orders.append(score.visit_order)
                self.positions.append(position)
                self.velocities.append(velocity)
                self.personal_positions.append([row[:] for row in position])
                self.personal_scores.append(score)
            winner = min(range(len(self.personal_scores)), key=lambda p: self.personal_scores[p].rank)
            self.global_position = [row[:] for row in self.personal_positions[winner]]
            return
        # Synchronous swarm: all particles use the same previous global position.
        global_position = self.global_position
        for p in range(limit):
            position_changed = False
            position, velocity, personal = (self.positions[p], self.velocities[p],
                                            self.personal_positions[p])
            for i, choices in enumerate(self.scenario.eligible):
                for j in (*choices, k) if dimensions > k else choices:
                    v = (self.settings.inertia * velocity[i][j]
                         + self.settings.cognitive * self.rng.random() * (personal[i][j] - position[i][j])
                         + self.settings.social * self.rng.random() * (global_position[i][j] - position[i][j]))
                    velocity[i][j] = max(-self.settings.velocity_limit, min(self.settings.velocity_limit, v))
                    new_value = max(0.0, min(1.0, position[i][j] + velocity[i][j]))
                    position_changed = position_changed or new_value != position[i][j]
                    position[i][j] = new_value
            decoded = self._decode(position)
            score = self._score(decoded, self.last_decoded[p], self.last_phenotypes[p], position_changed,
                                visit_order=sorted(range(n), key=lambda i: (position[i][k], i)) if dimensions > k else None,
                                previous_order=self.last_orders[p] if dimensions > k else None,
                                previous_phenotype_order=self.last_phenotype_orders[p] if dimensions > k else None)
            self.last_orders[p] = tuple(sorted(range(n), key=lambda i: (position[i][k], i))) if dimensions > k else ()
            self.last_phenotype_orders[p] = score.visit_order
            self.last_decoded[p] = decoded
            self.last_phenotypes[p] = score.assignment
            if score.rank < self.personal_scores[p].rank:
                self.personal_scores[p] = score
                self.personal_positions[p] = [row[:] for row in position]
        winner = min(range(len(self.personal_scores)), key=lambda p: self.personal_scores[p].rank)
        self.global_position = [row[:] for row in self.personal_positions[winner]]

    def _aco(self, limit):
        self.ant_trace = []
        for ant in range(limit):
            assignment = [0] * len(self.scenario.schools)
            loads = [0] * len(self.scenario.kitchens)
            for i in self.order:
                demand = self.scenario.schools[i].demand
                fits = ([j for j in self.scenario.eligible[i]
                         if loads[j] + demand <= self.scenario.kitchens[j].capacity]
                        if self.settings.constraint_policy == "aco_capacity_aware" else [])
                choices = fits or list(self.scenario.eligible[i])
                # Log weights avoid overflow for finite but large user exponents.
                logs = [self.settings.alpha * math.log(self.pheromone[i][j])
                        + self.settings.beta * math.log(self.heuristic[i][j]) for j in choices]
                peak = max(logs)
                weights = [math.exp(weight - peak) for weight in logs]
                j = self.rng.choices(choices, weights=weights, k=1)[0]
                if self.capture_search and ant == 0:
                    capture_started = time.perf_counter()
                    self.ant_trace.append({"school": i, "kitchen": j,
                                           "choices": tuple(choices),
                                           "probabilities": tuple(w / sum(weights) for w in weights)})
                    self.capture_seconds += time.perf_counter() - capture_started
                assignment[i] = j
                loads[j] += demand
            route_order = self._construct_routes(assignment) if self.scenario.routing_mode == "multi_stop" else None
            score = self._score(assignment, visit_order=route_order)
            if self.capture_search and ant == 0:
                capture_started = time.perf_counter()
                self.ant_raw = tuple(assignment)
                self.ant_repaired = score.assignment
                self.capture_seconds += time.perf_counter() - capture_started
        for row in self.pheromone:
            for j in range(len(row)):
                row[j] = max(0.01, row[j] * (1 - self.settings.evaporation))
        for row in self.route_pheromone:
            for i in range(len(row)):
                row[i] = max(.01, row[i]*(1-self.settings.evaporation))
        if self.best.feasible:
            deposit = 1 / (1 + self.best.objective / self.scenario.total_demand)
            for i, j in enumerate(self.best.assignment):
                self.pheromone[i][j] = min(1.0, self.pheromone[i][j] + deposit)
            for j in set(self.best.assignment):
                previous = len(self.scenario.schools)
                for i in self.best.visit_order:
                    if self.best.assignment[i] == j:
                        self.route_pheromone[previous][i] = min(1., self.route_pheromone[previous][i]+deposit)
                        previous = i

    def _construct_routes(self, assignment):
        order, n = [], len(assignment)
        for j in sorted(set(assignment)):
            remaining = [i for i in range(n) if assignment[i] == j]
            previous = n
            while remaining:
                logs = []
                for i in remaining:
                    time = self.scenario.travel_times[i][j] if previous == n else self.scenario.school_routes[i][previous][0]
                    logs.append(self.settings.alpha*math.log(self.route_pheromone[previous][i])
                                + self.settings.beta*math.log(self.scenario.schools[i].demand/(1+time)))
                peak = max(logs)
                chosen = self.rng.choices(remaining, weights=[math.exp(v-peak) for v in logs], k=1)[0]
                order.append(chosen)
                remaining.remove(chosen)
                previous = chosen
        return order

    def advance_to(self, evaluation_limit):
        """Advance without crossing a shared evaluation boundary."""
        return self.advance(evaluation_limit=evaluation_limit)

    def advance(self, evaluation_limit=None):
        if self.done:
            raise StopIteration
        started = time.perf_counter()
        self.capture_seconds = 0.
        self.diagnostics.new_batch()
        boundary = self.settings.budget if evaluation_limit is None else min(self.settings.budget, evaluation_limit)
        if boundary <= self.evaluations:
            raise ValueError("Evaluation boundary must exceed the current count.")
        limit = min(self.settings.population, boundary - self.evaluations)
        {"GA": self._ga, "PSO": self._pso, "ACO": self._aco}[self.algorithm](limit)
        self.iteration += 1
        self.solver_seconds += max(0., time.perf_counter() - started - self.capture_seconds)
        snapshot = ProgressSnapshot(self.algorithm, self.seed, self.iteration,
                                    self.evaluations, self.best, self.solver_seconds,
                                    self.diagnostics.snapshot(self.evaluations), self.oracle.optimal_objective,
                                    optimality_gap(self.best, self.oracle)[1], self.oracle.status,
                                    self._search_state(), self.oracle.lower_bound,
                                    lower_bound_gap(self.best, self.oracle), self.oracle.optimal_active_sppg_count)
        self.snapshots.append(snapshot)
        return snapshot

    def _search_state(self):
        """Read-only visualization capture, outside timed work; no RNG/evaluation."""
        if not self.capture_search or self.algorithm == "GA":
            return {}
        if self.algorithm == "ACO":
            return {"kind": "ACO", "matrix": tuple(tuple(row) for row in self.pheromone),
                    "trace": tuple(dict(step) for step in self.ant_trace),
                    "raw_assignment": self.ant_raw, "repaired_assignment": self.ant_repaired,
                    "evaporation": self.settings.evaporation,
                    "route_pheromone": tuple(tuple(row) for row in self.route_pheromone),
                    "best_visit_order": self.best.visit_order}
        def project(matrix):
            # Fixed linear projection across schools; axes never depend on best.
            pairs = [(i, row[0], row[1]) for i, row in enumerate(self.scenario.eligible) if len(row) > 1]
            if not pairs:
                return (0., 0.)
            return tuple(sum(matrix[i][a if axis == 0 else b] for i, a, b in pairs) / len(pairs)
                         for axis in (0, 1))
        matrix = tuple(tuple(sum(p[i][j] for p in self.positions) / len(self.positions)
                             for j in range(len(self.scenario.kitchens)))
                       for i in range(len(self.scenario.schools)))
        particles = tuple({"id": p, "position": project(self.positions[p]),
                           "velocity": project(self.velocities[p]),
                           "personal_best": project(self.personal_positions[p]),
                           "personal_best_feasible": evaluate_feasible,
                           "decoded": self.last_decoded[p], "phenotype": self.last_phenotypes[p]}
                          for p, evaluate_feasible in ((p, self.personal_scores[p].feasible)
                                                       for p in range(min(8, len(self.positions)))))
        return {"kind": "PSO", "matrix": matrix, "particles": particles,
                "global_best": project(self.global_position), "population": len(self.positions),
                "route_keys": tuple(self.global_position[i][-1] for i in range(len(self.scenario.schools))) if self.scenario.routing_mode == "multi_stop" else (),
                "projection": "Mean preferences for first/second eligible kitchens across schools with >=2 choices"}

    def result(self):
        if self.best is None:
            raise ValueError("Run has not evaluated a candidate yet.")
        return RunResult(self.algorithm, self.seed, self.settings, self.scenario.name,
                         self.scenario.seed, self.best, self.evaluations,
                         self.solver_seconds, self.history[:], self.snapshots[:], self.done,
                         self.oracle, self.diagnostics.snapshot(self.evaluations), self.scenario.instance_id)


def solve(scenario, algorithm, seed=1, settings=None, *, capture_search=False):
    optimizer = Optimizer(scenario, algorithm, seed, settings, capture_search=capture_search)
    while not optimizer.done:
        optimizer.advance()
    return optimizer.result()
