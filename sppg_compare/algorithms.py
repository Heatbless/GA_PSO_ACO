"""GA, preference-score PSO, and assignment Ant System with a common budget."""

from dataclasses import asdict, dataclass, field
import math
import random
import time

from .model import Evaluation, evaluate, repair


ALGORITHMS = ("GA", "PSO", "ACO")


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

    def validate(self):
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

    def to_dict(self):
        return asdict(self)


class Optimizer:
    """One advance() is an interruptible population-sized batch of work.

    Every evaluated candidate passes through common repair and scoring, including
    repeats. Stored evaluations are reused for elites and personal bests. Runtime
    measures advance() work only, excluding pauses, queue waits, and animation.
    """

    def __init__(self, scenario, algorithm, seed=1, settings=None):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.scenario = scenario
        self.algorithm = algorithm
        self.seed = seed
        self.settings = settings or SolverSettings()
        self.settings.validate()
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
        self.pheromone = [[1.0] * len(scenario.kitchens) for _ in scenario.schools]
        self.heuristic = [[1 / (1 + d) for d in row] for row in scenario.distances]
        self.order = scenario.school_order

    @property
    def done(self):
        return self.evaluations >= self.settings.budget

    def _score(self, assignment):
        score = evaluate(self.scenario, repair(self.scenario, assignment))
        self.evaluations += 1
        if self.best is None or score.rank < self.best.rank:
            self.best = score
        if self.evaluations == 1 or self.evaluations % 100 == 0 or self.done:
            self.history.append({"evaluations": self.evaluations,
                                 "objective": self.best.objective if self.best.feasible else None,
                                 "overflow": self.best.overflow,
                                 "radius_violation": self.best.radius_violation})
        return score

    def _random_assignment(self):
        return tuple(self.rng.choice(row) for row in self.scenario.eligible)

    def _tournament(self):
        return min((self.rng.choice(self.population) for _ in range(self.settings.tournament)),
                   key=lambda score: score.rank)

    def _ga(self, limit):
        if not self.population:
            self.population = [self._score(self._random_assignment()) for _ in range(limit)]
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
            next_population.append(self._score(child))
        self.population = next_population

    def _decode(self, position):
        return tuple(max(choices, key=lambda j: (position[i][j], -j))
                     for i, choices in enumerate(self.scenario.eligible))

    def _pso(self, limit):
        n, k = len(self.scenario.schools), len(self.scenario.kitchens)
        if not self.positions:
            for _ in range(limit):
                position = [[self.rng.random() for _ in range(k)] for _ in range(n)]
                velocity = [[self.rng.uniform(-self.settings.velocity_limit,
                                             self.settings.velocity_limit)
                             for _ in range(k)] for _ in range(n)]
                score = self._score(self._decode(position))
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
            position, velocity, personal = (self.positions[p], self.velocities[p],
                                            self.personal_positions[p])
            for i, choices in enumerate(self.scenario.eligible):
                for j in choices:
                    v = (self.settings.inertia * velocity[i][j]
                         + self.settings.cognitive * self.rng.random() * (personal[i][j] - position[i][j])
                         + self.settings.social * self.rng.random() * (global_position[i][j] - position[i][j]))
                    velocity[i][j] = max(-self.settings.velocity_limit, min(self.settings.velocity_limit, v))
                    position[i][j] = max(0.0, min(1.0, position[i][j] + velocity[i][j]))
            score = self._score(self._decode(position))
            if score.rank < self.personal_scores[p].rank:
                self.personal_scores[p] = score
                self.personal_positions[p] = [row[:] for row in position]
        winner = min(range(len(self.personal_scores)), key=lambda p: self.personal_scores[p].rank)
        self.global_position = [row[:] for row in self.personal_positions[winner]]

    def _aco(self, limit):
        for _ in range(limit):
            assignment = [0] * len(self.scenario.schools)
            loads = [0] * len(self.scenario.kitchens)
            for i in self.order:
                demand = self.scenario.schools[i].demand
                fits = [j for j in self.scenario.eligible[i]
                        if loads[j] + demand <= self.scenario.kitchens[j].capacity]
                choices = fits or list(self.scenario.eligible[i])
                # Log weights avoid overflow for finite but large user exponents.
                logs = [self.settings.alpha * math.log(self.pheromone[i][j])
                        + self.settings.beta * math.log(self.heuristic[i][j]) for j in choices]
                peak = max(logs)
                weights = [math.exp(weight - peak) for weight in logs]
                j = self.rng.choices(choices, weights=weights, k=1)[0]
                assignment[i] = j
                loads[j] += demand
            self._score(assignment)
        for row in self.pheromone:
            for j in range(len(row)):
                row[j] = max(0.01, row[j] * (1 - self.settings.evaporation))
        if self.best.feasible:
            deposit = 1 / (1 + self.best.objective / self.scenario.total_demand)
            for i, j in enumerate(self.best.assignment):
                self.pheromone[i][j] = min(1.0, self.pheromone[i][j] + deposit)

    def advance(self):
        if self.done:
            raise StopIteration
        started = time.perf_counter()
        limit = min(self.settings.population, self.settings.budget - self.evaluations)
        {"GA": self._ga, "PSO": self._pso, "ACO": self._aco}[self.algorithm](limit)
        self.iteration += 1
        self.solver_seconds += time.perf_counter() - started
        snapshot = ProgressSnapshot(self.algorithm, self.seed, self.iteration,
                                    self.evaluations, self.best, self.solver_seconds)
        self.snapshots.append(snapshot)
        return snapshot

    def result(self):
        if self.best is None:
            raise ValueError("Run has not evaluated a candidate yet.")
        return RunResult(self.algorithm, self.seed, self.settings, self.scenario.name,
                         self.scenario.seed, self.best, self.evaluations,
                         self.solver_seconds, self.history[:], self.snapshots[:], self.done)


def solve(scenario, algorithm, seed=1, settings=None):
    optimizer = Optimizer(scenario, algorithm, seed, settings)
    while not optimizer.done:
        optimizer.advance()
    return optimizer.result()
