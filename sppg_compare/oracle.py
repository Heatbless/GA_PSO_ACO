"""Binary MILP oracle for the exact same allocation model as the heuristics."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import time

from .model import evaluate


@dataclass(frozen=True)
class OracleResult:
    status: str
    optimal_objective: float | None = None
    incumbent_objective: float | None = None
    assignment: tuple[int, ...] | None = None
    lower_bound: float | None = None
    mip_gap: float | None = None
    solver_seconds: float = 0.0
    message: str = ""
    solver: str = "SciPy/HiGHS MILP"
    scipy_version: str = ""
    visit_order: tuple[int, ...] | None = None
    optimal_active_sppg_count: int | None = None
    optimal_allocation_distance_km: float | None = None

    def to_dict(self):
        return asdict(self)


def optimality_gap(score, oracle):
    """Return (absolute gap, percentage gap); never use an unproven incumbent."""
    if not score.feasible or oracle is None or oracle.status != "optimal":
        return None, None
    optimum = oracle.optimal_objective
    tolerance = 1e-7 * max(1.0, abs(optimum))
    delta = score.objective - optimum
    if delta < -tolerance:
        raise RuntimeError("Feasible heuristic objective is below the certified MILP optimum.")
    delta = max(0.0, delta)
    if abs(optimum) <= 1e-12:
        return delta, 0.0 if delta <= tolerance else None
    return delta, 100 * delta / abs(optimum)


def lower_bound_gap(score, oracle):
    """Upper bound on relative optimality gap; not a certified exact gap."""
    if not score.feasible or oracle is None or oracle.lower_bound is None or oracle.lower_bound <= 1e-12:
        return None
    return 100*max(0., score.objective-oracle.lower_bound)/oracle.lower_bound


@lru_cache(maxsize=128)
def solve_exact(scenario, time_limit=60.0):
    """One binary x[i,j] per radius-eligible edge; one kitchen per school.

    Caching uses the immutable full scenario, including geometry and capacities,
    plus the time limit. Oracle work never consumes candidate evaluations.
    """
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Oracle time limit must be positive and finite.")
    if scenario.routing_mode == "multi_stop":
        from .routing_oracle import solve_routing_exact
        return solve_routing_exact(scenario, time_limit)
    try:
        import numpy as np
        import scipy
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import csc_matrix
    except ImportError as error:
        raise RuntimeError("The MILP oracle requires SciPy. Run: python -m pip install -r requirements.txt") from error
    n, k = len(scenario.schools), len(scenario.kitchens)
    edges = [(i, j) for i, choices in enumerate(scenario.eligible) for j in choices]
    rows, columns, values = [], [], []
    objective = []
    for column, (i, j) in enumerate(edges):
        rows.extend((i, n + j))
        columns.extend((column, column))
        values.extend((1.0, float(scenario.schools[i].demand)))
        objective.append(scenario.distance_weight * scenario.costs[i][j] if scenario.objective_metric == "facility_distance"
                         else scenario.schools[i].demand * scenario.costs[i][j])
    lower = np.concatenate((np.ones(n), np.full(k, -np.inf)))
    upper = np.concatenate((np.ones(n), np.array([item.capacity for item in scenario.kitchens])))
    variable_count = len(edges)
    row_count = n + k
    if scenario.max_active_kitchens is not None or scenario.objective_metric == "facility_distance":
        # Capacity links imply x[i,j] <= y[j] because all demands are positive.
        for j, kitchen in enumerate(scenario.kitchens):
            rows.extend((n+j, n+k))
            columns.extend((variable_count+j, variable_count+j))
            values.extend((-float(kitchen.capacity), 1.0))
        upper[n:n+k] = 0
        lower = np.append(lower, -np.inf)
        upper = np.append(upper, scenario.max_active_kitchens or k)
        objective.extend([scenario.opening_weight if scenario.objective_metric == "facility_distance" else 0.0]*k)
        variable_count += k
        row_count += 1
    matrix = csc_matrix((values, (rows, columns)), shape=(row_count, variable_count))
    started = time.perf_counter()
    result = milp(np.array(objective), integrality=np.ones(variable_count),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper),
                  options={"mip_rel_gap": 0.0, "time_limit": time_limit, "presolve": True})
    elapsed = time.perf_counter() - started
    status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "error"}.get(result.status, "error")
    assignment, incumbent = None, None
    if result.x is not None:
        candidate = [-1] * n
        integral = all(abs(float(value) - round(float(value))) <= 1e-6 for value in result.x)
        if integral:
            for value, (i, j) in zip(result.x, edges):
                if value > 0.5:
                    if candidate[i] != -1:
                        integral = False
                    candidate[i] = j
            if integral and all(j >= 0 for j in candidate):
                score = evaluate(scenario, tuple(candidate))
                if score.feasible:
                    if status == "optimal" and abs(score.objective-result.fun) > 1e-6*max(1., abs(score.objective)):
                        raise RuntimeError("MILP objective does not match the shared evaluator.")
                    assignment, incumbent = score.assignment, score.objective
    if status == "optimal" and assignment is None:
        raise RuntimeError("MILP reported optimality but its assignment failed the shared evaluator.")

    def finite(value):
        return float(value) if value is not None and math.isfinite(value) else None

    return OracleResult(status=status, optimal_objective=incumbent if status == "optimal" else None,
                        incumbent_objective=incumbent, assignment=assignment,
                        lower_bound=finite(getattr(result, "mip_dual_bound", None)),
                        mip_gap=finite(getattr(result, "mip_gap", None)), solver_seconds=elapsed,
                        message=result.message, scipy_version=scipy.__version__,
                        optimal_active_sppg_count=len(set(assignment)) if status == "optimal" and assignment else None,
                        optimal_allocation_distance_km=sum(scenario.road_distances[i][j] for i, j in enumerate(assignment))
                            if status == "optimal" and assignment else None)
