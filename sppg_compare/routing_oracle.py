"""Exact multi-depot minimum-latency MILP with one returning tour per depot."""
import math
import time

from .model import evaluate


def solve_routing_exact(scenario, time_limit):
    import numpy as np
    import scipy
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix
    from .oracle import OracleResult

    n, k = len(scenario.schools), len(scenario.kitchens)
    variables, cost, lower_bounds, upper_bounds, integer = {}, [], [], [], []
    max_leg = max(max(t for row in scenario.travel_times for t in row),
                  max(r[0] for row in scenario.school_routes for r in row))
    horizon = max(1., n*(max_leg+scenario.unloading_minutes))
    big_m = horizon+max_leg+scenario.unloading_minutes

    def variable(key, c=0., upper=1., binary=True):
        variables[key] = len(cost)
        cost.append(c)
        lower_bounds.append(0.)
        upper_bounds.append(upper)
        integer.append(int(binary))

    for i, choices in enumerate(scenario.eligible):
        for j in choices:
            variable(("x", i, j))
            variable(("start", j, i))
            variable(("end", i, j))
    for j in range(k):
        variable(("y", j))
    arcs = [(i, h) for i in range(n) for h in range(n) if i != h
            and set(scenario.eligible[i]) & set(scenario.eligible[h])]
    for i, h in arcs:
        variable(("arc", i, h))
    for i, school in enumerate(scenario.schools):
        variable(("time", i), school.demand, horizon, False)
        variable(("order", i), 0., n, False)
    rows, columns, values, lows, ups = [], [], [], [], []

    def constraint(terms, low=-np.inf, high=np.inf):
        row = len(lows)
        for key, coefficient in terms:
            rows.append(row)
            columns.append(variables[key])
            values.append(coefficient)
        lows.append(low)
        ups.append(high)

    incoming = {i: [] for i in range(n)}
    outgoing = {i: [] for i in range(n)}
    for i, h in arcs:
        incoming[h].append(("arc", i, h))
        outgoing[i].append(("arc", i, h))
    for i, choices in enumerate(scenario.eligible):
        constraint([(("x", i, j), 1.) for j in choices], 1., 1.)
        constraint([(("start", j, i), 1.) for j in choices]+[(key, 1.) for key in incoming[i]], 1., 1.)
        constraint([(("end", i, j), 1.) for j in choices]+[(key, 1.) for key in outgoing[i]], 1., 1.)
        # Direct shortest path is a valid lower bound on tour arrival time.
        constraint([(("time", i), 1.)]+[(("x", i, j), -scenario.travel_times[i][j]) for j in choices], 0.)
        for j in choices:
            constraint([(("start", j, i), 1.), (("x", i, j), -1.)], high=0.)
            constraint([(("end", i, j), 1.), (("x", i, j), -1.)], high=0.)
            constraint([(("time", i), 1.), (("start", j, i), -big_m)], scenario.travel_times[i][j]-big_m)
    for j, kitchen in enumerate(scenario.kitchens):
        served = [i for i in range(n) if j in scenario.eligible[i]]
        constraint([(("x", i, j), scenario.schools[i].demand) for i in served]+[(("y", j), -kitchen.capacity)], high=0.)
        constraint([(("start", j, i), 1.) for i in served]+[(("end", i, j), -1.) for i in served], 0., 0.)
        constraint([(("start", j, i), 1.) for i in served]+[(("y", j), -1.)], high=0.)
    constraint([(("y", j), 1.) for j in range(k)], high=scenario.max_active_kitchens or k)
    for i, h in arcs:
        key = ("arc", i, h)
        # An arc can only join schools allocated to the same SPPG.
        for j in scenario.eligible[i]:
            terms = [(key, 1.), (("x", i, j), 1.)]
            if j in scenario.eligible[h]:
                terms.append((("x", h, j), -1.))
            constraint(terms, high=1.)
        travel = scenario.school_routes[h][i][0]
        constraint([(("time", h), 1.), (("time", i), -1.), (key, -big_m)],
                   scenario.unloading_minutes+travel-big_m)
        # This also prevents subtours when distinct schools coincide and service=0.
        constraint([(("order", h), 1.), (("order", i), -1.), (key, -n)], 1.-n)
    matrix = csc_matrix((values, (rows, columns)), shape=(len(lows), len(cost)))
    started = time.perf_counter()
    result = milp(np.array(cost), integrality=np.array(integer),
                  bounds=Bounds(lower_bounds, upper_bounds),
                  constraints=LinearConstraint(matrix, lows, ups),
                  options={"mip_rel_gap": 0., "time_limit": time_limit, "presolve": True})
    elapsed = time.perf_counter()-started
    status = {0: "optimal", 1: "limit_reached", 2: "infeasible", 3: "unbounded", 4: "error"}.get(result.status, "error")
    assignment = order = None
    incumbent = None
    if result.x is not None:
        integral = all(abs(result.x[column]-round(result.x[column])) <= 1e-5
                       for column, flag in enumerate(integer) if flag)
        if integral:
            chosen = lambda key: result.x[variables[key]] > .5
            candidate = [next((j for j in scenario.eligible[i] if chosen(("x", i, j))), -1) for i in range(n)]
            successor = {i: h for i, h in arcs if chosen(("arc", i, h))}
            visits = []
            for j in range(k):
                starts = [i for i in range(n) if ("start", j, i) in variables and chosen(("start", j, i))]
                if len(starts) > 1:
                    integral = False
                    break
                if starts:
                    i = starts[0]
                    while i not in visits:
                        visits.append(i)
                        if i not in successor:
                            break
                        i = successor[i]
            if integral and len(visits) == n and all(j >= 0 for j in candidate):
                score = evaluate(scenario, candidate, visits)
                if score.feasible and abs(score.objective-result.fun) <= 1e-5*max(1., score.objective):
                    assignment, order, incumbent = score.assignment, score.visit_order, score.objective
    if status == "optimal" and assignment is None:
        raise RuntimeError("Routing MILP optimum failed shared tour evaluation.")
    finite = lambda value: float(value) if value is not None and math.isfinite(value) else None
    return OracleResult(status=status, optimal_objective=incumbent if status == "optimal" else None,
                        incumbent_objective=incumbent, assignment=assignment, visit_order=order,
                        lower_bound=finite(getattr(result, "mip_dual_bound", None)),
                        mip_gap=finite(getattr(result, "mip_gap", None)), solver_seconds=elapsed,
                        message=result.message, solver="SciPy/HiGHS multi-depot minimum-latency MILP",
                        scipy_version=scipy.__version__)
