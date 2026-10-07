"""Paired multi-instance benchmarks, certified gaps, and schema-v2 exports."""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import statistics
import sys

from . import __version__
from .algorithms import ALGORITHMS, ALGORITHM_LABELS, CONSTRAINT_POLICIES, Optimizer, SolverSettings
from .model import delivery_metrics
from .model import SIZES, generate_scenario, greedy_baseline
from .oracle import solve_exact, optimality_gap, lower_bound_gap


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def run_statistics(runs):
    feasible = [r.best.objective for r in runs if r.best.feasible]
    row = {"runs": len(runs), "feasible_runs": len(feasible),
           "feasible_percent": 100 * len(feasible) / len(runs) if runs else None,
           "best_objective": min(feasible) if feasible else None,
           "mean_objective": mean(feasible),
           "sample_std": statistics.stdev(feasible) if len(feasible) > 1 else None,
           "mean_solver_seconds": mean([r.solver_seconds for r in runs]),
           "mean_optimality_gap_percent": mean([r.optimality_gap_percent for r in runs]),
           "mean_lower_bound_gap_percent": mean([lower_bound_gap(r.best, r.oracle) for r in runs])}
    for key in ("duplicate_decoded_rate", "duplicate_phenotype_rate", "repair_attempt_rate",
                "repair_change_rate", "repair_mean_objective_delta", "pso_decoded_change_rate",
                "pso_phenotype_change_rate"):
        row["mean_" + key] = mean([r.diagnostics.get(key) for r in runs])
    return row


@dataclass
class BenchmarkResult:
    settings: SolverSettings
    scenarios: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    baselines: dict = field(default_factory=dict)
    completed: bool = False
    planned_runs: int = 0
    oracles: dict = field(default_factory=dict)
    policies: tuple = ("shared_repair",)
    optimizer_seeds: tuple = ()
    scenario_seed_plan: dict = field(default_factory=dict)

    def instance_summaries(self):
        rows = []
        for scenario in self.scenarios:
            oracle = self.oracles[scenario.instance_id]
            baseline = self.baselines[scenario.instance_id]
            for policy in self.policies:
                for algorithm in ALGORITHMS:
                    runs = [r for r in self.runs if r.scenario_id == scenario.instance_id
                            and r.algorithm == algorithm and r.settings.constraint_policy == policy and r.completed]
                    row = {"scenario": scenario.name, "scenario_seed": scenario.seed,
                           "instance_id": scenario.instance_id, "algorithm": algorithm,
                           "algorithm_label": ALGORITHM_LABELS[algorithm], "constraint_policy": policy,
                           "radius": scenario.radius, "capacity_slack": scenario.capacity_slack,
                           "difficulty": scenario.difficulty, "oracle_status": oracle.status,
                           "objective_unit": scenario.objective_unit, "objective_metric": scenario.objective_metric,
                           "map_source": scenario.road_source.get("source", "Synthetic"),
                           "osm_city": scenario.road_source.get("city"),
                           "optimal_objective": oracle.optimal_objective,
                           "oracle_lower_bound": oracle.lower_bound,
                           "baseline_feasible": baseline.feasible,
                           "baseline_objective": baseline.objective if baseline.feasible else None}
                    row.update(run_statistics(runs))
                    metrics = [delivery_metrics(scenario, run.best, include_routes=False) for run in runs if run.best.feasible]
                    for key in ("weighted_average_km", "weighted_average_minutes", "max_school_eta_minutes",
                                "active_sppg_count", "total_allocation_distance_km"):
                        row["mean_" + key] = mean([item[key] for item in metrics])
                    rows.append(row)
        return rows

    def summaries(self):
        """Raw J remains pooled for compatibility; gaps weight instances equally."""
        rows, instances = [], self.instance_summaries()
        sizes = tuple(dict.fromkeys(s.name for s in self.scenarios))
        for size in sizes:
            scenarios = [s for s in self.scenarios if s.name == size]
            baselines = [self.baselines[s.instance_id] for s in scenarios]
            for policy in self.policies:
                for algorithm in ALGORITHMS:
                    runs = [r for r in self.runs if r.scenario_name == size and r.algorithm == algorithm
                            and r.settings.constraint_policy == policy and r.completed]
                    groups = [row for row in instances if row["scenario"] == size
                              and row["algorithm"] == algorithm and row["constraint_policy"] == policy]
                    gaps = [row["mean_optimality_gap_percent"] for row in groups
                            if row["mean_optimality_gap_percent"] is not None]
                    row = {"scenario": size, "algorithm": algorithm, "algorithm_label": ALGORITHM_LABELS[algorithm],
                           "constraint_policy": policy, "instance_count": len(scenarios),
                           "objective_unit": scenarios[0].objective_unit, "objective_metric": scenarios[0].objective_metric,
                           "oracle_optimal_instances": sum(self.oracles[s.instance_id].status == "optimal" for s in scenarios),
                           "oracle_infeasible_instances": sum(self.oracles[s.instance_id].status == "infeasible" for s in scenarios),
                           "gap_instance_count": len(gaps),
                           "sample_std_instance_mean_gap_percent": statistics.stdev(gaps) if len(gaps) > 1 else None,
                           "baseline_feasible": all(b.feasible for b in baselines),
                           "baseline_objective": mean([b.objective for b in baselines if b.feasible])}
                    row.update(run_statistics(runs))
                    row["mean_optimality_gap_percent"] = mean(gaps)
                    row["mean_lower_bound_gap_percent"] = mean([g["mean_lower_bound_gap_percent"] for g in groups])
                    for key in ("weighted_average_km", "weighted_average_minutes", "max_school_eta_minutes",
                                "active_sppg_count", "total_allocation_distance_km"):
                        row["mean_" + key] = mean([group["mean_" + key] for group in groups])
                    rows.append(row)
        return rows


def run_benchmark(settings=None, seeds=None, sizes=tuple(SIZES), radius=6.0,
                  on_event=None, cancelled=None, *, optimizer_seeds=None, scenario_seeds=None,
                  instances_per_size=3, capacity_slack=None, difficulty="Standard", policies=None,
                  capture_search=False, map_source="OpenStreetMap", osm_city="Ciledug", school_count=None,
                  school_seed=101, school_demand=200, sppg_capacity=3000, opening_weight=10., distance_weight=1.):
    """Cross scenario seeds with independent optimizer seeds on every policy.

    seeds remains an alias for optimizer_seeds. Explicit scenario seeds may be a
    sequence shared by all sizes or a size-to-sequence mapping. MILP work is
    excluded from candidate budgets and heuristic runtime.
    """
    settings = settings or SolverSettings()
    settings.validate()
    if seeds is not None and optimizer_seeds is not None:
        raise ValueError("Use optimizer_seeds or its legacy seeds alias, not both.")
    opt_seeds = tuple(optimizer_seeds if optimizer_seeds is not None else seeds if seeds is not None else range(1, 11))
    sizes = tuple(sizes)
    policies = tuple(policies if policies is not None else (settings.constraint_policy,))
    if not sizes or any(s not in SIZES for s in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Benchmark sizes must be nonempty, valid and unique.")
    if not opt_seeds or len(set(opt_seeds)) != len(opt_seeds) or any(type(s) is not int for s in opt_seeds):
        raise ValueError("Optimizer seeds must be nonempty unique integers.")
    if type(instances_per_size) is not int or instances_per_size < 1:
        raise ValueError("Instances per size must be a positive integer.")
    if not policies or len(set(policies)) != len(policies) or any(p not in CONSTRAINT_POLICIES for p in policies):
        raise ValueError("Constraint policies must be nonempty, valid and unique.")
    if scenario_seeds is None:
        plan = {size: tuple(SIZES[size][2] + i for i in range(instances_per_size)) for size in sizes}
    elif isinstance(scenario_seeds, dict):
        if any(size not in scenario_seeds for size in sizes):
            raise ValueError("Provide scenario seeds for every requested size.")
        plan = {size: tuple(scenario_seeds[size]) for size in sizes}
    else:
        shared = tuple(scenario_seeds)
        plan = {size: shared for size in sizes}
    if any(not values or len(set(values)) != len(values) or any(type(s) is not int for s in values)
           for values in plan.values()):
        raise ValueError("Scenario seeds must be nonempty unique integers within each size.")
    emit, stop = on_event or (lambda kind, payload: None), cancelled or (lambda: False)
    result = BenchmarkResult(settings, planned_runs=sum(map(len, plan.values())) * len(opt_seeds) * 3 * len(policies),
                             policies=policies, optimizer_seeds=opt_seeds, scenario_seed_plan=plan)
    for size in sizes:
        for scenario_seed in plan[size]:
            if stop():
                emit("finished", result)
                return result
            scenario = generate_scenario(size, scenario_seed, radius, capacity_slack, difficulty,
                                         map_source=map_source, osm_city=osm_city, school_count=school_count,
                                         school_seed=school_seed, school_demand=school_demand, sppg_capacity=sppg_capacity,
                                         opening_weight=opening_weight, distance_weight=distance_weight)
            emit("oracle_started", scenario)
            oracle = solve_exact(scenario, settings.oracle_time_limit)
            result.scenarios.append(scenario)
            result.oracles[scenario.instance_id] = oracle
            result.baselines[scenario.instance_id] = greedy_baseline(scenario)
            emit("oracle_finished", {"scenario": scenario, "oracle": oracle})
            emit("scenario", scenario)
            for index, seed in enumerate(opt_seeds):
                order = ALGORITHMS[index % 3:] + ALGORITHMS[:index % 3]
                policy_order = policies[index % len(policies):] + policies[:index % len(policies)]
                for policy in policy_order:
                    configuration = replace(settings, constraint_policy=policy)
                    for algorithm in order:
                        if stop():
                            emit("finished", result)
                            return result
                        optimizer = Optimizer(scenario, algorithm, seed, configuration, capture_search=capture_search)
                        emit("run_started", {"scenario": scenario, "algorithm": algorithm, "seed": seed,
                                             "constraint_policy": policy, "number": len(result.runs) + 1,
                                             "total": result.planned_runs})
                        while not optimizer.done and not stop():
                            emit("progress", optimizer.advance())
                        if optimizer.best is not None:
                            run = optimizer.result()
                            result.runs.append(run)
                            emit("run_finished", run)
                        if stop():
                            emit("finished", result)
                            return result
    result.completed = True
    emit("finished", result)
    return result


def metadata():
    import numpy
    import scipy
    return {"schema_version": 2, "road_model_version": 5, "application_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "numpy_version": numpy.__version__, "scipy_version": scipy.__version__, "platform": platform.platform(),
            "model": "Ciledug OSM road graph; scenario objective_metric and road_source specify synthetic fixed-school facility-distance planning or legacy mapped-school multi-stop arrival model; capacities are meals, assignments indivisible; planning distance is sum of one-way shortest road paths, not vehicle mileage",
            "runtime": "heuristic work including diagnostics; excludes MILP, GUI and pauses",
            "scenario_preprocessing": "OSM download/import/projection and shortest-path tables are excluded from heuristic runtime",
            "gap_formula": "100 * (feasible objective - proven optimum) / abs(proven optimum)",
            "gap_aggregation": "equal-weight mean of instance mean gaps; infeasible/unproven gaps unavailable",
            "raw_objective_statistics": "pooled descriptive legacy fields; not independent-instance inference",
            "legacy_baselines": "first instance per size; complete baselines are in baselines_by_instance",
            "constraint_policies": {"shared_repair": "all algorithms capacity-neutral; shared relocation repair",
                                    "aco_capacity_aware": "legacy ACO residual capacity information; shared repair",
                                    "no_repair": "all capacity-neutral; feasibility-first ranking only"},
            "diagnostic_diversity": "unique ratio and mean pairwise normalized Hamming distance within evaluated batch"}


def _write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_benchmark(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    representative_ids = {}
    for scenario in result.scenarios:
        representative_ids.setdefault(scenario.name, scenario.instance_id)
    payload = {"metadata": metadata(), "completed": result.completed, "planned_runs": result.planned_runs,
               "settings": asdict(result.settings), "optimizer_seeds": result.optimizer_seeds,
               "scenario_seed_plan": result.scenario_seed_plan, "constraint_policies": result.policies,
               "scenarios": [dict(s.to_dict(), instance_id=s.instance_id) for s in result.scenarios],
               "oracles": {key: value.to_dict() for key, value in result.oracles.items()},
               "baselines": {name: asdict(result.baselines[key]) for name, key in representative_ids.items()},
               "legacy_baseline_instance_ids": representative_ids,
               "baselines_by_instance": {key: asdict(value) for key, value in result.baselines.items()},
               "runs": [dict(r.to_dict(), **delivery_metrics(next(s for s in result.scenarios if s.instance_id == r.scenario_id), r.best)) for r in result.runs], "summaries": result.summaries(),
               "instance_summaries": result.instance_summaries()}
    (directory / "benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    scenarios = {s.instance_id: s for s in result.scenarios}
    rows, diagnostics = [], []
    for run in result.runs:
        scenario = scenarios[run.scenario_id]
        row = {"scenario": run.scenario_name, "scenario_seed": run.scenario_seed,
               "algorithm": run.algorithm, "optimizer_seed": run.seed, "completed": run.completed,
               "evaluations": run.evaluations, "feasible": run.best.feasible,
               "objective": run.best.objective if run.best.feasible else None,
               "weighted_average_km": delivery_metrics(scenario, run.best)["weighted_average_km"],
               "overflow": run.best.overflow, "radius_violation": run.best.radius_violation,
               "solver_seconds": run.solver_seconds, "instance_id": run.scenario_id,
               "algorithm_label": ALGORITHM_LABELS[run.algorithm], "constraint_policy": run.settings.constraint_policy,
               "oracle_status": run.oracle.status, "optimal_objective": run.optimal_objective,
               "absolute_optimality_gap": optimality_gap(run.best, run.oracle)[0],
               "optimality_gap_percent": run.optimality_gap_percent,
               "capacity_slack": scenario.capacity_slack, "difficulty": scenario.difficulty, "radius": scenario.radius}
        row.update(map_source=scenario.road_source.get("source", "Synthetic"),
                   osm_city=scenario.road_source.get("city"),
                   road_data_sha256=scenario.road_source.get("raw_sha256"),
                   school_location_model=scenario.road_source.get("school_locations"))
        row["lower_bound_gap_percent"] = lower_bound_gap(run.best, run.oracle)
        row.update(delivery_metrics(scenario, run.best, include_routes=False))
        row.update(run.diagnostics)
        rows.append(row)
        for snapshot in run.snapshots:
            diagnostics.append({"instance_id": run.scenario_id, "scenario": run.scenario_name,
                                "scenario_seed": run.scenario_seed, "optimizer_seed": run.seed,
                                "algorithm": run.algorithm, "constraint_policy": run.settings.constraint_policy,
                                "iteration": snapshot.iteration, "evaluations": snapshot.evaluations,
                                "optimality_gap_percent": snapshot.optimality_gap_percent, **snapshot.diagnostics})
    legacy_run_fields = ["scenario", "scenario_seed", "algorithm", "optimizer_seed", "completed", "evaluations",
                         "feasible", "objective", "weighted_average_km", "overflow", "radius_violation", "solver_seconds"]
    run_fields = legacy_run_fields + list(dict.fromkeys(key for row in rows for key in row if key not in legacy_run_fields))
    _write_csv(directory / "runs.csv", rows, run_fields)
    summaries = result.summaries()
    legacy_summary_fields = ["scenario", "algorithm", "runs", "feasible_runs", "feasible_percent",
                             "best_objective", "mean_objective", "sample_std", "mean_solver_seconds",
                             "baseline_feasible", "baseline_objective"]
    summary_fields = legacy_summary_fields + list(dict.fromkeys(
        key for row in summaries for key in row if key not in legacy_summary_fields))
    _write_csv(directory / "summary.csv", summaries, summary_fields)
    instances = result.instance_summaries()
    _write_csv(directory / "instance_summary.csv", instances,
               None if instances else ["instance_id", "scenario_seed", "algorithm", "optimal_objective"])
    _write_csv(directory / "diagnostics.csv", diagnostics, None if diagnostics else ["instance_id", "evaluations"])
    return directory


def export_demo(scenario, runs, directory):
    if not runs:
        raise ValueError("No demo runs to export.")
    result = BenchmarkResult(runs[0].settings, [scenario], list(runs),
                             {scenario.instance_id: greedy_baseline(scenario)},
                             all(r.completed for r in runs), len(runs),
                             {scenario.instance_id: runs[0].oracle or solve_exact(scenario)},
                             tuple(dict.fromkeys(r.settings.constraint_policy for r in runs)),
                             tuple(dict.fromkeys(r.seed for r in runs)), {scenario.name: (scenario.seed,)})
    return export_benchmark(result, directory)
