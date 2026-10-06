"""Sequential benchmark orchestration and reproducible JSON/CSV exports."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import statistics
import sys

from . import __version__
from .algorithms import ALGORITHMS, Optimizer, SolverSettings
from .model import SIZES, generate_scenario, greedy_baseline


@dataclass
class BenchmarkResult:
    settings: SolverSettings
    scenarios: list = field(default_factory=list)
    runs: list = field(default_factory=list)
    baselines: dict = field(default_factory=dict)
    completed: bool = False
    planned_runs: int = 0

    def summaries(self):
        rows = []
        for scenario in self.scenarios:
            for algorithm in ALGORITHMS:
                runs = [r for r in self.runs if r.scenario_name == scenario.name
                        and r.scenario_seed == scenario.seed and r.algorithm == algorithm and r.completed]
                feasible = [r.best.objective for r in runs if r.best.feasible]
                times = [r.solver_seconds for r in runs]
                baseline = self.baselines[scenario.name]
                rows.append({"scenario": scenario.name, "algorithm": algorithm,
                             "runs": len(runs), "feasible_runs": len(feasible),
                             "feasible_percent": 100 * len(feasible) / len(runs) if runs else None,
                             "best_objective": min(feasible) if feasible else None,
                             "mean_objective": statistics.mean(feasible) if feasible else None,
                             "sample_std": statistics.stdev(feasible) if len(feasible) > 1 else None,
                             "mean_solver_seconds": statistics.mean(times) if times else None,
                             "baseline_feasible": baseline.feasible,
                             "baseline_objective": baseline.objective if baseline.feasible else None})
        return rows


def run_benchmark(settings=None, seeds=range(1, 11), sizes=tuple(SIZES), radius=6.0,
                  on_event=None, cancelled=None):
    """Callbacks receive (event_kind, payload); cancelled is a callable.

    Event kinds: scenario, run_started, progress, run_finished, finished.
    Callback and cancellation overhead is outside measured solver runtime.
    """
    settings = settings or SolverSettings()
    settings.validate()
    seeds, sizes = tuple(seeds), tuple(sizes)
    if not seeds or not sizes or len(set(seeds)) != len(seeds) or len(set(sizes)) != len(sizes):
        raise ValueError("Benchmark requires nonempty, unique seeds and scenario sizes.")
    emit = on_event or (lambda kind, payload: None)
    stop = cancelled or (lambda: False)
    result = BenchmarkResult(settings, planned_runs=len(seeds) * len(sizes) * len(ALGORITHMS))
    for size in sizes:
        if stop():
            emit("finished", result)
            return result
        scenario = generate_scenario(size, radius=radius)
        result.scenarios.append(scenario)
        result.baselines[size] = greedy_baseline(scenario)
        emit("scenario", scenario)
        for index, seed in enumerate(seeds):
            order = ALGORITHMS[index % 3:] + ALGORITHMS[:index % 3]
            for algorithm in order:
                if stop():
                    emit("finished", result)
                    return result
                optimizer = Optimizer(scenario, algorithm, seed, settings)
                emit("run_started", {"scenario": scenario, "algorithm": algorithm,
                                     "seed": seed, "number": len(result.runs) + 1,
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
    return {"schema_version": 1, "application_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "platform": platform.platform(),
            "model": "indivisible school assignment; Euclidean km; objective portion-km",
            "runtime": "solver work including decoding, repair and scoring; excludes GUI/pauses",
            "constraint_handling": "eligible kitchens, shared relocation repair, feasibility-first rank"}


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_benchmark(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata(), "completed": result.completed,
               "planned_runs": result.planned_runs, "settings": asdict(result.settings),
               "scenarios": [s.to_dict() for s in result.scenarios],
               "baselines": {name: asdict(score) for name, score in result.baselines.items()},
               "runs": [r.to_dict() for r in result.runs], "summaries": result.summaries()}
    (directory / "benchmark.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["scenario", "scenario_seed", "algorithm", "optimizer_seed", "completed",
              "evaluations", "feasible", "objective", "weighted_average_km", "overflow",
              "radius_violation", "solver_seconds"]
    scenarios = {(s.name, s.seed): s for s in result.scenarios}
    rows = []
    for run in result.runs:
        scenario = scenarios[run.scenario_name, run.scenario_seed]
        rows.append({"scenario": run.scenario_name, "scenario_seed": run.scenario_seed,
                     "algorithm": run.algorithm, "optimizer_seed": run.seed,
                     "completed": run.completed, "evaluations": run.evaluations,
                     "feasible": run.best.feasible,
                     "objective": run.best.objective if run.best.feasible else None,
                     "weighted_average_km": run.best.objective / scenario.total_demand if run.best.feasible else None,
                     "overflow": run.best.overflow, "radius_violation": run.best.radius_violation,
                     "solver_seconds": run.solver_seconds})
    _write_csv(directory / "runs.csv", rows, fields)
    summaries = result.summaries()
    summary_fields = ["scenario", "algorithm", "runs", "feasible_runs", "feasible_percent",
                      "best_objective", "mean_objective", "sample_std", "mean_solver_seconds",
                      "baseline_feasible", "baseline_objective"]
    _write_csv(directory / "summary.csv", summaries, summary_fields)
    return directory


def export_demo(scenario, runs, directory):
    """Export finished/partial demo runs through the same benchmark schema."""
    result = BenchmarkResult(runs[0].settings, [scenario], list(runs),
                             {scenario.name: greedy_baseline(scenario)},
                             all(r.completed for r in runs), len(runs))
    return export_benchmark(result, directory)
