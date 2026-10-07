"""GUI launcher and headless benchmark CLI."""

import argparse
from datetime import datetime
from pathlib import Path

from .algorithms import ALGORITHM_LABELS, CONSTRAINT_POLICIES, SolverSettings
from .model import DIFFICULTIES
from .experiments import export_benchmark, run_benchmark
from .osm import CITIES


def main():
    parser = argparse.ArgumentParser(description="Compare GA, discrete PSO and ACO on the Ciledug OSM SPPG scenario.")
    parser.add_argument("--benchmark", action="store_true", help="Run without the Tkinter interface")
    parser.add_argument("--capture-search", action="store_true", help="Record PSO/ACO animation state in JSON (larger exports)")
    parser.add_argument("--map-source", choices=("OpenStreetMap",), default="OpenStreetMap", help=argparse.SUPPRESS)
    parser.add_argument("--osm-city", choices=tuple(CITIES), default="Ciledug", help="OSM area; Ciledug includes Ciledug Raya")
    parser.add_argument("--budget", type=int, default=5000, help="Candidate evaluations per algorithm run")
    parser.add_argument("--population", type=int, default=30, help="Population, swarm, or ant count")
    parser.add_argument("--seeds", type=int, help="Legacy option: number of optimizer seeds starting at 1")
    parser.add_argument("--optimizer-seeds", nargs="+", type=int, help="Explicit independent optimizer seeds")
    parser.add_argument("--scenario-seeds", nargs="+", type=int, help="Explicit problem-instance seeds for each size")
    parser.add_argument("--instances", type=int, default=3, help="Problem instances per size (default: 3)")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="Standard")
    parser.add_argument("--capacity-slack", type=float, help="Override difficulty slack; e.g. 0.05 = 5%%")
    parser.add_argument("--constraint-policies", nargs="+", choices=CONSTRAINT_POLICIES, default=["shared_repair"])
    parser.add_argument("--oracle-time-limit", type=float, default=60, help="MILP seconds per instance; unproven gaps stay blank")
    parser.add_argument("--sizes", nargs="+", choices=("Small", "Medium", "Large"),
                        default=["Small", "Medium", "Large"])
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--schools", type=int, default=30, help="Fixed synthetic school count (default: 30)")
    parser.add_argument("--school-seed", type=int, default=101, help="Independent fixed school-location seed")
    parser.add_argument("--school-demand", type=int, default=200, help="Meals per synthetic school")
    parser.add_argument("--sppg-capacity", type=int, default=3000, help="Meals per candidate SPPG")
    parser.add_argument("--opening-weight", type=float, default=10., help="Weight per active SPPG")
    parser.add_argument("--distance-weight", type=float, default=1., help="Weight per km of allocation distance")
    parser.add_argument("--mapped-school-tours", action="store_true", help="Run the legacy mapped-school arrival model")
    parser.add_argument("--output", type=Path, help="Directory for benchmark.json, runs.csv, summary.csv")
    args = parser.parse_args()
    if not args.benchmark:
        from .gui import Application
        Application().mainloop()
        return
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.seeds is not None and args.optimizer_seeds is not None:
        parser.error("Use --optimizer-seeds or --seeds, not both")
    optimizer_seeds = args.optimizer_seeds if args.optimizer_seeds is not None else range(1, (args.seeds or 10) + 1)
    settings = SolverSettings(budget=args.budget, population=args.population, oracle_time_limit=args.oracle_time_limit)

    def report(kind, payload):
        if kind == "oracle_finished":
            oracle = payload["oracle"]
            print(f"MILP {payload['scenario'].name} scenario seed={payload['scenario'].seed}: "
                  f"{oracle.status}, optimum={oracle.optimal_objective}, {oracle.solver_seconds:.3f}s", flush=True)
        elif kind == "run_started":
            print(f"[{payload['number']:2}/{payload['total']}] {payload['scenario'].name:6} "
                  f"scenario={payload['scenario'].seed} {ALGORITHM_LABELS[payload['algorithm']]} "
                  f"optimizer={payload['seed']} policy={payload['constraint_policy']}", flush=True)
        elif kind == "run_finished":
            score = f"J={payload.best.objective:.2f}" if payload.best.feasible else f"INFEASIBLE overflow={payload.best.overflow}"
            gap = "unavailable" if payload.optimality_gap_percent is None else f"{payload.optimality_gap_percent:.3f}%"
            print(f"    {score}; gap={gap}; {payload.evaluations} evaluations; {payload.solver_seconds:.3f}s", flush=True)

    try:
        result = run_benchmark(settings, sizes=args.sizes, radius=args.radius, on_event=report,
                               optimizer_seeds=optimizer_seeds, scenario_seeds=args.scenario_seeds,
                               instances_per_size=args.instances, difficulty=args.difficulty,
                               capacity_slack=args.capacity_slack, policies=args.constraint_policies,
                               capture_search=args.capture_search, map_source=args.map_source, osm_city=args.osm_city,
                               school_count=None if args.mapped_school_tours else args.schools, school_seed=args.school_seed,
                               school_demand=args.school_demand, sppg_capacity=args.sppg_capacity,
                               opening_weight=args.opening_weight, distance_weight=args.distance_weight)
        directory = args.output or Path("results") / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
        export_benchmark(result, directory)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"\nExported {len(result.runs)} runs to {directory.resolve()}")
    for row in result.summaries():
        mean = "n/a" if row["mean_objective"] is None else f"{row['mean_objective']:.2f}"
        gap = "n/a" if row["mean_optimality_gap_percent"] is None else f"{row['mean_optimality_gap_percent']:.3f}%"
        print(f"{row['scenario']:6} {row['algorithm']:3} {row['constraint_policy']}: "
              f"feasible {row['feasible_percent']:.0f}%, mean J {mean}, mean instance gap {gap}")


if __name__ == "__main__":
    main()
