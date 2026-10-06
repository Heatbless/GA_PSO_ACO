"""GUI launcher and headless benchmark CLI."""

import argparse
from datetime import datetime
from pathlib import Path

from .algorithms import SolverSettings
from .experiments import export_benchmark, run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Compare GA, PSO and ACO for SPPG kitchen assignment.")
    parser.add_argument("--benchmark", action="store_true", help="Run without the Tkinter interface")
    parser.add_argument("--budget", type=int, default=5000, help="Candidate evaluations per algorithm run")
    parser.add_argument("--population", type=int, default=30, help="Population, swarm, or ant count")
    parser.add_argument("--seeds", type=int, default=10, help="Number of optimizer seeds, starting at 1")
    parser.add_argument("--sizes", nargs="+", choices=("Small", "Medium", "Large"),
                        default=["Small", "Medium", "Large"])
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--output", type=Path, help="Directory for benchmark.json, runs.csv, summary.csv")
    args = parser.parse_args()
    if not args.benchmark:
        from .gui import Application
        Application().mainloop()
        return
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    settings = SolverSettings(budget=args.budget, population=args.population)

    def report(kind, payload):
        if kind == "run_started":
            print(f"[{payload['number']:2}/{payload['total']}] {payload['scenario'].name:6} "
                  f"{payload['algorithm']:3} seed={payload['seed']}", flush=True)
        elif kind == "run_finished":
            score = f"J={payload.best.objective:.2f}" if payload.best.feasible else f"INFEASIBLE overflow={payload.best.overflow}"
            print(f"    {score}; {payload.evaluations} evaluations; {payload.solver_seconds:.3f}s", flush=True)

    try:
        result = run_benchmark(settings, range(1, args.seeds + 1), args.sizes, args.radius, report)
        directory = args.output or Path("results") / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
        export_benchmark(result, directory)
    except ValueError as error:
        parser.error(str(error))
    print(f"\nExported {len(result.runs)} runs to {directory.resolve()}")
    for row in result.summaries():
        mean = "n/a" if row["mean_objective"] is None else f"{row['mean_objective']:.2f}"
        print(f"{row['scenario']:6} {row['algorithm']:3}: feasible {row['feasible_percent']:.0f}%, mean J {mean}")


if __name__ == "__main__":
    main()
