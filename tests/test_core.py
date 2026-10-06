import csv
import itertools
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sppg_compare.algorithms import ALGORITHMS, Optimizer, SolverSettings, solve
from sppg_compare.experiments import export_benchmark, run_benchmark
from sppg_compare.model import (Kitchen, School, Scenario, SIZES, evaluate,
                                generate_scenario, greedy_baseline, repair)


def tiny_scenario():
    return Scenario("Tiny", 7, 6.0,
                    (Kitchen("K1", 0, 0, 100), Kitchen("K2", 4, 0, 100)),
                    (School("S1", 1, 0, 60), School("S2", 3, 0, 60)))


class ModelTests(unittest.TestCase):
    def test_hand_calculated_objective_and_capacity(self):
        scenario = tiny_scenario()
        score = evaluate(scenario, (0, 1))
        self.assertEqual(score.objective, 120)
        self.assertEqual(score.loads, (60, 60))
        self.assertTrue(score.feasible)
        overloaded = evaluate(scenario, (0, 0))
        self.assertEqual(overloaded.objective, 240)
        self.assertEqual(overloaded.overflow, 20)
        self.assertLess(score.rank, overloaded.rank)

    def test_radius_boundary_and_violation(self):
        scenario = Scenario("Boundary", 1, 6.0,
                            (Kitchen("K1", 0, 0, 100), Kitchen("K2", 20, 0, 100)),
                            (School("S1", 6, 0, 30),))
        self.assertEqual(scenario.eligible, ((0,),))
        self.assertTrue(evaluate(scenario, (0,)).feasible)
        self.assertEqual(evaluate(scenario, (1,)).radius_violation, 30)
        with self.assertRaisesRegex(ValueError, "outside"):
            Scenario("Invalid", 1, 5.9, (Kitchen("K1", 0, 0, 100),),
                     (School("S1", 6, 0, 30),))

    def test_repair_is_deterministic_and_eligible(self):
        scenario = tiny_scenario()
        self.assertEqual(repair(scenario, (0, 0)), (0, 1))
        self.assertEqual(repair(scenario, (0, 0)), repair(scenario, (0, 0)))
        self.assertTrue(evaluate(scenario, repair(scenario, (1, 1))).feasible)

    def test_validation(self):
        scenario = tiny_scenario()
        for assignment in ((0,), (0, 2), (0, -1), (0, 1.0)):
            with self.assertRaises(ValueError):
                evaluate(scenario, assignment)
        for radius in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                generate_scenario(radius=radius)
        with self.assertRaises(ValueError):
            Scenario("Invalid", 1, 6, (Kitchen("K1", 0, 0, 0),), (School("S1", 0, 0, 30),))

    def test_generated_scenarios_roundtrip_and_challenge(self):
        for size in SIZES:
            for seed in (None, 1, 2, 10):
                scenario = generate_scenario(size, seed)
                again = generate_scenario(size, seed)
                self.assertEqual(scenario, again)
                self.assertEqual(Scenario.from_dict(scenario.to_dict()), scenario)
                self.assertGreaterEqual(sum(len(e) > 1 for e in scenario.eligible), len(scenario.schools) // 3)
                nearest = tuple(min(row, key=lambda j: scenario.distances[i][j])
                                for i, row in enumerate(scenario.eligible))
                self.assertGreater(evaluate(scenario, nearest).overflow, 0)
                self.assertGreaterEqual(sum(k.capacity for k in scenario.kitchens), scenario.total_demand)
                self.assertEqual(len(greedy_baseline(scenario).assignment), len(scenario.schools))


class SolverTests(unittest.TestCase):
    def test_exact_budgets_including_partial_populations(self):
        scenario = tiny_scenario()
        for algorithm in ALGORITHMS:
            for budget in (1, 7, 30, 31, 59, 100, 127):
                with self.subTest(algorithm=algorithm, budget=budget):
                    result = solve(scenario, algorithm, settings=SolverSettings(budget=budget))
                    self.assertEqual(result.evaluations, budget)
                    self.assertEqual(result.snapshots[-1].evaluations, budget)
                    self.assertEqual(result.history[-1]["evaluations"], budget)
                    self.assertTrue(result.completed)
                    self.assertEqual(result.best, evaluate(scenario, result.best.assignment))

    def test_same_seed_same_search(self):
        scenario = generate_scenario("Medium")
        for algorithm in ALGORITHMS:
            settings = SolverSettings(budget=233)
            a = solve(scenario, algorithm, 5, settings)
            b = solve(scenario, algorithm, 5, settings)
            self.assertEqual(a.best, b.best)
            self.assertEqual(a.history, b.history)
            self.assertEqual([(s.evaluations, s.best) for s in a.snapshots],
                             [(s.evaluations, s.best) for s in b.snapshots])

    def test_tiny_exhaustive_oracle_and_incumbents(self):
        scenario = tiny_scenario()
        scores = [evaluate(scenario, a) for a in itertools.product(*scenario.eligible)]
        optimum = min((s.objective for s in scores if s.feasible))
        self.assertEqual(optimum, 120)
        for algorithm in ALGORITHMS:
            result = solve(scenario, algorithm, settings=SolverSettings(budget=91))
            self.assertTrue(result.best.feasible)
            self.assertEqual(result.best.objective, optimum)
            ranks = [s.best.rank for s in result.snapshots]
            self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_radius_masks_and_impossible_capacity(self):
        scenario = Scenario("Impossible", 1, 2,
                            (Kitchen("K1", 0, 0, 50), Kitchen("K2", 10, 0, 50)),
                            (School("S1", 0, 0, 80), School("S2", 10, 0, 80)))
        for algorithm in ALGORITHMS:
            result = solve(scenario, algorithm, settings=SolverSettings(budget=101))
            self.assertFalse(result.best.feasible)
            self.assertEqual(result.best.assignment, (0, 1))
            self.assertEqual(result.best.overflow, 60)
            self.assertTrue(all(h["objective"] is None for h in result.history))

    def test_random_assignment_properties(self):
        scenario = generate_scenario("Large")
        for algorithm in ALGORITHMS:
            result = solve(scenario, algorithm, settings=SolverSettings(budget=150))
            for snapshot in result.snapshots:
                self.assertEqual(snapshot.best.radius_violation, 0)
                self.assertEqual(sum(snapshot.best.loads), scenario.total_demand)
                self.assertTrue(all(j in scenario.eligible[i] for i, j in enumerate(snapshot.best.assignment)))

    def test_settings_validation_and_partial_result(self):
        for settings in (SolverSettings(budget=0), SolverSettings(population=2),
                         SolverSettings(elites=30), SolverSettings(mutation=2),
                         SolverSettings(inertia=math.nan), SolverSettings(evaporation=1),
                         SolverSettings(alpha=1e308)):
            with self.assertRaises(ValueError):
                settings.validate()
        optimizer = Optimizer(tiny_scenario(), "GA", settings=SolverSettings(budget=100))
        optimizer.advance()
        self.assertFalse(optimizer.result().completed)


class ExperimentTests(unittest.TestCase):
    def test_benchmark_order_and_exports(self):
        starts = []
        result = run_benchmark(SolverSettings(budget=73), seeds=(1, 2), sizes=("Small", "Medium"),
                               on_event=lambda kind, value: starts.append(value["algorithm"]) if kind == "run_started" else None)
        self.assertTrue(result.completed)
        self.assertEqual(len(result.runs), 12)
        self.assertEqual(starts[:6], ["GA", "PSO", "ACO", "PSO", "ACO", "GA"])
        self.assertTrue(all(r.evaluations == 73 for r in result.runs))
        with tempfile.TemporaryDirectory() as directory:
            export_benchmark(result, directory)
            payload = json.loads((Path(directory) / "benchmark.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["settings"]["budget"], 73)
            self.assertEqual(len(payload["runs"]), 12)
            scenarios = {s["name"]: Scenario.from_dict(s) for s in payload["scenarios"]}
            for run in payload["runs"]:
                score = evaluate(scenarios[run["scenario_name"]], run["best"]["assignment"])
                self.assertEqual(score.objective, run["best"]["objective"])
                self.assertEqual(list(score.loads), run["best"]["loads"])
            with (Path(directory) / "runs.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["evaluations"] == "73" for row in rows))
            self.assertEqual(len(payload["summaries"]), 6)

    def test_cancelled_benchmark_retains_partial_run(self):
        state = {"stop": False}

        def event(kind, value):
            if kind == "progress":
                state["stop"] = True

        result = run_benchmark(SolverSettings(budget=100), seeds=(1,), sizes=("Small",),
                               on_event=event, cancelled=lambda: state["stop"])
        self.assertFalse(result.completed)
        self.assertEqual(len(result.runs), 1)
        self.assertFalse(result.runs[0].completed)
        self.assertTrue(all(row["runs"] == 0 for row in result.summaries()))

    def test_invalid_benchmark_arguments(self):
        for seeds, sizes in (((), ("Small",)), ((1, 1), ("Small",)), ((1,), ())):
            with self.assertRaises(ValueError):
                run_benchmark(seeds=seeds, sizes=sizes)


if __name__ == "__main__":
    unittest.main()
