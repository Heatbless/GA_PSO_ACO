"""MILP, diagnostics, paired experimental design, and geography regression tests."""

import csv
import itertools
import json
import tempfile
import unittest
from tests.support import OsmFixtureTests
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import milp

from sppg_compare.algorithms import ALGORITHMS, CONSTRAINT_POLICIES, Optimizer, RunResult, SolverSettings, solve
from sppg_compare.diagnostics import SearchDiagnostics, hamming_diversity
from sppg_compare.experiments import BenchmarkResult, export_benchmark, run_benchmark
from sppg_compare.model import Kitchen, Scenario, School, evaluate, generate_scenario, repair
from sppg_compare.oracle import OracleResult, optimality_gap, solve_exact
from tests.test_core import tiny_scenario


class OracleTests(OsmFixtureTests):
    def test_matches_exhaustive_enumeration(self):
        scenarios = [tiny_scenario(), Scenario("Unequal", 1, 6,
                     (Kitchen("K1", 0, 0, 7), Kitchen("K2", 4, 0, 8)),
                     (School("S1", 1, 0, 4), School("S2", 2, 0, 3), School("S3", 3, 0, 5)))]
        for scenario in scenarios:
            optimum = min(e.objective for a in itertools.product(*scenario.eligible)
                          if (e := evaluate(scenario, a)).feasible)
            oracle = solve_exact(scenario)
            self.assertEqual(oracle.status, "optimal")
            self.assertAlmostEqual(oracle.optimal_objective, optimum)
            self.assertTrue(evaluate(scenario, oracle.assignment).feasible)
            self.assertAlmostEqual(oracle.lower_bound, optimum)

    def test_infeasible_and_radius_edges(self):
        scenario = Scenario("Radius-infeasible", 1, 2,
                            (Kitchen("K1", 0, 0, 50), Kitchen("K2", 10, 0, 100)),
                            (School("S1", 0, 0, 80),))
        oracle = solve_exact(scenario)
        self.assertEqual(oracle.status, "infeasible")
        self.assertIsNone(oracle.optimal_objective)
        self.assertIsNone(oracle.assignment)
        self.assertEqual(optimality_gap(evaluate(scenario, (0,)), oracle), (None, None))

    def test_time_limit_incumbent_is_not_an_optimum(self):
        scenario = replace(tiny_scenario(), name="Mock time limit")
        fake = SimpleNamespace(status=1, x=np.array([1, 0, 0, 1]), message="Time limit reached",
                               mip_dual_bound=100.0, mip_gap=1 / 6)
        with patch("scipy.optimize.milp", return_value=fake):
            oracle = solve_exact(scenario, 0.123)
        self.assertEqual(oracle.status, "limit_reached")
        self.assertIsNone(oracle.optimal_objective)
        self.assertEqual(oracle.incumbent_objective, 120)
        self.assertEqual(optimality_gap(evaluate(scenario, (0, 1)), oracle), (None, None))

    def test_gap_formula_and_zero_optimum(self):
        scenario = tiny_scenario()
        oracle = solve_exact(scenario)
        self.assertEqual(optimality_gap(evaluate(scenario, (1, 0)), oracle), (240, 200))
        zero = Scenario("Zero", 1, 6, (Kitchen("K1", 0, 0, 5), Kitchen("K2", 1, 0, 5)),
                        (School("S1", 0, 0, 5),))
        exact = solve_exact(zero)
        self.assertEqual(optimality_gap(evaluate(zero, (0,)), exact), (0, 0))
        self.assertEqual(optimality_gap(evaluate(zero, (1,)), exact), (5, None))
        with self.assertRaises(RuntimeError):
            optimality_gap(evaluate(scenario, (0, 1)), OracleResult("optimal", optimal_objective=130))


class GeographyTests(OsmFixtureTests):
    def test_radius_changes_only_eligibility(self):
        for size in ("Small", "Medium", "Large"):
            short = generate_scenario(size, radius=1.3)
            long = generate_scenario(size, radius=8)
            self.assertEqual(short.kitchens, long.kitchens)
            self.assertEqual(short.schools, long.schools)
            self.assertEqual(short.distances, long.distances)
            self.assertNotEqual(short.eligible, long.eligible)
            self.assertTrue(all(set(a) <= set(b) for a, b in zip(short.eligible, long.eligible)))

    def test_slack_changes_capacities_not_geometry_or_demand(self):
        tight = generate_scenario("Medium", difficulty="Tight")
        easy = generate_scenario("Medium", difficulty="Easy")
        self.assertEqual(tight.schools, easy.schools)
        self.assertEqual([(k.x, k.y) for k in tight.kitchens], [(k.x, k.y) for k in easy.kitchens])
        self.assertEqual(tight.eligible, easy.eligible)
        self.assertTrue(all(a.capacity < b.capacity for a, b in zip(tight.kitchens, easy.kitchens)))
        self.assertEqual(solve_exact(generate_scenario("Small", difficulty="Overloaded")).status, "infeasible")
        for slack in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                generate_scenario(capacity_slack=slack)


class DiagnosticTests(OsmFixtureTests):
    def test_hand_calculated_repair_duplicates_and_pso_plateau(self):
        scenario = tiny_scenario()
        raw = evaluate(scenario, (0, 0))
        fixed = evaluate(scenario, repair(scenario, raw.assignment))
        diagnostics = SearchDiagnostics()
        diagnostics.record(raw, fixed, True)
        diagnostics.record(raw, fixed, True, raw.assignment, fixed.assignment, True)
        diagnostics.record(fixed, fixed, True)
        metrics = diagnostics.snapshot(3)
        self.assertEqual(metrics["duplicate_decoded_evaluations"], 1)
        self.assertEqual(metrics["duplicate_phenotype_evaluations"], 2)
        self.assertEqual(metrics["repair_attempts"], 2)
        self.assertEqual(metrics["repair_changes"], 2)
        self.assertEqual(metrics["repair_successes"], 2)
        self.assertEqual(metrics["repair_mean_objective_delta"], -120)
        self.assertEqual(metrics["repair_absolute_objective_delta_sum"], 240)
        self.assertEqual(metrics["pso_position_changes"], 1)
        self.assertEqual(metrics["pso_decoded_changes"], 0)
        self.assertEqual(metrics["pso_phenotype_changes"], 0)
        self.assertEqual(hamming_diversity([(0, 0), (1, 1)]), 1)
        self.assertEqual(hamming_diversity([(0, 0), (0, 1)]), 0.5)

    def test_budget_counts_duplicates_and_pso_update_denominator(self):
        for algorithm in ALGORITHMS:
            run = solve(tiny_scenario(), algorithm, settings=SolverSettings(oracle_time_limit=.15, budget=101))
            d = run.diagnostics
            self.assertEqual(d["unique_decoded_assignments"] + d["duplicate_decoded_evaluations"], 101)
            self.assertEqual(d["unique_phenotypes"] + d["duplicate_phenotype_evaluations"], 101)
            self.assertLessEqual(d["repair_successes"], d["repair_attempts"])
            self.assertEqual(run.optimal_objective, 120)
            self.assertEqual(run.optimality_gap_percent, 0)
            self.assertTrue(all(0 <= s.diagnostics["decoded_hamming_diversity"] <= 1 for s in run.snapshots))
            if algorithm == "PSO":
                self.assertEqual(d["pso_updates"], 71)

    def test_fair_policy_changes_only_aco_construction(self):
        scenario = tiny_scenario()
        settings = SolverSettings(oracle_time_limit=.15, budget=91)
        for algorithm in ("GA", "PSO"):
            neutral = solve(scenario, algorithm, 7, settings)
            legacy = solve(scenario, algorithm, 7, replace(settings, constraint_policy="aco_capacity_aware"))
            self.assertEqual(neutral.best, legacy.best)
            self.assertEqual(neutral.history, legacy.history)
            self.assertEqual(neutral.diagnostics, legacy.diagnostics)
        neutral = Optimizer(scenario, "ACO", 1, settings)
        aware = Optimizer(scenario, "ACO", 1, replace(settings, constraint_policy="aco_capacity_aware"))
        neutral.advance()
        aware.advance()
        self.assertEqual(aware.diagnostics.totals["pre_repair_feasible"], 30)
        self.assertLess(neutral.diagnostics.totals["pre_repair_feasible"], 30)

    def test_no_repair_protocol(self):
        optimizer = Optimizer(tiny_scenario(), "GA", settings=SolverSettings(oracle_time_limit=.15, budget=1, constraint_policy="no_repair"))
        optimizer._score((0, 0))
        result = optimizer.result()
        self.assertFalse(result.best.feasible)
        self.assertEqual(result.diagnostics["repair_attempts"], 0)
        self.assertEqual(result.diagnostics["repair_changes"], 0)
        self.assertIsNone(result.optimality_gap_percent)


class MultiInstanceTests(OsmFixtureTests):
    def test_cancel_before_first_instance_keeps_legacy_headers(self):
        result = run_benchmark(cancelled=lambda: True)
        self.assertFalse(result.completed)
        self.assertEqual(result.runs, [])
        with tempfile.TemporaryDirectory() as directory:
            export_benchmark(result, directory)
            for filename, field in (("runs.csv", "optimizer_seed"), ("summary.csv", "mean_objective"),
                                    ("instance_summary.csv", "instance_id"), ("diagnostics.csv", "evaluations")):
                with (Path(directory) / filename).open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertIn(field, reader.fieldnames)
                    self.assertEqual(list(reader), [])

    def test_aggregate_gaps_weight_instances_not_restarts(self):
        scenarios = [replace(tiny_scenario(), name="Small", seed=101),
                     replace(tiny_scenario(), name="Small", seed=102)]
        oracles = {s.instance_id: solve_exact(s) for s in scenarios}
        baselines = {s.instance_id: evaluate(s, (0, 1)) for s in scenarios}
        settings = SolverSettings()
        runs = []
        for scenario, seeds, assignment in ((scenarios[0], (1, 2, 3), (0, 1)), (scenarios[1], (1,), (1, 0))):
            for seed in seeds:
                runs.append(RunResult("GA", seed, settings, scenario.name, scenario.seed,
                                      evaluate(scenario, assignment), 5000, 0.01, oracle=oracles[scenario.instance_id],
                                      scenario_id=scenario.instance_id))
        result = BenchmarkResult(settings, scenarios, runs, baselines, True, 4, oracles)
        row = next(row for row in result.summaries() if row["algorithm"] == "GA")
        # Instance means are 0% and 200%; pooling four restarts would incorrectly give 50%.
        self.assertEqual(row["mean_optimality_gap_percent"], 100)
        self.assertEqual(row["gap_instance_count"], 2)

    def test_seed_cross_product_and_exports(self):
        result = run_benchmark(SolverSettings(oracle_time_limit=.15, budget=31), sizes=("Small",), scenario_seeds=(501, 502),
                               optimizer_seeds=(7, 9), policies=CONSTRAINT_POLICIES)
        self.assertEqual(len(result.scenarios), 2)
        self.assertEqual(len(result.runs), 36)
        self.assertEqual(result.planned_runs, 36)
        expected = set(itertools.product((501, 502), (7, 9), ALGORITHMS, CONSTRAINT_POLICIES))
        actual = {(r.scenario_seed, r.seed, r.algorithm, r.settings.constraint_policy) for r in result.runs}
        self.assertEqual(expected, actual)
        self.assertTrue(all(r.evaluations == 31 for r in result.runs))
        self.assertEqual(len(result.instance_summaries()), 18)
        self.assertEqual(len(result.summaries()), 9)
        with tempfile.TemporaryDirectory() as directory:
            export_benchmark(result, directory)
            payload = json.loads((Path(directory) / "benchmark.json").read_text())
            self.assertEqual(payload["metadata"]["schema_version"], 2)
            self.assertEqual(payload["optimizer_seeds"], [7, 9])
            self.assertEqual(payload["scenario_seed_plan"], {"Small": [501, 502]})
            self.assertEqual(len(payload["oracles"]), 2)
            self.assertIn("Small", payload["baselines"])
            self.assertEqual(len(payload["baselines_by_instance"]), 2)
            self.assertEqual(payload["baselines"]["Small"],
                             payload["baselines_by_instance"][payload["legacy_baseline_instance_ids"]["Small"]])
            self.assertTrue(all("optimality_gap_percent" in r and "diagnostics" in r for r in payload["runs"]))
            self.assertIn("preference-score/discrete", next(r["algorithm_label"] for r in payload["runs"] if r["algorithm"] == "PSO"))
            for filename in ("runs.csv", "summary.csv", "instance_summary.csv", "diagnostics.csv"):
                with (Path(directory) / filename).open(newline="") as handle:
                    self.assertGreater(len(list(csv.DictReader(handle))), 0)

    def test_oracle_backend_runs_once_per_instance(self):
        with patch("scipy.optimize.milp", wraps=milp) as backend:
            result = run_benchmark(SolverSettings(oracle_time_limit=.15, budget=1), sizes=("Small",),
                                   scenario_seeds=(8001, 8002), optimizer_seeds=(1, 2))
        self.assertEqual(len(result.runs), 12)
        self.assertEqual(backend.call_count, 2)

    def test_invalid_seed_plans_and_policy(self):
        for kwargs in ({"scenario_seeds": []}, {"scenario_seeds": [1, 1]},
                       {"scenario_seeds": {"Small": [1]}}, {"instances_per_size": 0},
                       {"policies": ["unknown"]}, {"seeds": [1], "optimizer_seeds": [2]}):
            with self.assertRaises(ValueError):
                run_benchmark(**kwargs)


if __name__ == "__main__":
    unittest.main()
