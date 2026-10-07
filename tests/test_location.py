import itertools
import unittest
from tests.support import OsmFixtureTests
from dataclasses import replace

from sppg_compare.model import Scenario, Kitchen, School, evaluate, repair, generate_scenario, delivery_metrics
from sppg_compare.oracle import solve_exact
from sppg_compare.algorithms import solve, SolverSettings
from sppg_compare.diagnostics import SearchDiagnostics


class LocationTests(OsmFixtureTests):
    def fixture(self):
        return Scenario("Selection", 1, 10,
                        (Kitchen("A", 0, 0, 20), Kitchen("B", 4, 0, 20), Kitchen("C", 8, 0, 20)),
                        (School("S1", 0, 0, 5), School("S2", 4, 0, 5), School("S3", 8, 0, 5)),
                        max_active_kitchens=2)

    def test_milp_matches_exhaustive_site_selection(self):
        scenario = self.fixture()
        scores = [evaluate(scenario, a) for a in itertools.product(*scenario.eligible)]
        optimum = min(s.objective for s in scores if s.feasible)
        self.assertEqual(optimum, 20)
        oracle = solve_exact(scenario)
        self.assertEqual(oracle.status, "optimal")
        self.assertEqual(oracle.optimal_objective, optimum)
        self.assertLessEqual(len(set(oracle.assignment)), 2)
        impossible = replace(scenario, kitchens=tuple(replace(k, capacity=6) for k in scenario.kitchens))
        self.assertEqual(solve_exact(impossible).status, "infeasible")

    def test_site_repair_and_budget_diagnostics(self):
        scenario = self.fixture()
        raw = evaluate(scenario, (0, 1, 2))
        self.assertEqual(raw.overflow, 0)
        self.assertEqual(raw.active_site_violation, 1)
        fixed = evaluate(scenario, repair(scenario, raw.assignment))
        self.assertTrue(fixed.feasible)
        d = SearchDiagnostics()
        d.record(raw, fixed, True)
        self.assertEqual(d.snapshot(1)["repair_site_limit_attempts"], 1)
        self.assertEqual(d.snapshot(1)["repair_mean_objective_delta"], 20)
        for algorithm in ("GA", "PSO", "ACO"):
            run = solve(scenario, algorithm, settings=SolverSettings(oracle_time_limit=.15, budget=61))
            self.assertEqual(run.evaluations, 61)
            self.assertTrue(run.best.feasible)
            self.assertLessEqual(len(set(run.best.assignment)), 2)
            self.assertEqual(run.best, evaluate(scenario, run.best.assignment, run.best.visit_order or None))

    def test_fixed_schools_roads_and_demand_across_instances(self):
        for size in ("Small", "Medium", "Large"):
            a, b = generate_scenario(size, 101), generate_scenario(size, 102)
            self.assertEqual(a.schools, b.schools)
            self.assertEqual(a.road_nodes, b.road_nodes)
            self.assertEqual(a.road_edges, b.road_edges)
            self.assertNotEqual(a.kitchen_nodes, b.kitchen_nodes)
            self.assertEqual(len(a.kitchens), 2*a.max_active_kitchens)
            self.assertEqual(len({k.capacity for k in a.kitchens}), 1)
            self.assertEqual(Scenario.from_dict(a.to_dict()), a)

    def test_all_osm_schools_are_shared_across_sppg_configurations(self):
        from tests.support import network_fixture
        expected = {school[0] for school in network_fixture().schools}
        scenarios = [generate_scenario(size, 17) for size in ("Small", "Medium", "Large")]
        for s in scenarios:
            self.assertEqual({f["osm_id"] for f in s.road_source["school_features"]}, expected)
            self.assertEqual(len(s.schools), len(expected))
            self.assertEqual(s.schools, scenarios[0].schools)
            self.assertEqual(s.road_nodes, scenarios[0].road_nodes)
            self.assertEqual(s.road_edges, scenarios[0].road_edges)

    def test_selection_metrics_and_legacy_compatibility(self):
        scenario = self.fixture()
        score = evaluate(scenario, (0, 0, 2))
        metrics = delivery_metrics(scenario, score)
        self.assertEqual(metrics["selected_sppg_ids"], ["A", "C"])
        self.assertEqual(metrics["selected_capacity"], 40)
        old = scenario.to_dict()
        del old["max_active_kitchens"]
        self.assertTrue(evaluate(Scenario.from_dict(old), (0, 1, 2)).feasible)


if __name__ == "__main__":
    unittest.main()
