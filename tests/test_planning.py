import itertools
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from tests.support import OsmFixtureTests
from sppg_compare.algorithms import ALGORITHMS, SolverSettings, solve
from sppg_compare.experiments import export_benchmark, run_benchmark
from sppg_compare.model import Kitchen, School, Scenario, evaluate, generate_scenario, delivery_metrics
from sppg_compare.oracle import solve_exact
from sppg_compare.roads import RoadNode, RoadEdge, route_tables


class PlanningTests(OsmFixtureTests):
    def tiny(self, opening=10., distance=1.):
        nodes = tuple(RoadNode(str(i), i, 0.) for i in range(5))
        edges = tuple(RoadEdge(str(i), str(i+1), 25.) for i in range(4))
        return Scenario("Planning", 1, 10., (Kitchen("K1", 0., 0., 400), Kitchen("K2", 4., 0., 400)),
                        (School("S1", 1., 0., 200), School("S2", 3., 0., 200)),
                        road_nodes=nodes, road_edges=edges, kitchen_nodes=("0", "4"), school_nodes=("1", "3"),
                        objective_metric="facility_distance", opening_weight=opening, distance_weight=distance)

    def test_weighted_milp_matches_exhaustive_and_weights_change_site_count(self):
        for opening, distance, expected_count in ((10., 1., 1), (0., 1., 2), (1., 10., 2), (10., 0., 1)):
            scenario = self.tiny(opening, distance)
            feasible = [evaluate(scenario, a) for a in itertools.product(*scenario.eligible)]
            oracle = solve_exact(scenario)
            self.assertEqual(oracle.status, "optimal")
            self.assertAlmostEqual(oracle.optimal_objective, min(s.objective for s in feasible if s.feasible))
            self.assertEqual(len(set(oracle.assignment)), expected_count)
            score = evaluate(scenario, oracle.assignment)
            metrics = delivery_metrics(scenario, score)
            self.assertEqual(sum(item["school_count"] for item in metrics["sppg_load_utilization"]), 2)
            self.assertAlmostEqual(score.objective, opening*expected_count+distance*metrics["total_allocation_distance_km"])
        capacity_limited = replace(self.tiny(), kitchens=(Kitchen("K1", 0., 0., 200), Kitchen("K2", 4., 0., 200)))
        self.assertEqual(len(set(solve_exact(capacity_limited).assignment)), 2)
        impossible = replace(capacity_limited, kitchens=tuple(replace(k, capacity=100) for k in capacity_limited.kitchens))
        self.assertEqual(solve_exact(impossible).status, "infeasible")

    def test_distance_paths_are_shortest_distance_not_fastest(self):
        nodes = (RoadNode("a", 0., 0.), RoadNode("b", 1., 0.), RoadNode("c", .5, 1.))
        edges = (RoadEdge("a", "b", 5., True), RoadEdge("a", "c", 50., True), RoadEdge("c", "b", 50., True))
        fastest = route_tables(nodes, edges, ("a",), ("b",))[0][0]
        shortest = route_tables(nodes, edges, ("a",), ("b",), metric="distance")[0][0]
        self.assertEqual(shortest[2], ("a", "b"))
        self.assertEqual(fastest[2], ("a", "c", "b"))
        self.assertLess(shortest[1], fastest[1])
        self.assertGreater(shortest[0], fastest[0])
        with self.assertRaises(ValueError):
            route_tables(nodes, edges, ("b",), ("a",), metric="distance")

    def test_fixed_nested_schools_and_independent_candidate_seeds(self):
        a = generate_scenario(seed=1, school_count=20)
        b = generate_scenario(seed=2, school_count=20)
        larger = generate_scenario(size="Large", seed=1, school_count=30)
        self.assertEqual(a.schools, b.schools)
        self.assertEqual(a.schools, larger.schools[:20])
        self.assertNotEqual(a.kitchens, b.kitchens)
        self.assertEqual({k.capacity for k in a.kitchens}, {3000})
        self.assertIsNone(a.max_active_kitchens)
        self.assertEqual(a.objective_metric, "facility_distance")
        self.assertEqual(Scenario.from_dict(a.to_dict()), a)
        with self.assertRaises(ValueError):
            generate_scenario(school_count=0)
        with self.assertRaises(ValueError):
            generate_scenario(school_count=30, opening_weight=0., distance_weight=0.)

    def test_equal_budgets_oracle_gaps_and_planning_exports(self):
        result = run_benchmark(SolverSettings(budget=61, oracle_time_limit=2.), sizes=("Small",),
                               scenario_seeds=(1, 2), optimizer_seeds=(11,), school_count=12,
                               school_seed=7, sppg_capacity=1000, opening_weight=8., distance_weight=2.)
        self.assertEqual(len(result.runs), 6)
        for run in result.runs:
            self.assertEqual(run.evaluations, 61)
            scenario = next(s for s in result.scenarios if s.instance_id == run.scenario_id)
            self.assertEqual(run.best, evaluate(scenario, run.best.assignment))
            if run.best.feasible and run.oracle.status == "optimal":
                self.assertIsNotNone(run.optimality_gap_percent)
        with tempfile.TemporaryDirectory() as directory:
            export_benchmark(result, directory)
            payload = json.loads((Path(directory)/"benchmark.json").read_text())
            self.assertEqual(payload["scenarios"][0]["opening_weight"], 8.)
            self.assertIn("total_allocation_distance_km", payload["runs"][0])
            self.assertIn("school_count", payload["runs"][0]["sppg_load_utilization"][0])
            self.assertTrue((Path(directory)/"diagnostics.csv").exists())
