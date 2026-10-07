import csv
from dataclasses import replace
import itertools
import json
import tempfile
import unittest
from tests.support import OsmFixtureTests
from pathlib import Path

from sppg_compare.algorithms import ALGORITHMS, SolverSettings, solve
from sppg_compare.experiments import export_demo
from sppg_compare.model import Kitchen, School, Scenario, delivery_metrics, evaluate, generate_scenario
from sppg_compare.oracle import solve_exact
from sppg_compare.roads import RoadNode, RoadEdge


def road_scenario():
    # A longer 60 kph route beats the direct 5 kph road.
    return Scenario("Road test", 4, 10,
        (Kitchen("K1", 0, 0, 100), Kitchen("K2", 4, 0, 100)),
        (School("S1", 1, 0, 60), School("S2", 3, 0, 60)),
        road_nodes=(RoadNode("a", 0, 0), RoadNode("b", 4, 0),
                    RoadNode("c", 1, 0), RoadNode("d", 3, 0), RoadNode("e", 0, 1)),
        road_edges=(RoadEdge("a", "c", 5), RoadEdge("a", "e", 60),
                    RoadEdge("e", "c", 60), RoadEdge("c", "d", 30), RoadEdge("d", "b", 30)),
        kitchen_nodes=("a", "b"), school_nodes=("c", "d"), objective_metric="travel_time")


class RoadTests(OsmFixtureTests):
    def test_only_ciledug_osm_is_generated(self):
        s = generate_scenario()
        self.assertEqual(s.road_source["source"], "OpenStreetMap")
        self.assertEqual(s.road_source["city"], "Ciledug")
        self.assertTrue(any("ciledug raya" in e.name.lower() for e in s.road_edges))
        self.assertEqual(s.routing_mode, "multi_stop")
        for kwargs in ({"map_source": "Synthetic"}, {"osm_city": "Yogyakarta"}):
            with self.assertRaises(ValueError):
                generate_scenario(**kwargs)

    def test_crossings_do_not_create_osm_junctions(self):
        # Two paths cross on screen but have no shared OSM node (e.g. bridge).
        from sppg_compare.roads import route_tables
        nodes = (RoadNode("w", -1, 0), RoadNode("e", 1, 0),
                 RoadNode("s", 0, -1), RoadNode("n", 0, 1))
        with self.assertRaises(ValueError):
            route_tables(nodes, (RoadEdge("w", "e", 25), RoadEdge("s", "n", 25)), ("w",), ("n",))

    def test_fastest_path_not_shortest_distance(self):
        s = road_scenario()
        self.assertEqual(s.paths[0][0], ("a", "e", "c"))
        self.assertAlmostEqual(s.travel_times[0][0], 1 + 2 ** .5)
        self.assertGreater(s.road_distances[0][0], s.distances[0][0])
        self.assertAlmostEqual(evaluate(s, (0, 1)).objective, 60 * (3 + 2 ** .5))

    def test_shared_time_objective_oracle_and_algorithms(self):
        s = road_scenario()
        exact = min(evaluate(s, a).objective for a in itertools.product(*s.eligible) if evaluate(s, a).feasible)
        oracle = solve_exact(s)
        self.assertEqual(oracle.status, "optimal")
        self.assertAlmostEqual(oracle.optimal_objective, exact)
        for algorithm in ALGORITHMS:
            run = solve(s, algorithm, 2, SolverSettings(budget=91))
            self.assertEqual(run.evaluations, 91)
            self.assertAlmostEqual(run.best.objective, exact)
            self.assertAlmostEqual(run.optimality_gap_percent, 0)

    def test_speed_changes_time_without_changing_geography_or_loads(self):
        s = road_scenario()
        faster = replace(s, road_edges=tuple(replace(e, speed_kph=e.speed_kph * 2) for e in s.road_edges))
        a, b = evaluate(s, (0, 1)), evaluate(faster, (0, 1))
        self.assertEqual(a.loads, b.loads)
        self.assertEqual(s.distances, faster.distances)
        self.assertAlmostEqual(a.objective, 2 * b.objective)
        self.assertNotEqual(s.instance_id, faster.instance_id)

    def test_reproducibility_radius_and_slack_independence(self):
        a = generate_scenario(radius=1.3)
        for b in (generate_scenario(radius=8), generate_scenario(difficulty="Easy")):
            self.assertEqual(a.road_nodes, b.road_nodes)
            self.assertEqual(a.road_edges, b.road_edges)
            self.assertEqual(a.travel_times, b.travel_times)
        self.assertNotEqual(a.eligible, generate_scenario(radius=8).eligible)
        self.assertEqual(Scenario.from_dict(a.to_dict()), a)
        self.assertEqual(a.road_edges, generate_scenario(seed=102).road_edges)

    def test_invalid_networks_fail_explicitly(self):
        s = road_scenario()
        for changes in ({"road_edges": ()}, {"road_nodes": ()},
                        {"road_edges": (RoadEdge("a", "c", 0),)},
                        {"kitchen_nodes": ("missing", "b")},
                        {"kitchen_nodes": ("c", "b")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(s, **changes)

    def test_exports_report_distance_and_time_with_paths(self):
        s = road_scenario()
        run = solve(s, "GA", settings=SolverSettings(budget=31))
        with tempfile.TemporaryDirectory() as directory:
            export_demo(s, [run], directory)
            payload = json.loads((Path(directory) / "benchmark.json").read_text(encoding="utf-8"))
            item = payload["runs"][0]
            self.assertEqual(item["objective_unit"], "portion-min")
            self.assertAlmostEqual(item["weighted_average_minutes"], run.best.objective / s.total_demand)
            self.assertTrue(item["deliveries"][0]["path"])
            restored = Scenario.from_dict(payload["scenarios"][0])
            self.assertEqual(restored, s)
            with (Path(directory) / "runs.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["weighted_average_km"]), delivery_metrics(s, run.best)["weighted_average_km"])
            self.assertAlmostEqual(float(row["weighted_average_minutes"]), item["weighted_average_minutes"])
