import itertools
import unittest
from dataclasses import replace

from sppg_compare.model import Kitchen, School, Scenario, evaluate, vehicle_tours, delivery_metrics
from sppg_compare.roads import RoadNode, RoadEdge
from sppg_compare.oracle import solve_exact, optimality_gap
from sppg_compare.algorithms import ALGORITHMS, Optimizer, SolverSettings, solve


def routing_fixture():
    nodes = tuple(RoadNode(str(i), i, 0) for i in range(5))
    return Scenario("Tours", 1, 10, (Kitchen("A", 0, 0, 10), Kitchen("B", 4, 0, 10)),
                    (School("S1", 1, 0, 4), School("S2", 2, 0, 3), School("S3", 3, 0, 2)),
                    road_nodes=nodes, road_edges=tuple(RoadEdge(str(i), str(i+1), 60) for i in range(4)),
                    kitchen_nodes=("0", "4"), school_nodes=("1", "2", "3"),
                    objective_metric="travel_time", max_active_kitchens=1, routing_mode="multi_stop")


class RoutingTests(unittest.TestCase):
    def test_arrivals_unloading_capacity_and_return(self):
        s = routing_fixture()
        score = evaluate(s, (0, 0, 0), (0, 1, 2))
        self.assertEqual(score.school_arrivals, (1., 7., 13.))
        self.assertEqual(score.objective, 51.)
        self.assertEqual(score.route_durations, (21., 0.))
        tour = vehicle_tours(s, score)[0]
        self.assertEqual(tour["initial_load"], 9)
        self.assertEqual(tour["legs"][-1]["to"], "A")
        self.assertEqual(tour["legs"][-1]["arrival_minutes"], 21)
        self.assertEqual(tour["legs"][1]["departure_minutes"], 6)
        self.assertEqual(delivery_metrics(s, score)["weighted_average_minutes"], 51/9)
        self.assertFalse(evaluate(replace(s, kitchens=tuple(replace(k, capacity=8) for k in s.kitchens)), (0, 0, 0)).feasible)
        with self.assertRaises(ValueError):
            evaluate(s, (0, 0, 0), (0, 0, 1))

    def test_routing_milp_matches_exhaustive_joint_search(self):
        s = routing_fixture()
        exact = min(score.objective for a in itertools.product(*s.eligible)
                    for order in itertools.permutations(range(3))
                    if (score := evaluate(s, a, order)).feasible)
        oracle = solve_exact(s, 10)
        self.assertEqual(oracle.status, "optimal")
        self.assertEqual(oracle.optimal_objective, exact)
        self.assertEqual(exact, 51)
        score = evaluate(s, oracle.assignment, oracle.visit_order)
        self.assertEqual(optimality_gap(score, oracle), (0, 0))

    def test_zero_time_subtours_are_excluded(self):
        s = routing_fixture()
        s = replace(s, schools=tuple(replace(school, x=0) for school in s.schools),
                    school_nodes=("0", "0", "0"), unloading_minutes=0.)
        oracle = solve_exact(s, 10)
        self.assertEqual(oracle.status, "optimal")
        self.assertEqual(oracle.optimal_objective, 0)
        self.assertEqual(set(oracle.visit_order), {0, 1, 2})

    def test_visit_order_is_optimized_by_all_methods_with_exact_budgets(self):
        s = routing_fixture()
        for a in ALGORITHMS:
            settings = SolverSettings(budget=127, oracle_time_limit=10)
            plain, captured = solve(s, a, 9, settings), solve(s, a, 9, settings, capture_search=True)
            self.assertEqual(plain.evaluations, 127)
            self.assertEqual(plain.best, captured.best)
            self.assertEqual(plain.history, captured.history)
            self.assertTrue(plain.best.feasible)
            self.assertEqual(set(plain.best.visit_order), {0, 1, 2})
            self.assertEqual(plain.best, evaluate(s, plain.best.assignment, plain.best.visit_order))
            self.assertGreaterEqual(plain.best.objective, 51)

    def test_synchronized_evaluation_boundaries(self):
        s = routing_fixture()
        optimizers = [Optimizer(s, a, settings=SolverSettings(budget=67, oracle_time_limit=10)) for a in ALGORITHMS]
        for boundary in (30, 60, 67):
            for optimizer in optimizers:
                while optimizer.evaluations < boundary:
                    optimizer.advance_to(boundary)
                self.assertEqual(optimizer.evaluations, boundary)
