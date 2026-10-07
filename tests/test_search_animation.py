"""Search recording must not alter the optimizer's experimental behavior."""
import json
import unittest
from tests.support import OsmFixtureTests

from sppg_compare.algorithms import Optimizer, SolverSettings
from sppg_compare.model import generate_scenario


class SearchCaptureTests(OsmFixtureTests):
    def test_capture_preserves_search_rng_budget_and_diagnostics(self):
        scenario = generate_scenario()
        settings = SolverSettings(oracle_time_limit=.15, budget=67)
        for algorithm in ("PSO", "ACO"):
            plain = Optimizer(scenario, algorithm, 9, settings)
            captured = Optimizer(scenario, algorithm, 9, settings, capture_search=True)
            while not plain.done:
                a, b = plain.advance(), captured.advance()
                self.assertEqual(a.best, b.best)
                self.assertEqual(a.diagnostics, b.diagnostics)
                self.assertEqual(a.evaluations, b.evaluations)
                self.assertEqual(plain.rng.getstate(), captured.rng.getstate())
                self.assertFalse(a.search_state)
                self.assertEqual(b.search_state["kind"], algorithm)
            self.assertEqual(plain.history, captured.history)
            self.assertEqual(captured.evaluations, 67)

    def test_ant_trace_is_complete_eligible_and_immutable(self):
        scenario = generate_scenario()
        opt = Optimizer(scenario, "ACO", settings=SolverSettings(oracle_time_limit=.15, budget=61), capture_search=True)
        first = opt.advance()
        signature = json.dumps(first.search_state)
        opt.advance()
        self.assertEqual(json.dumps(first.search_state), signature)
        state = first.search_state
        self.assertEqual(tuple(step["school"] for step in state["trace"]), scenario.school_order)
        for step in state["trace"]:
            i, j = step["school"], step["kitchen"]
            self.assertIn(j, scenario.eligible[i])
            self.assertEqual(j, state["raw_assignment"][i])
            self.assertAlmostEqual(sum(step["probabilities"]), 1.)
            self.assertEqual(len(step["choices"]), len(step["probabilities"]))

    def test_swarm_projection_matches_actual_preferences(self):
        scenario = generate_scenario()
        opt = Optimizer(scenario, "PSO", settings=SolverSettings(oracle_time_limit=.15, budget=31), capture_search=True)
        frame = opt.advance()
        pairs = [(i, choices) for i, choices in enumerate(scenario.eligible) if len(choices)>1]
        p = frame.search_state["particles"][0]
        for axis in (0, 1):
            expected = sum(opt.positions[0][i][choices[axis]] for i, choices in pairs) / len(pairs)
            self.assertAlmostEqual(p["position"][axis], expected)
        signature = json.dumps(frame.search_state)
        opt.advance()
        self.assertEqual(json.dumps(frame.search_state), signature)
        self.assertEqual(len(frame.search_state["particles"]), 8)
        self.assertEqual(frame.search_state["population"], 30)
