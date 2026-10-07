"""Optional desktop smoke tests: python -m unittest tests.test_gui -v."""

import time
import unittest
from tests.support import OsmFixtureTests
from unittest.mock import patch

from sppg_compare.algorithms import ALGORITHMS
from sppg_compare.gui import Application


class GuiTests(OsmFixtureTests):
    def setUp(self):
        super().setUp()
        self.app = Application()
        self.pump_until(lambda: not self.app.busy)
        # Existing tests exercise the preserved mapped-school tour model.
        self.app.scenario_mode_var.set("Mapped school tours")
        self.app.generate()
        self.pump_until(lambda: not self.app.busy)
        self.app.oracle_limit_var.set("0.15")
        self.app.withdraw()
        self.app.update()

    def tearDown(self):
        self.app.close()

    def pump_until(self, predicate, timeout=10):
        end = time.monotonic() + timeout
        while not predicate() and time.monotonic() < end:
            self.app.update()
            time.sleep(0.01)
        self.app.update()
        self.assertTrue(predicate(), "GUI condition timed out")

    def test_each_algorithm_demo_and_resize(self):
        self.app.budget_var.set("91")
        self.app.speed_var.set(60)
        for algorithm in ALGORITHMS:
            self.app.algorithm_var.set(algorithm)
            self.app.selected_button.invoke()
            self.pump_until(lambda: not self.app.busy)
            self.assertEqual(self.app.demo_runs[algorithm].evaluations, 91)
            self.assertEqual(self.app.current_snapshot.algorithm, algorithm)
        self.app.deiconify()
        self.app.geometry("1120x780")
        self.app.update()
        self.app.radius_visible.set(True)
        self.app.draw_map()
        self.assertGreater(len(self.app.map_canvas.find_all()), 0)

    def test_synthetic_planning_controls_raster_and_school_allocations(self):
        self.app.scenario_mode_var.set("Synthetic planning")
        self.app.school_count_var.set("20")
        self.app.sppg_capacity_var.set("1000")
        self.app.opening_weight_var.set("8")
        self.app.generate()
        self.pump_until(lambda: not self.app.busy)
        self.assertEqual(len(self.app.scenario.schools), 20)
        self.assertEqual(self.app.scenario.opening_weight, 8.)
        self.app.budget_var.set("61")
        self.app.speed_var.set(60)
        self.app.run_button.invoke()
        self.pump_until(lambda: not self.app.busy)
        for algorithm in ALGORITHMS:
            self.app.algorithm_var.set(algorithm)
            self.assertEqual(self.app.current_snapshot.evaluations, 61)
            self.assertEqual(len(self.app.map_canvas.find_withtag("map_background")), 1)
            self.assertEqual(len(self.app.map_canvas.find_withtag("school_marker")), 20)
            self.assertEqual(len(self.app.map_canvas.find_withtag("delivery_path")), 20)
            parents = self.app.tour_table.get_children()
            self.assertEqual(len(parents), len(set(self.app.current_snapshot.best.assignment)))
            self.assertEqual(sum(len(self.app.tour_table.get_children(p)) for p in parents), 20)
        self.app.school_count_var.set("25")
        self.app.generate()
        self.pump_until(lambda: not self.app.busy)
        self.assertEqual(len(self.app.scenario.schools), 25)

    def test_synchronized_three_method_run_switch_pause_and_route_search(self):
        self.app.budget_var.set("127")
        self.app.speed_var.set(2)
        self.app.run_button.invoke()
        self.pump_until(lambda: len(self.app.live_snapshots) == 3)
        self.app.pause()
        self.assertEqual(len({s.evaluations for s in self.app.live_snapshots.values()}), 1)
        self.app.algorithm_var.set("PSO")
        self.assertEqual(self.app.current_snapshot.algorithm, "PSO")
        count = self.app.current_snapshot.evaluations
        self.app.step()
        self.pump_until(lambda: self.app.live_snapshots["PSO"].evaluations > count)
        self.assertEqual(len({s.evaluations for s in self.app.live_snapshots.values()}), 1)
        self.app.speed_var.set(60)
        self.app._speed_changed()
        self.app.pause()
        self.pump_until(lambda: not self.app.busy)
        self.assertEqual(set(self.app.demo_runs), set(ALGORITHMS))
        self.assertTrue(all(r.evaluations == 127 for r in self.app.demo_runs.values()))
        for algorithm, tag in (("PSO", "route_key"), ("ACO", "route_pheromone")):
            self.app.algorithm_var.set(algorithm)
            self.app.search_view.matrix_mode.set("Visit order")
            self.app.search_view.draw()
            self.assertTrue(self.app.search_view.canvas.find_withtag(tag))
            active = len(set(self.app.current_snapshot.best.assignment))
            self.assertEqual(len(self.app.map_canvas.find_withtag("delivery_vehicle")), active)
            parents = self.app.tour_table.get_children()
            self.assertEqual(len(parents), active)
            self.assertEqual(sum(len(self.app.tour_table.get_children(parent)) for parent in parents), len(self.app.scenario.schools)+active)
        self.app._delivery_clock_minutes = max(self.app.current_snapshot.best.school_arrivals)
        self.app.draw_map()
        self.assertTrue(all(self.app.map_canvas.itemcget(marker, "outline") == "#15803d"
                            for marker, _ in self.app._school_markers))
        # At the minimum size, every control remains reachable by scrolling.
        self.app.sidebar_canvas.yview_moveto(1)
        self.app.update()
        export_bottom = self.app.export_button.winfo_rooty() + self.app.export_button.winfo_height()
        canvas_bottom = self.app.sidebar_canvas.winfo_rooty() + self.app.sidebar_canvas.winfo_height()
        self.assertLessEqual(export_bottom, canvas_bottom)
        self.assertGreaterEqual(self.app.export_button.winfo_rooty(), self.app.sidebar_canvas.winfo_rooty())

    def test_pause_step_reset_and_stale_events(self):
        self.app.budget_var.set("300")
        self.app.speed_var.set(2)
        self.app.run_demo()
        self.pump_until(lambda: self.app.current_snapshot is not None)
        self.app.pause()
        time.sleep(0.55)
        self.app.update()
        count = self.app.current_snapshot.evaluations
        self.app.step()
        self.pump_until(lambda: self.app.current_snapshot.evaluations > count)
        stepped = self.app.current_snapshot.evaluations
        end = time.monotonic() + 0.65
        while time.monotonic() < end:
            self.app.update()
            time.sleep(0.01)
        self.assertEqual(self.app.current_snapshot.evaluations, stepped)
        old_token = self.app.token
        old_snapshot = self.app.current_snapshot
        self.app.reset()
        self.app.events.put((old_token, "demo_progress", old_snapshot))
        self.pump_until(lambda: self.app.events.empty())
        self.assertIsNone(self.app.current_snapshot)
        self.assertFalse(self.app.busy)

    def test_benchmark_worker_and_replay(self):
        self.app.budget_var.set("61")
        self.app.instances_var.set("1")
        # Exercise completion/replay without depending on MILP proof difficulty.
        self.app.oracle_limit_var.set("1")
        self.app.run_comparison()
        self.pump_until(lambda: not self.app.busy, timeout=90)
        self.assertEqual(len(self.app.benchmark.runs), 90)
        self.assertTrue(self.app.benchmark.completed)
        self.assertEqual(len(self.app.summary_table.get_children()), 9)
        self.app.speed_var.set(2)
        self.app.start_replay()
        self.assertEqual(self.app.replay_index, 1)
        self.app.pause()
        self.assertFalse(self.app.replay_playing)
        self.app.step()
        self.assertEqual(self.app.replay_index, 2)
        self.app.reset()
        self.assertEqual(self.app.replay_frames, [])

    def test_settings_invalidation_and_validation_error(self):
        self.app.budget_var.set("31")
        self.app.speed_var.set(60)
        self.app.run_demo()
        self.pump_until(lambda: not self.app.busy)
        self.assertTrue(self.app.demo_runs)
        self.app.population_var.set("40")
        self.assertEqual(self.app.demo_runs, {})
        self.app.radius_var.set("0")
        with patch("sppg_compare.gui.messagebox.showerror") as error:
            self.app.generate()
            error.assert_called_once()

    def test_road_paths_arrival_metrics_and_delivery_animation(self):
        self.app.budget_var.set("31")
        self.app.speed_var.set(60)
        self.app.run_demo()
        self.pump_until(lambda: not self.app.busy)
        canvas = self.app.map_canvas
        scenario = self.app.display_scenario
        self.assertEqual(len(canvas.find_withtag("map_background")), 1)
        self.assertEqual(len(canvas.find_withtag("road")), 0)
        self.assertEqual(len(canvas.find_withtag("delivery_path")), len(scenario.schools)+len(set(self.app.current_snapshot.best.assignment)))
        self.assertEqual(len(canvas.find_withtag("delivery_vehicle")), len(set(self.app.current_snapshot.best.assignment)))
        self.assertNotEqual(self.app.metric_vars["Weighted ETA · min"].get(), "—")
        marker = canvas.find_withtag("delivery_vehicle")[0]
        before = canvas.coords(marker)
        self.app._delivery_last_tick -= 1
        # Redraw cancels the prior timer before starting the next animation.
        self.app.draw_map()
        marker = canvas.find_withtag("delivery_vehicle")[0]
        self.assertNotEqual(before, canvas.coords(marker))
        self.app.delivery_visible.set(False)
        self.app.draw_map()
        self.assertEqual(canvas.find_withtag("delivery_vehicle"), ())
        self.assertIsNone(self.app._delivery_after_id)
        self.app.reset()

    def test_multi_instance_replay_and_oracle_metrics(self):
        self.app.budget_var.set("31")
        self.app.instances_var.set("2")
        self.app.optimizer_count_var.set("2")
        self.app.run_comparison()
        self.pump_until(lambda: not self.app.busy, timeout=40)
        self.assertEqual(len(self.app.benchmark.runs), 36)
        self.assertEqual(len(self.app.replay_lookup), 36)
        self.assertEqual(len(self.app.benchmark.oracles), 6)
        self.app.start_replay()
        self.assertNotEqual(self.app.metric_vars["MILP optimum"].get(), "—")
        self.assertIn("scenario", self.app.replay_var.get())
        self.app.reset()
        self.app.difficulty_var.set("Tight")
        self.assertEqual(self.app.slack_var.get(), "0.0")

    def test_pso_and_aco_search_views_and_timer_cleanup(self):
        self.app.budget_var.set("61")
        self.app.speed_var.set(60)
        for algorithm in ("PSO", "ACO"):
            self.app.algorithm_var.set(algorithm)
            self.app.run_demo()
            self.pump_until(lambda: not self.app.busy)
            view = self.app.search_view
            view.cancel()
            view.phase = 1.
            view.draw()
            self.assertEqual(view.snapshot.search_state["kind"], algorithm)
            self.assertTrue(view.canvas.find_withtag("search_cell"))
            if algorithm == "PSO":
                self.assertEqual(len(view.canvas.find_withtag("particle")), 8)
                self.assertEqual(len(view.canvas.find_withtag("velocity")), 8)
                self.assertTrue(view.canvas.find_withtag("global_best"))
                self.assertTrue(view.canvas.find_withtag("personal_best"))
            else:
                self.assertTrue(view.canvas.find_withtag("ant_choice"))
                self.assertTrue(view.canvas.find_withtag("ant_cell"))
                view.phase = 0.
                view.draw()
                self.assertFalse(view.canvas.find_withtag("ant_choice"))
            view.school_var.set(self.app.display_scenario.schools[-1].id)
            view.draw()
            view.set_paused(True)
            self.assertIsNone(view.timer)
        self.app.reset()
        self.assertIsNone(self.app.search_view.snapshot)
        self.assertIsNone(self.app.search_view.timer)

    def test_osm_import_worker_labels_attribution_and_directed_delivery(self):
        from tests.test_osm import fixture
        from sppg_compare.osm import parse_network
        from sppg_compare.model import generate_scenario
        network = parse_network(fixture(), "Ciledug")
        with patch("sppg_compare.osm.load_network", return_value=network):
            scenario = generate_scenario(map_source="OpenStreetMap", osm_city="Ciledug")
        self.app.map_source_var.set("OpenStreetMap")
        self.app.osm_city_var.set("Ciledug")
        with patch("sppg_compare.gui.generate_scenario", return_value=scenario):
            self.app.generate()
            self.pump_until(lambda: not self.app.busy)
        self.assertEqual(self.app.scenario, scenario)
        self.assertEqual(len(self.app.map_canvas.find_withtag("school_marker")), len(scenario.schools))
        self.app.school_names_visible.set(True)
        self.app.draw_map()
        labels = [self.app.map_canvas.itemcget(item, "text") for item in self.app.map_canvas.find_withtag("school_label")]
        self.assertEqual(set(labels), {f["name"] for f in scenario.road_source["school_features"]})
        self.assertTrue(self.app.map_canvas.find_withtag("osm_attribution"))
        self.assertTrue(self.app.map_canvas.find_withtag("osm_street_label"))
        # Use a recorded optimizer frame to exercise directed route animation.
        from sppg_compare.algorithms import Optimizer, SolverSettings
        opt = Optimizer(scenario, "ACO", settings=SolverSettings(oracle_time_limit=.15, budget=31), capture_search=True)
        self.app.display_snapshot(opt.advance())
        self.assertEqual(len(self.app.map_canvas.find_withtag("delivery_vehicle")), len(set(self.app.current_snapshot.best.assignment)))
        active = len(set(self.app.current_snapshot.best.assignment))
        self.assertEqual(len(self.app.map_canvas.find_withtag("selected_sppg")), active)
        self.assertEqual(len(self.app.map_canvas.find_withtag("unused_sppg")), len(scenario.kitchens)-active)
        for (_, segments, eta), expected in zip(self.app._delivery_routes,
                [t for t in self.app.current_snapshot.best.route_durations if t]):
            self.assertAlmostEqual(eta, expected)
        self.app.reset()


if __name__ == "__main__":
    unittest.main()
