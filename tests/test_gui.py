"""Optional desktop smoke tests: python -m unittest tests.test_gui -v."""

import time
import unittest
from unittest.mock import patch

from sppg_compare.algorithms import ALGORITHMS
from sppg_compare.gui import Application


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.app = Application()
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
            self.app.run_button.invoke()
            self.pump_until(lambda: not self.app.busy)
            self.assertEqual(self.app.demo_runs[algorithm].evaluations, 91)
            self.assertEqual(self.app.current_snapshot.algorithm, algorithm)
        self.app.deiconify()
        self.app.geometry("1120x780")
        self.app.update()
        self.app.radius_visible.set(True)
        self.app.draw_map()
        self.assertGreater(len(self.app.map_canvas.find_all()), 0)
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
        self.app.run_comparison()
        self.pump_until(lambda: not self.app.busy, timeout=30)
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


if __name__ == "__main__":
    unittest.main()
