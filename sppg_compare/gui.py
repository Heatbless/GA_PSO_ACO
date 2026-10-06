"""Tkinter dashboard. Workers only touch Python objects, never Tk widgets."""

import math
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .algorithms import ALGORITHMS, Optimizer, SolverSettings
from .experiments import export_benchmark, export_demo, run_benchmark
from .model import SIZES, generate_scenario, greedy_baseline


COLORS = ("#2878b5", "#d96c22", "#2b9973", "#a05daf", "#ce5265", "#a58b20", "#4678a6", "#7873ba")
INK, MUTED, BACKGROUND = "#19304a", "#61758b", "#eef3f8"


class WorkerControl:
    def __init__(self, delay=0.1):
        self.cancelled = threading.Event()
        self.condition = threading.Condition()
        self.paused = False
        self.steps = 0
        self.delay = delay

    def wait_turn(self):
        with self.condition:
            while self.paused and not self.steps and not self.cancelled.is_set():
                self.condition.wait(0.1)
            if self.cancelled.is_set():
                return False
            if self.paused:
                self.steps -= 1
            return True

    def cancel(self):
        self.cancelled.set()
        with self.condition:
            self.condition.notify_all()


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SPPG Lab | GA · PSO · ACO")
        self.geometry("1380x920")
        self.minsize(1120, 780)
        self.configure(bg=BACKGROUND)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.events = queue.Queue()
        self.token = 0
        self.control = None
        self.worker = None
        self.busy = False
        self.benchmark = None
        self.demo_runs = {}
        self.display_scenario = None
        self.current_snapshot = None
        self.plot_history = []
        self.replay_frames = []
        self.replay_index = 0
        self.replay_playing = False
        self.replay_token = 0
        self._replay_after_id = None
        self.replay_lookup = {}
        self.closed = False
        self._initializing = True
        self.size_var = tk.StringVar(value="Small")
        self.dataset_seed_var = tk.StringVar(value="101")
        self.radius_var = tk.StringVar(value="6")
        self.algorithm_var = tk.StringVar(value="GA")
        self.optimizer_seed_var = tk.StringVar(value="1")
        self.budget_var = tk.StringVar(value="5000")
        self.population_var = tk.StringVar(value="30")
        self.speed_var = tk.DoubleVar(value=10)
        self.status_var = tk.StringVar(value="Generate a scenario, then run an algorithm or the full comparison.")
        self.metric_vars = {name: tk.StringVar(value="—") for name in
                            ("Feasibility", "Objective · portion-km", "Weighted distance · km", "Evaluations", "Solver time")}
        defaults = SolverSettings()
        self.parameter_vars = {name: tk.StringVar(value="auto" if name == "mutation" else str(getattr(defaults, name)))
                               for name in ("crossover", "mutation", "elites", "tournament", "inertia",
                                            "cognitive", "social", "velocity_limit", "alpha", "beta", "evaporation")}
        self.input_widgets = []
        self.parameter_widgets = []
        self._style()
        self._build()
        for variable in (self.dataset_seed_var, self.radius_var, self.optimizer_seed_var,
                         self.budget_var, self.population_var, *self.parameter_vars.values()):
            variable.trace_add("write", self._invalidate)
        self.algorithm_var.trace_add("write", self._algorithm_changed)
        self.size_var.trace_add("write", self._size_changed)
        self._initializing = False
        self.generate()
        self._poll_id = self.after(40, self._poll)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background="white")
        style.configure("TLabel", background=BACKGROUND, foreground=INK, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="white", foreground=INK)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Metric.TLabel", background="white", font=("Segoe UI", 15, "bold"))
        style.configure("TButton", font=("Segoe UI", 10), padding=(9, 6))
        style.configure("Primary.TButton", background="#2878b5", foreground="white")
        style.map("Primary.TButton", background=[("active", "#1c6094"), ("disabled", "#aabacb")])
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=30)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("TNotebook.Tab", padding=(18, 9))

    def _field(self, parent, label, variable, row, choices=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        if choices:
            widget = ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly", width=12)
        else:
            widget = ttk.Entry(parent, textvariable=variable, width=14)
        widget.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.input_widgets.append(widget)
        return widget

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self, padding=(22, 14))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="SPPG assignment lab", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Three search methods. One shared assignment model. Reproducible comparisons.",
                  style="Muted.TLabel").pack(anchor="w", pady=(3, 0))
        body = ttk.Frame(self, padding=(18, 0, 18, 10))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar_host = ttk.Frame(body, padding=(0, 0, 12, 0))
        sidebar_host.grid(row=0, column=0, sticky="ns")
        sidebar_host.rowconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar_host, width=242, bg=BACKGROUND,
                                        highlightthickness=0, yscrollincrement=24)
        self.sidebar_canvas.grid(row=0, column=0, sticky="ns")
        sidebar_bar = ttk.Scrollbar(sidebar_host, orient="vertical", command=self.sidebar_canvas.yview)
        sidebar_bar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_bar.set)
        sidebar = ttk.Frame(self.sidebar_canvas, padding=(0, 0, 8, 0))
        sidebar_window = self.sidebar_canvas.create_window(0, 0, window=sidebar, anchor="nw", width=242)
        sidebar.bind("<Configure>", lambda event: self.sidebar_canvas.configure(
            scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda event: self.sidebar_canvas.itemconfigure(
            sidebar_window, width=event.width))
        self.bind("<MouseWheel>", self._sidebar_wheel)
        sidebar.columnconfigure(1, weight=1)
        ttk.Label(sidebar, text="Scenario", style="Heading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(5, 6))
        self._field(sidebar, "Size", self.size_var, 1, tuple(SIZES))
        self._field(sidebar, "Dataset seed", self.dataset_seed_var, 2)
        self._field(sidebar, "Radius · km", self.radius_var, 3)
        self.generate_button = ttk.Button(sidebar, text="Generate scenario", command=self.generate)
        self.generate_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 14))
        ttk.Label(sidebar, text="Search", style="Heading.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._field(sidebar, "Algorithm", self.algorithm_var, 6, ALGORITHMS)
        self._field(sidebar, "Optimizer seed", self.optimizer_seed_var, 7)
        self._field(sidebar, "Eval. budget", self.budget_var, 8)
        self._field(sidebar, "Population / ants", self.population_var, 9)
        self.parameters_frame = ttk.Frame(sidebar)
        self.parameters_frame.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(5, 8))
        self.parameters_frame.columnconfigure(1, weight=1)
        self._build_parameters()
        controls = ttk.Frame(sidebar)
        controls.grid(row=11, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure((0, 1), weight=1)
        self.run_button = ttk.Button(controls, text="Run selected", command=self.run_demo, style="Primary.TButton")
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=3)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause, state="disabled")
        self.pause_button.grid(row=1, column=0, sticky="ew", pady=3, padx=(0, 3))
        self.step_button = ttk.Button(controls, text="Step", command=self.step, state="disabled")
        self.step_button.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(controls, text="Reset / cancel", command=self.reset).grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(sidebar, text="Playback speed · updates / sec", style="Muted.TLabel").grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Scale(sidebar, from_=2, to=60, variable=self.speed_var, command=self._speed_changed).grid(row=13, column=0, columnspan=2, sticky="ew")
        self.compare_button = ttk.Button(sidebar, text="Compare all · 90 runs", command=self.run_comparison)
        self.compare_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(12, 3))
        self.export_button = ttk.Button(sidebar, text="Export results…", command=self.export, state="disabled")
        self.export_button.grid(row=15, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(sidebar, text="Comparison uses 3 fixed datasets,\nseeds 1–10 and an equal evaluation\nbudget. Animation delays are excluded.",
                  style="Muted.TLabel", justify="left").grid(row=16, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.notebook = ttk.Notebook(body)
        self.notebook.grid(row=0, column=1, sticky="nsew")
        self.animation_tab = ttk.Frame(self.notebook, padding=12)
        self.comparison_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.animation_tab, text="Animation")
        self.notebook.add(self.comparison_tab, text="Comparison")
        self._build_animation()
        self._build_comparison()
        ttk.Label(self, textvariable=self.status_var, style="Muted.TLabel", padding=(20, 8),
                  wraplength=1300).grid(row=2, column=0, sticky="ew")

    def _build_parameters(self):
        for widget in self.parameter_widgets:
            if widget in self.input_widgets:
                self.input_widgets.remove(widget)
        self.parameter_widgets.clear()
        for widget in self.parameters_frame.winfo_children():
            widget.destroy()
        fields = {"GA": (("Crossover", "crossover"), ("Mutation / auto", "mutation"),
                         ("Elite count", "elites"), ("Tournament size", "tournament")),
                  "PSO": (("Inertia", "inertia"), ("Cognitive", "cognitive"),
                          ("Social", "social"), ("Velocity limit", "velocity_limit")),
                  "ACO": (("Pheromone alpha", "alpha"), ("Distance beta", "beta"),
                          ("Evaporation", "evaporation"))}
        for row, (label, name) in enumerate(fields[self.algorithm_var.get()]):
            self.parameter_widgets.append(self._field(self.parameters_frame, label, self.parameter_vars[name], row))

    def _sidebar_wheel(self, event):
        canvas = self.sidebar_canvas
        if (canvas.winfo_rootx() <= event.x_root <= canvas.winfo_rootx() + canvas.winfo_width()
                and canvas.winfo_rooty() <= event.y_root <= canvas.winfo_rooty() + canvas.winfo_height()):
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

    def _build_animation(self):
        frame = self.animation_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=3)
        frame.rowconfigure(3, weight=2)
        self.map_title = ttk.Label(frame, text="School-to-kitchen assignment", style="Heading.TLabel")
        self.map_title.grid(row=0, column=0, sticky="w", pady=(0, 9))
        metrics = ttk.Frame(frame)
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column, (label, variable) in enumerate(self.metric_vars.items()):
            metrics.columnconfigure(column, weight=1)
            card = ttk.Frame(metrics, style="Card.TFrame", padding=9)
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 6))
            ttk.Label(card, text=label, style="Card.TLabel", font=("Segoe UI", 9)).pack(anchor="w")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(4, 0))
        map_frame = ttk.Frame(frame, style="Card.TFrame")
        map_frame.grid(row=2, column=0, sticky="nsew")
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(map_frame, bg="white", highlightthickness=0, height=340)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.radius_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(map_frame, text="Show service radii", variable=self.radius_visible,
                        command=self.draw_map).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.map_canvas.bind("<Configure>", lambda event: self.draw_map())
        bottom = ttk.Frame(frame)
        bottom.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)
        bottom.rowconfigure(1, weight=1)
        ttk.Label(bottom, text="Best feasible objective", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(bottom, text="Kitchen utilization", style="Heading.TLabel").grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(0, 5))
        self.curve_canvas = tk.Canvas(bottom, bg="white", highlightthickness=0, height=175)
        self.curve_canvas.grid(row=1, column=0, sticky="nsew")
        self.load_canvas = tk.Canvas(bottom, bg="white", highlightthickness=0, height=175)
        self.load_canvas.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self.curve_canvas.bind("<Configure>", lambda event: self.draw_curve())
        self.load_canvas.bind("<Configure>", lambda event: self.draw_loads())

    def _build_comparison(self):
        frame = self.comparison_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=1)
        ttk.Label(frame, text="Repeatable benchmark", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Feasible objectives only · sample standard deviation · solver time excludes playback",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 10))
        self.benchmark_progress = ttk.Progressbar(frame, maximum=90)
        self.benchmark_progress.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        columns = ("scenario", "algorithm", "feasible", "best", "mean", "std", "time", "baseline")
        self.summary_table = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for key, title in zip(columns, ("Scenario", "Method", "Feasible", "Best J", "Mean J", "Std. dev.", "Mean sec.", "Greedy J")):
            self.summary_table.heading(key, text=title)
            self.summary_table.column(key, width=98, minwidth=70, anchor="center", stretch=True)
        self.summary_table.grid(row=3, column=0, sticky="nsew")
        plots = ttk.Frame(frame)
        plots.grid(row=4, column=0, sticky="ew", pady=(14, 8))
        ttk.Label(plots, text="Mean feasible convergence", style="Heading.TLabel").pack(side="left")
        self.compare_size_var = tk.StringVar(value="Small")
        self.compare_size_box = ttk.Combobox(plots, textvariable=self.compare_size_var, values=tuple(SIZES), state="readonly", width=12)
        self.compare_size_box.pack(side="right")
        self.compare_size_box.bind("<<ComboboxSelected>>", lambda event: self.draw_comparison_curve())
        self.compare_curve = tk.Canvas(frame, bg="white", highlightthickness=0, height=210)
        self.compare_curve.grid(row=5, column=0, sticky="ew")
        self.compare_curve.bind("<Configure>", lambda event: self.draw_comparison_curve())
        ttk.Label(frame, text="Curves average only runs feasible at each checkpoint; contributing counts may change.",
                  style="Muted.TLabel", wraplength=900).grid(row=6, column=0, sticky="w", pady=(6, 10))
        replay = ttk.Frame(frame)
        replay.grid(row=7, column=0, sticky="ew")
        replay.columnconfigure(0, weight=1)
        self.replay_var = tk.StringVar()
        self.replay_box = ttk.Combobox(replay, textvariable=self.replay_var, state="readonly")
        self.replay_box.grid(row=0, column=0, sticky="ew")
        self.replay_button = ttk.Button(replay, text="Replay selected run", command=self.start_replay, state="disabled")
        self.replay_button.grid(row=0, column=1, padx=(10, 0))

    def _size_changed(self, *_):
        if self._initializing:
            return
        self.dataset_seed_var.set(str(SIZES[self.size_var.get()][2]))

    def _algorithm_changed(self, *_):
        if self._initializing:
            return
        self._build_parameters()
        self._stop_replay()
        run = self.demo_runs.get(self.algorithm_var.get())
        self.display_scenario = self.scenario
        self.plot_history = []
        self.current_snapshot = None
        if run:
            self.plot_history = [(h["evaluations"], h["objective"]) for h in run.history]
            self.display_snapshot(run.snapshots[-1])
            self.status_var.set(f"Showing saved {run.algorithm} demo, seed {run.seed}.")
        else:
            self.clear_display()

    def _invalidate(self, *_):
        if self._initializing or self.busy:
            return
        self._stop_replay()
        self.demo_runs.clear()
        self.benchmark = None
        self.current_snapshot = None
        self.plot_history = []
        self._clear_comparison()
        self.export_button.configure(state="disabled")
        self.display_scenario = getattr(self, "scenario", None)
        self.clear_display()
        self.status_var.set("Settings changed. Generate or run to use the new settings; previous results cleared.")

    def _clear_comparison(self):
        for row in self.summary_table.get_children():
            self.summary_table.delete(row)
        self.replay_lookup.clear()
        self.replay_box.configure(values=())
        self.replay_var.set("")
        self.replay_button.configure(state="disabled")
        self.benchmark_progress["value"] = 0
        self.draw_comparison_curve()

    def read_settings(self):
        values = {}
        for name, variable in self.parameter_vars.items():
            value = variable.get().strip()
            if name == "mutation" and value.lower() in ("auto", ""):
                values[name] = None
            elif name in ("elites", "tournament"):
                values[name] = int(value)
            else:
                values[name] = float(value)
        settings = SolverSettings(budget=int(self.budget_var.get()), population=int(self.population_var.get()), **values)
        settings.validate()
        return settings

    def _scenario_from_controls(self):
        return generate_scenario(self.size_var.get(), int(self.dataset_seed_var.get()), float(self.radius_var.get()))

    def generate(self):
        if self.busy:
            return
        try:
            scenario = self._scenario_from_controls()
        except ValueError as error:
            messagebox.showerror("Invalid scenario", str(error), parent=self)
            return
        self._invalidate()
        self.scenario = self.display_scenario = scenario
        self.clear_display()
        baseline = greedy_baseline(scenario)
        baseline_text = f"greedy J {baseline.objective:,.1f}" if baseline.feasible else f"greedy overflow {baseline.overflow}"
        self.status_var.set(f"{scenario.name}: {len(scenario.kitchens)} kitchens, {len(scenario.schools)} schools, "
                            f"{scenario.total_demand:,} portions/day; {baseline_text}. Select an algorithm to begin.")

    def _set_busy(self, busy, demo=False):
        self.busy = busy
        for widget in self.input_widgets:
            widget.configure(state="disabled" if busy else ("readonly" if isinstance(widget, ttk.Combobox) else "normal"))
        for button in (self.generate_button, self.run_button, self.compare_button):
            button.configure(state="disabled" if busy else "normal")
        self.pause_button.configure(state="normal" if busy and demo else "disabled", text="Pause")
        self.step_button.configure(state="normal" if busy and demo else "disabled")
        self.replay_button.configure(state="normal" if not busy and self.replay_lookup else "disabled")
        self.export_button.configure(state="normal" if not busy and (self.benchmark or self.demo_runs) else "disabled")

    def run_demo(self):
        if self.busy:
            return
        try:
            settings, seed = self.read_settings(), int(self.optimizer_seed_var.get())
            scenario = self._scenario_from_controls()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error), parent=self)
            return
        self._stop_replay()
        self.scenario = self.display_scenario = scenario
        self.current_snapshot = None
        self.plot_history = []
        self.clear_display()
        self.token += 1
        token = self.token
        control = self.control = WorkerControl(1 / self.speed_var.get())
        algorithm = self.algorithm_var.get()
        self._set_busy(True, demo=True)
        self.notebook.select(self.animation_tab)
        self.status_var.set(f"Running {algorithm} · seed {seed} · {settings.budget:,} evaluations")

        def work():
            try:
                optimizer = Optimizer(scenario, algorithm, seed, settings)
                while not optimizer.done and control.wait_turn():
                    self.events.put((token, "demo_progress", optimizer.advance()))
                    if not optimizer.done:
                        control.cancelled.wait(control.delay)
                self.events.put((token, "demo_finished", optimizer.result() if optimizer.best else None))
            except Exception as error:
                self.events.put((token, "error", str(error)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def run_comparison(self):
        if self.busy:
            return
        try:
            settings, radius = self.read_settings(), float(self.radius_var.get())
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Radius must be positive and finite.")
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error), parent=self)
            return
        self._stop_replay()
        self.benchmark = None
        self._clear_comparison()
        self.token += 1
        token = self.token
        control = self.control = WorkerControl()
        self._set_busy(True)
        self.notebook.select(self.comparison_tab)
        self.status_var.set("Starting 90-run comparison. Benchmark datasets use seeds 101, 202 and 303.")

        def emit(kind, payload):
            # Benchmark rendering is intentionally throttled; complete histories
            # remain in RunResult. Tk never waits for or times queue delivery.
            if kind != "progress" or payload.evaluations % 300 < settings.population or payload.evaluations == settings.budget:
                self.events.put((token, "benchmark_" + kind, payload))

        def work():
            try:
                run_benchmark(settings, radius=radius, on_event=emit,
                              cancelled=control.cancelled.is_set)
            except Exception as error:
                self.events.put((token, "error", str(error)))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _poll(self):
        if self.closed:
            return
        for _ in range(150):
            try:
                token, kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if token != self.token:
                continue
            if kind == "demo_progress":
                self.plot_history.append((payload.evaluations, payload.best.objective if payload.best.feasible else None))
                self.display_snapshot(payload)
            elif kind == "demo_finished":
                if payload:
                    self.demo_runs[payload.algorithm] = payload
                    self.status_var.set(f"{payload.algorithm} {'completed' if payload.completed else 'cancelled'}: "
                                        f"{payload.evaluations:,} evaluations; solver {payload.solver_seconds:.3f}s.")
                self._set_busy(False)
            elif kind == "benchmark_scenario":
                self.display_scenario = payload
            elif kind == "benchmark_run_started":
                self.display_scenario = payload["scenario"]
                self.current_snapshot = None
                self.plot_history = []
                self.status_var.set(f"Run {payload['number']}/{payload['total']}: "
                                    f"{payload['scenario'].name} · {payload['algorithm']} · seed {payload['seed']}")
            elif kind == "benchmark_progress":
                self.plot_history.append((payload.evaluations, payload.best.objective if payload.best.feasible else None))
                if self.notebook.select() == str(self.animation_tab):
                    self.display_snapshot(payload)
            elif kind == "benchmark_run_finished":
                self.benchmark_progress["value"] = float(self.benchmark_progress["value"]) + int(payload.completed)
            elif kind == "benchmark_finished":
                self.benchmark = payload
                self._refresh_comparison()
                self._set_busy(False)
                count = sum(r.completed for r in payload.runs)
                self.status_var.set(f"Comparison {'complete' if payload.completed else 'cancelled'}: "
                                    f"{count}/{payload.planned_runs} completed runs. Select a saved run to replay or export the results.")
            elif kind == "error":
                self._set_busy(False)
                self.status_var.set("Run stopped because of an error.")
                messagebox.showerror("Run error", payload, parent=self)
        self._poll_id = self.after(40, self._poll)

    def pause(self):
        if self.busy and self.control:
            with self.control.condition:
                self.control.paused = not self.control.paused
                self.control.steps = 0
                self.control.condition.notify_all()
                paused = self.control.paused
            self.pause_button.configure(text="Resume" if paused else "Pause")
            self.status_var.set("Paused at the next generation/iteration boundary." if paused else "Search resumed.")
        elif self.replay_frames:
            self.replay_playing = not self.replay_playing
            self.pause_button.configure(text="Pause replay" if self.replay_playing else "Resume replay")
            self._cancel_replay_timer()
            if self.replay_playing:
                self._replay_tick(self.replay_token)

    def step(self):
        if self.busy and self.control:
            with self.control.condition:
                self.control.paused = True
                self.control.steps += 1
                self.control.condition.notify_all()
            self.pause_button.configure(text="Resume")
        elif self.replay_frames:
            self.replay_playing = False
            self._cancel_replay_timer()
            self.pause_button.configure(text="Resume replay")
            self._replay_frame()

    def reset(self):
        if self.control:
            self.control.cancel()
        self.token += 1
        self._stop_replay()
        self.demo_runs.clear()
        self.benchmark = None
        self._clear_comparison()
        self._set_busy(False)
        self.display_scenario = self.scenario
        self.current_snapshot = None
        self.plot_history = []
        self.clear_display()
        self.status_var.set("Reset. Previous workers are cancelled and stale updates discarded.")

    def _speed_changed(self, *_):
        if self.control:
            self.control.delay = 1 / max(2, self.speed_var.get())

    def clear_display(self):
        for variable in self.metric_vars.values():
            variable.set("—")
        self.map_title.configure(text="School-to-kitchen assignment · synthetic coordinates")
        self.draw_map()
        self.draw_loads()
        self.draw_curve()

    def display_snapshot(self, snapshot):
        previous = self.current_snapshot
        self.current_snapshot = snapshot
        scenario = self.display_scenario
        self.map_title.configure(text=f"{snapshot.algorithm} · {scenario.name} · seed {snapshot.seed} · "
                                      f"generation / iteration {snapshot.iteration}")
        best = snapshot.best
        self.metric_vars["Feasibility"].set("Feasible" if best.feasible else f"Overflow {best.overflow:,}")
        self.metric_vars["Objective · portion-km"].set(f"{best.objective:,.1f}" if best.feasible else "Infeasible")
        self.metric_vars["Weighted distance · km"].set(f"{best.objective / scenario.total_demand:.3f}" if best.feasible else "—")
        self.metric_vars["Evaluations"].set(f"{snapshot.evaluations:,}")
        self.metric_vars["Solver time"].set(f"{snapshot.solver_seconds:.2f}s")
        changed = set()
        if previous and len(previous.best.assignment) == len(best.assignment):
            changed = {i for i, (a, b) in enumerate(zip(previous.best.assignment, best.assignment)) if a != b}
        self.draw_map(changed)
        self.draw_loads()
        self.draw_curve()

    def draw_map(self, changed=None):
        canvas = self.map_canvas
        canvas.delete("all")
        scenario = self.display_scenario
        if not scenario:
            return
        width, height = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
        points = list(scenario.kitchens) + list(scenario.schools)
        xmin, xmax = min(p.x for p in points), max(p.x for p in points)
        ymin, ymax = min(p.y for p in points), max(p.y for p in points)
        if self.radius_visible.get():
            xmin = min(xmin, min(k.x - scenario.radius for k in scenario.kitchens))
            xmax = max(xmax, max(k.x + scenario.radius for k in scenario.kitchens))
            ymin = min(ymin, min(k.y - scenario.radius for k in scenario.kitchens))
            ymax = max(ymax, max(k.y + scenario.radius for k in scenario.kitchens))
        scale = min((width - 90) / max(1, xmax - xmin), (height - 75) / max(1, ymax - ymin))
        scale = max(0.1, scale)
        xoffset = (width - (xmax - xmin) * scale) / 2
        yoffset = (height - (ymax - ymin) * scale) / 2

        def xy(point):
            return xoffset + (point.x - xmin) * scale, height - yoffset - (point.y - ymin) * scale

        for x in range(0, width, 40):
            canvas.create_line(x, 0, x, height, fill="#f2f5f8")
        for y in range(0, height, 40):
            canvas.create_line(0, y, width, y, fill="#f2f5f8")
        if self.radius_visible.get():
            for j, kitchen in enumerate(scenario.kitchens):
                x, y = xy(kitchen)
                r = scenario.radius * scale
                canvas.create_oval(x - r, y - r, x + r, y + r, outline=COLORS[j % len(COLORS)], dash=(4, 5))
        assignment = self.current_snapshot.best.assignment if self.current_snapshot else None
        if assignment:
            for i, j in enumerate(assignment):
                x, y = xy(scenario.schools[i])
                kx, ky = xy(scenario.kitchens[j])
                canvas.create_line(x, y, kx, ky, fill=COLORS[j % len(COLORS)],
                                   width=3 if changed and i in changed else 1, dash=(3, 2) if changed and i in changed else ())
        for i, school in enumerate(scenario.schools):
            x, y = xy(school)
            color = COLORS[assignment[i] % len(COLORS)] if assignment else "#a0adba"
            r = 4 + 2 * school.demand / 300
            if changed and i in changed:
                canvas.create_oval(x - r - 4, y - r - 4, x + r + 4, y + r + 4, outline="#19304a", width=2)
            canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=1)
            if len(scenario.schools) <= 40:
                canvas.create_text(x + 7, y - 9, text=school.id, fill=MUTED, anchor="w", font=("Segoe UI", 8))
        for j, kitchen in enumerate(scenario.kitchens):
            x, y = xy(kitchen)
            canvas.create_rectangle(x - 8, y - 8, x + 8, y + 8, fill=COLORS[j % len(COLORS)], outline=INK, width=2)
            canvas.create_text(x, y + 19, text=kitchen.id, fill=INK, font=("Segoe UI", 10, "bold"))
        canvas.create_text(12, 12, anchor="nw", text="■ kitchen   ● school (size = demand)   — assignment", fill=MUTED, font=("Segoe UI", 9))
        # A true map scale independent of the decorative grid.
        scale_km = max(0.5, round(65 / scale * 2) / 2)
        bar = scale_km * scale
        canvas.create_line(15, height - 20, 15 + bar, height - 20, fill=INK, width=2)
        canvas.create_text(15, height - 25, anchor="sw", text=f"{scale_km:g} km", fill=MUTED, font=("Segoe UI", 8))

    def draw_loads(self):
        canvas = self.load_canvas
        canvas.delete("all")
        scenario = self.display_scenario
        if not scenario:
            return
        width, height = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
        loads = self.current_snapshot.best.loads if self.current_snapshot else [0] * len(scenario.kitchens)
        row_height = min(32, (height - 12) / len(scenario.kitchens))
        max_ratio = max(1, max(load / k.capacity for load, k in zip(loads, scenario.kitchens)))
        for j, (kitchen, load) in enumerate(zip(scenario.kitchens, loads)):
            y = 10 + j * row_height
            canvas.create_text(10, y + row_height / 2, anchor="w", text=kitchen.id, fill=INK, font=("Segoe UI", 9))
            left, right = 43, max(65, width - 105)
            canvas.create_rectangle(left, y + 4, right, y + row_height - 3, fill="#edf2f7", outline="")
            end = left + (right - left) * load / kitchen.capacity / max_ratio
            color = "#ce5265" if load > kitchen.capacity else COLORS[j % len(COLORS)]
            if end > left:
                canvas.create_rectangle(left, y + 4, end, y + row_height - 3, fill=color, outline="")
            cap_x = left + (right - left) / max_ratio
            canvas.create_line(cap_x, y + 2, cap_x, y + row_height - 1, fill=INK)
            canvas.create_text(width - 7, y + row_height / 2, anchor="e", text=f"{load:,}/{kitchen.capacity:,}", fill=INK, font=("Segoe UI", 8))

    def _plot(self, canvas, series, xmax=None):
        canvas.delete("all")
        width, height = max(150, canvas.winfo_width()), max(120, canvas.winfo_height())
        left, top, right, bottom = 64, 28, width - 16, height - 34
        points = [(x, y) for _, _, values in series for x, y in values if y is not None]
        if not points:
            canvas.create_text(width / 2, height / 2, text="No feasible solution recorded yet", fill=MUTED, font=("Segoe UI", 10))
            return
        xmin, xmax = 0, xmax or max(x for x, y in points)
        low, high = min(y for x, y in points), max(y for x, y in points)
        margin = max(1, (high - low) * 0.1)
        low, high = max(0, low - margin), high + margin
        for fraction in (0, 0.5, 1):
            y = bottom - fraction * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#e5ebf2")
            canvas.create_text(left - 6, y, anchor="e", text=f"{low + fraction * (high - low):,.0f}", fill=MUTED, font=("Segoe UI", 8))
            x = left + fraction * (right - left)
            canvas.create_text(x, bottom + 12, text=f"{fraction * xmax:,.0f}", fill=MUTED, font=("Segoe UI", 8))
        canvas.create_text((left + right) / 2, height - 6, text="candidate evaluations", fill=MUTED, font=("Segoe UI", 8))
        legend_x = left
        for name, color, values in series:
            canvas.create_text(legend_x, 12, text=name, anchor="w", fill=color, font=("Segoe UI", 9, "bold"))
            legend_x += 115
            segment = []

            def flush():
                if len(segment) >= 4:
                    canvas.create_line(*segment, fill=color, width=2)
                elif segment:
                    x, y = segment
                    canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)

            for x, y in values:
                if y is None:
                    flush()
                    segment = []
                else:
                    segment.extend((left + x / max(1, xmax) * (right - left),
                                    bottom - (y - low) / (high - low) * (bottom - top)))
            flush()

    def draw_curve(self):
        algorithm = self.current_snapshot.algorithm if self.current_snapshot else self.algorithm_var.get()
        self._plot(self.curve_canvas, [(algorithm, COLORS[ALGORITHMS.index(algorithm)], self.plot_history)])

    def _refresh_comparison(self):
        for row in self.summary_table.get_children():
            self.summary_table.delete(row)
        if not self.benchmark:
            return

        def fmt(value, places=1):
            return "—" if value is None else f"{value:,.{places}f}"

        for row in self.benchmark.summaries():
            values = (row["scenario"], row["algorithm"], f"{row['feasible_runs']}/{row['runs']}",
                      fmt(row["best_objective"]), fmt(row["mean_objective"]), fmt(row["sample_std"]),
                      fmt(row["mean_solver_seconds"], 3), fmt(row["baseline_objective"]))
            self.summary_table.insert("", "end", values=values)
        self.replay_lookup = {f"{run.scenario_name} · {run.algorithm} · seed {run.seed}"
                              + (" · partial" if not run.completed else ""): run
                              for run in self.benchmark.runs if run.snapshots}
        self.replay_box.configure(values=tuple(self.replay_lookup))
        if self.replay_lookup:
            self.replay_var.set(next(iter(self.replay_lookup)))
        self.draw_comparison_curve()

    def draw_comparison_curve(self):
        if not hasattr(self, "compare_curve"):
            return
        series = []
        if self.benchmark:
            for a, algorithm in enumerate(ALGORITHMS):
                checkpoints = {}
                for run in self.benchmark.runs:
                    if run.scenario_name == self.compare_size_var.get() and run.algorithm == algorithm and run.completed:
                        for entry in run.history:
                            checkpoints.setdefault(entry["evaluations"], []).append(entry["objective"])
                values = []
                for x, ys in sorted(checkpoints.items()):
                    feasible = [y for y in ys if y is not None]
                    values.append((x, sum(feasible) / len(feasible) if feasible else None))
                series.append((algorithm, COLORS[a], values))
        self._plot(self.compare_curve, series)

    def _stop_replay(self):
        self._cancel_replay_timer()
        self.replay_token += 1
        self.replay_playing = False
        self.replay_frames = []
        self.replay_index = 0
        if hasattr(self, "pause_button") and not self.busy:
            self.pause_button.configure(state="disabled", text="Pause")
            self.step_button.configure(state="disabled")

    def _cancel_replay_timer(self):
        if self._replay_after_id is not None:
            self.after_cancel(self._replay_after_id)
            self._replay_after_id = None

    def start_replay(self):
        if self.busy or self.replay_var.get() not in self.replay_lookup:
            return
        self._stop_replay()
        run = self.replay_lookup[self.replay_var.get()]
        self.display_scenario = next(s for s in self.benchmark.scenarios
                                     if (s.name, s.seed) == (run.scenario_name, run.scenario_seed))
        self.current_snapshot = None
        self.plot_history = []
        self.replay_frames = run.snapshots
        self.replay_playing = True
        self.pause_button.configure(state="normal", text="Pause replay")
        self.step_button.configure(state="normal")
        self.notebook.select(self.animation_tab)
        self.status_var.set(f"Replaying {self.replay_var.get()} — recorded solver time is unchanged.")
        self._replay_tick(self.replay_token)

    def _replay_frame(self):
        if self.replay_index >= len(self.replay_frames):
            self.replay_playing = False
            self.pause_button.configure(text="Replay ended", state="disabled")
            self.step_button.configure(state="disabled")
            return
        snapshot = self.replay_frames[self.replay_index]
        self.replay_index += 1
        self.plot_history.append((snapshot.evaluations, snapshot.best.objective if snapshot.best.feasible else None))
        self.display_snapshot(snapshot)

    def _replay_tick(self, token):
        self._replay_after_id = None
        if self.closed or token != self.replay_token or not self.replay_playing:
            return
        self._replay_frame()
        if self.replay_playing:
            self._replay_after_id = self.after(round(1000 / self.speed_var.get()), lambda: self._replay_tick(token))

    def export(self):
        if self.busy or not (self.benchmark or self.demo_runs):
            return
        directory = filedialog.askdirectory(title="Choose a results directory", parent=self)
        if not directory:
            return
        try:
            if self.benchmark:
                export_benchmark(self.benchmark, directory)
            else:
                export_demo(self.scenario, list(self.demo_runs.values()), directory)
        except (OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return
        self.status_var.set(f"Exported benchmark.json, runs.csv and summary.csv to {directory}")

    def close(self):
        self.closed = True
        if self.control:
            self.control.cancel()
        self._stop_replay()
        self.after_cancel(self._poll_id)
        self.destroy()
