"""Tkinter dashboard. Workers only touch Python objects, never Tk widgets."""

import math
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageTk

from .algorithms import ALGORITHMS, ALGORITHM_LABELS, CONSTRAINT_POLICIES, Optimizer, SolverSettings
from .experiments import export_benchmark, export_demo, run_benchmark
from .model import SIZES, DIFFICULTIES, generate_scenario, greedy_baseline, delivery_metrics, vehicle_tours
from .search_view import SearchView
from .osm import CITIES, CACHE_DIR, FOCUS_CENTER, FOCUS_URL
from .basemap import VIEW_URL, load_view_tiles, paint_view


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
        self.live_snapshots = {}
        self.live_histories = {}
        self.race_running = False
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
        self._delivery_after_id = None
        self._delivery_started = time.monotonic()
        self._delivery_clock_minutes = 0.
        self._delivery_last_tick = time.monotonic()
        self._school_markers = []
        self._delivery_routes = []
        self._map_cache_key = None
        self._map_road_tables = None
        self._basemap_raster = None
        self._basemap_origin = None
        self._map_image = None
        self._initializing = True
        self.size_var = tk.StringVar(value="Small")
        self.dataset_seed_var = tk.StringVar(value="101")
        self.radius_var = tk.StringVar(value="6")
        self.algorithm_var = tk.StringVar(value="GA")
        self.optimizer_seed_var = tk.StringVar(value="1")
        self.budget_var = tk.StringVar(value="5000")
        self.population_var = tk.StringVar(value="30")
        self.difficulty_var = tk.StringVar(value="Standard")
        self.slack_var = tk.StringVar(value="0.05")
        self.scenario_mode_var = tk.StringVar(value="Synthetic planning")
        self.school_count_var = tk.StringVar(value="30")
        self.school_seed_var = tk.StringVar(value="101")
        self.school_demand_var = tk.StringVar(value="200")
        self.sppg_capacity_var = tk.StringVar(value="3000")
        self.opening_weight_var = tk.StringVar(value="10")
        self.distance_weight_var = tk.StringVar(value="1")
        self.instances_var = tk.StringVar(value="3")
        self.optimizer_count_var = tk.StringVar(value="10")
        self.policy_var = tk.StringVar(value="shared_repair")
        self.oracle_limit_var = tk.StringVar(value="60")
        self.map_source_var = tk.StringVar(value="OpenStreetMap")
        self.osm_city_var = tk.StringVar(value="Ciledug")
        self.speed_var = tk.DoubleVar(value=10)
        self.status_var = tk.StringVar(value="Generate a scenario, then run an algorithm or the full comparison.")
        self.metric_vars = {name: tk.StringVar(value="—") for name in
                            ("Feasibility", "Objective · portion-min", "Weighted distance · km", "Evaluations", "Solver time",
                             "MILP optimum", "MILP lower bound", "Gap · %", "Repair changed · %", "Decoded unique · %",
                             "Weighted ETA · min", "Max ETA · min")}
        defaults = SolverSettings()
        self.parameter_vars = {name: tk.StringVar(value="auto" if name == "mutation" else str(getattr(defaults, name)))
                               for name in ("crossover", "mutation", "elites", "tournament", "inertia",
                                            "cognitive", "social", "velocity_limit", "alpha", "beta", "evaporation")}
        self.input_widgets = []
        self.parameter_widgets = []
        self._style()
        self._build()
        for variable in (self.dataset_seed_var, self.radius_var, self.optimizer_seed_var,
                         self.budget_var, self.population_var, self.slack_var, self.instances_var,
                         self.optimizer_count_var, self.policy_var, self.oracle_limit_var,
                         self.map_source_var, self.osm_city_var, *self.parameter_vars.values()):
            variable.trace_add("write", self._invalidate)
        for variable in (self.scenario_mode_var, self.school_count_var, self.school_seed_var,
                         self.school_demand_var, self.sppg_capacity_var, self.opening_weight_var, self.distance_weight_var):
            variable.trace_add("write", self._invalidate)
        self.algorithm_var.trace_add("write", self._algorithm_changed)
        self.size_var.trace_add("write", self._size_changed)
        self.difficulty_var.trace_add("write", self._difficulty_changed)
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
        ttk.Label(header, text="Ciledug SPPG comparison", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Pedurenan / Ciledug OSM | Fixed synthetic schools | SPPG quantity, capacity and road distance",
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
        self._field(sidebar, "Candidate pool", self.size_var, 1, tuple(SIZES))
        self._field(sidebar, "Candidate-site seed", self.dataset_seed_var, 2)
        self._field(sidebar, "Radius · km", self.radius_var, 3)
        self.generate_button = ttk.Button(sidebar, text="Generate scenario", command=self.generate)
        self.generate_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 14))
        ttk.Label(sidebar, text="Search", style="Heading.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.algorithm_box = self._field(sidebar, "View algorithm", self.algorithm_var, 6, ALGORITHMS)
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
        self.run_button = ttk.Button(controls, text="Run GA / PSO / ACO", command=self.run_all, style="Primary.TButton")
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=3)
        self.pause_button = ttk.Button(controls, text="Pause", command=self.pause, state="disabled")
        self.pause_button.grid(row=1, column=0, sticky="ew", pady=3, padx=(0, 3))
        self.step_button = ttk.Button(controls, text="Step", command=self.step, state="disabled")
        self.step_button.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(controls, text="Reset / cancel", command=self.reset).grid(row=2, column=0, columnspan=2, sticky="ew", pady=3)
        self.selected_button = ttk.Button(controls, text="Run selected only", command=self.run_demo)
        self.selected_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(sidebar, text="Playback speed · updates / sec", style="Muted.TLabel").grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Scale(sidebar, from_=2, to=60, variable=self.speed_var, command=self._speed_changed).grid(row=13, column=0, columnspan=2, sticky="ew")
        self.compare_button = ttk.Button(sidebar, text="Compare all instances", command=self.run_comparison)
        self.compare_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(12, 3))
        self.export_button = ttk.Button(sidebar, text="Export results…", command=self.export, state="disabled")
        self.export_button.grid(row=15, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Label(sidebar, text="Fixed synthetic schools on OSM roads.\nSeparate one-way allocation distances.\nEqual budgets + exact MILP.\nPSO = preference-score / discrete.",
                  style="Muted.TLabel", justify="left").grid(row=16, column=0, columnspan=2, sticky="w", pady=(12, 0))
        for widget in sidebar.winfo_children():
            info = widget.grid_info()
            if info and int(info["row"]) >= 4:
                widget.grid_configure(row=int(info["row"]) + 1)
        experiment_fields = ttk.Frame(sidebar)
        experiment_fields.grid(row=4, column=0, columnspan=2, sticky="ew", pady=5)
        experiment_fields.columnconfigure(1, weight=1)
        self._field(experiment_fields, "School model", self.scenario_mode_var, 0, ("Synthetic planning", "Mapped school tours"))
        self._field(experiment_fields, "School count", self.school_count_var, 1)
        self._field(experiment_fields, "School location seed", self.school_seed_var, 2)
        self._field(experiment_fields, "Meals / school", self.school_demand_var, 3)
        self._field(experiment_fields, "SPPG capacity", self.sppg_capacity_var, 4)
        self._field(experiment_fields, "SPPG opening weight", self.opening_weight_var, 5)
        self._field(experiment_fields, "Distance weight", self.distance_weight_var, 6)
        self._field(experiment_fields, "Instances / size", self.instances_var, 7)
        self._field(experiment_fields, "Optimizer runs", self.optimizer_count_var, 8)
        self._field(experiment_fields, "Constraint policy", self.policy_var, 9, CONSTRAINT_POLICIES)
        self._field(experiment_fields, "MILP limit · sec", self.oracle_limit_var, 10)
        ttk.Label(experiment_fields, text="Pedurenan OSM view\n-6.22253, 106.69706\nPool: Small 6 / Medium 10 / Large 16\nSynthetic meals, not surveyed demand", wraplength=220).grid(row=11, column=0, columnspan=2, sticky="w", pady=8)
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
        frame.rowconfigure(2, weight=1)
        self.map_title = ttk.Label(frame, text="SPPG sites and multi-stop tours", style="Heading.TLabel")
        self.map_title.grid(row=0, column=0, sticky="w", pady=(0, 9))
        metrics = ttk.Frame(frame)
        self.metric_labels = {}
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column, (label, variable) in enumerate(self.metric_vars.items()):
            metrics.columnconfigure(column % 6, weight=1)
            card = ttk.Frame(metrics, style="Card.TFrame", padding=6)
            card.grid(row=column // 6, column=column % 6, sticky="nsew", padx=(0, 6), pady=(0, 6))
            heading = ttk.Label(card, text=label, style="Card.TLabel", font=("Segoe UI", 9))
            heading.pack(anchor="w")
            self.metric_labels[label] = heading
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(4, 0))
        self.search_notebook = ttk.Notebook(frame)
        self.search_notebook.grid(row=2, column=0, sticky="nsew")
        map_frame = self.road_tab = ttk.Frame(self.search_notebook, style="Card.TFrame")
        self.search_notebook.add(map_frame, text="Road map & deliveries")
        self.search_view = SearchView(self.search_notebook)
        self.search_notebook.add(self.search_view, text="PSO / ACO search")
        tour_frame = self.tour_frame = ttk.Frame(self.search_notebook)
        self.search_notebook.add(tour_frame, text="Vehicle tours & arrivals")
        tour_frame.columnconfigure(0, weight=1)
        tour_frame.rowconfigure(0, weight=1)
        columns = ("sppg", "stop", "school", "meals", "arrival", "leave", "duration")
        self.tour_table = ttk.Treeview(tour_frame, columns=columns, show="tree headings")
        self.tour_table.heading("#0", text="Vehicle / tour")
        self.tour_table.column("#0", width=130)
        for key, title in zip(columns, ("SPPG", "Stop", "Mapped school", "Meals", "Arrival min", "Leave min", "Tour min")):
            self.tour_table.heading(key, text=title)
            self.tour_table.column(key, width=210 if key == "school" else 75, stretch=True)
        self.tour_table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(tour_frame, orient="vertical", command=self.tour_table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tour_table.configure(yscrollcommand=scrollbar.set)
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(map_frame, bg="white", highlightthickness=0, height=340)
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        map_controls = ttk.Frame(map_frame)
        map_controls.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.radius_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(map_controls, text="Show service radii", variable=self.radius_visible,
                        command=self.draw_map).pack(side="left", padx=5)
        self.delivery_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(map_controls, text="Animate deliveries (12 s)", variable=self.delivery_visible,
                        command=self.draw_map).pack(side="left", padx=5)
        self.full_network_visible = tk.BooleanVar(value=False)
        self.school_names_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(map_controls, text="Show school names", variable=self.school_names_visible,
                        command=self.draw_map).pack(side="left", padx=5)
        self.basemap_button = ttk.Button(map_controls, text="Load OSM background", command=self.load_basemap)
        self.basemap_button.pack(side="right", padx=5)
        ttk.Checkbutton(map_controls, text="Full network", variable=self.full_network_visible,
                        command=self.draw_map).pack(side="right", padx=5)
        self.map_canvas.bind("<Configure>", lambda event: self.draw_map())
        bottom = ttk.Frame(self.search_notebook, padding=8)
        self.search_notebook.add(bottom, text="Convergence & capacity")
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
        ttk.Label(frame, text="Certified gaps · equal-weight instance means · solver time excludes MILP and playback",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(5, 10))
        self.benchmark_progress = ttk.Progressbar(frame, maximum=90)
        self.benchmark_progress.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        columns = ("scenario", "algorithm", "instances", "feasible", "gap", "best", "mean", "std", "time", "baseline", "eta", "max_eta")
        self.summary_table = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for key, title in zip(columns, ("Scenario", "Method", "Instances", "Feasible", "Gap %", "Best J", "Mean J", "Std. dev.", "Solver sec.", "Greedy J", "ETA min", "Max ETA")):
            self.summary_table.heading(key, text=title)
            self.summary_table.column(key, width=65, minwidth=50, anchor="center", stretch=True)
        self.summary_table.grid(row=3, column=0, sticky="nsew")
        plots = ttk.Frame(frame)
        plots.grid(row=4, column=0, sticky="ew", pady=(14, 8))
        self.compare_metric_var = tk.StringVar(value="Objective")
        metric_box = ttk.Combobox(plots, textvariable=self.compare_metric_var, values=("Objective", "Certified gap", "Gap to MILP bound"), state="readonly", width=24)
        metric_box.pack(side="left")
        metric_box.bind("<<ComboboxSelected>>", lambda event: self.draw_comparison_curve())
        self.compare_size_var = tk.StringVar(value="Small")
        self.compare_size_box = ttk.Combobox(plots, textvariable=self.compare_size_var, values=tuple(SIZES), state="readonly", width=12)
        self.compare_size_box.pack(side="right")
        self.compare_size_box.bind("<<ComboboxSelected>>", lambda event: self.draw_comparison_curve())
        self.compare_curve = tk.Canvas(frame, bg="white", highlightthickness=0, height=210)
        self.compare_curve.grid(row=5, column=0, sticky="ew")
        self.compare_curve.bind("<Configure>", lambda event: self.draw_comparison_curve())
        ttk.Label(frame, text="Curves average optimizer runs within each instance, then weight instances equally. Unavailable gaps are excluded.",
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

    def _difficulty_changed(self, *_):
        self.slack_var.set(str(DIFFICULTIES[self.difficulty_var.get()]))

    def _algorithm_changed(self, *_):
        if self._initializing:
            return
        if self.race_running:
            snapshot = self.live_snapshots.get(self.algorithm_var.get())
            if snapshot:
                self.plot_history = self.live_histories[self.algorithm_var.get()][:]
                self.display_snapshot(snapshot)
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
        self.live_snapshots.clear()
        self.race_running = False
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
        settings = SolverSettings(budget=int(self.budget_var.get()), population=int(self.population_var.get()),
                                  constraint_policy=self.policy_var.get(), oracle_time_limit=float(self.oracle_limit_var.get()), **values)
        settings.validate()
        return settings

    def _planning_options(self):
        if self.scenario_mode_var.get() == "Mapped school tours":
            return {}
        return dict(school_count=int(self.school_count_var.get()), school_seed=int(self.school_seed_var.get()),
                    school_demand=int(self.school_demand_var.get()), sppg_capacity=int(self.sppg_capacity_var.get()),
                    opening_weight=float(self.opening_weight_var.get()), distance_weight=float(self.distance_weight_var.get()))

    def _scenario_from_controls(self):
        arguments = (self.size_var.get(), int(self.dataset_seed_var.get()), float(self.radius_var.get()),
                     float(self.slack_var.get()), self.difficulty_var.get())
        options = self._planning_options()
        signature = (*arguments, self.map_source_var.get(), self.osm_city_var.get(), tuple(options.items()))
        if getattr(self, "_scenario_signature", None) == signature:
            return self.scenario
        if self.map_source_var.get() == "OpenStreetMap" and not (CACHE_DIR / f"{self.osm_city_var.get().lower()}.json").exists():
            raise ValueError("Click Generate scenario first to download the OpenStreetMap area in the background.")
        return generate_scenario(*arguments,
                                 map_source=self.map_source_var.get(), osm_city=self.osm_city_var.get(), **options)

    def generate(self):
        if self.busy:
            return
        if self.map_source_var.get() == "OpenStreetMap":
            try:
                arguments = (self.size_var.get(), int(self.dataset_seed_var.get()), float(self.radius_var.get()),
                             float(self.slack_var.get()), self.difficulty_var.get())
                options = self._planning_options()
                if not math.isfinite(arguments[2]) or arguments[2] <= 0:
                    raise ValueError("Service radius must be positive and finite.")
            except ValueError as error:
                messagebox.showerror("Invalid scenario", str(error), parent=self)
                return
            self._invalidate()
            self.token += 1
            token, city = self.token, self.osm_city_var.get()
            self._set_busy(True)
            self.status_var.set(f"Loading OpenStreetMap {city}; cached downloads are reused. Ciledug includes Ciledug Raya.")
            def load():
                try:
                    scenario = generate_scenario(*arguments, map_source="OpenStreetMap", osm_city=city, **options)
                    self.events.put((token, "osm_generated", scenario))
                except Exception as error:
                    self.events.put((token, "error", str(error)))
            self.worker = threading.Thread(target=load, daemon=True)
            self.worker.start()
            return
        try:
            scenario = self._scenario_from_controls()
        except ValueError as error:
            messagebox.showerror("Invalid scenario", str(error), parent=self)
            return
        self._accept_scenario(scenario)

    def _accept_scenario(self, scenario):
        self._invalidate()
        self.scenario = self.display_scenario = scenario
        self._scenario_signature = (self.size_var.get(), int(self.dataset_seed_var.get()), float(self.radius_var.get()),
                                    float(self.slack_var.get()), self.difficulty_var.get(),
                                    self.map_source_var.get(), self.osm_city_var.get(), tuple(self._planning_options().items()))
        self.clear_display()
        if scenario.objective_metric == "facility_distance" and self._basemap_raster is None and str(self.basemap_button["state"]) != "disabled":
            self.load_basemap()
        baseline = greedy_baseline(scenario)
        baseline_text = f"greedy J {baseline.objective:,.1f}" if baseline.feasible else f"greedy overflow {baseline.overflow}"
        school_limit = (f" Up to {scenario.kitchens[0].capacity//scenario.schools[0].demand} schools per SPPG at this demand."
                        if scenario.objective_metric == "facility_distance" else "")
        self.status_var.set(f"{scenario.name}: {len(scenario.kitchens)} candidate SPPGs / {scenario.max_active_kitchens or len(scenario.kitchens)} active limit, {len(scenario.schools)} fixed schools, "
                            f"{scenario.total_demand:,} portions/day; {baseline_text}. "
                            f"{scenario.road_source.get('city', 'Synthetic')} · {scenario.road_source.get('school_locations', 'simulated schools')}.{school_limit} Select an algorithm to begin.")

    def load_basemap(self):
        self.basemap_button.configure(state="disabled")
        self.status_var.set("Loading the displayed Pedurenan OSM raster; cached tiles are reused.")
        def load():
            try:
                self.events.put((None, "basemap_loaded", load_view_tiles()))
            except Exception as error:
                self.events.put((None, "basemap_error", str(error)))
        threading.Thread(target=load, daemon=True).start()

    def _set_busy(self, busy, demo=False):
        self.busy = busy
        for widget in self.input_widgets:
            widget.configure(state="disabled" if busy else ("readonly" if isinstance(widget, ttk.Combobox) else "normal"))
        for button in (self.generate_button, self.run_button, self.selected_button, self.compare_button):
            button.configure(state="disabled" if busy else "normal")
        if busy and self.race_running:
            self.algorithm_box.configure(state="readonly")
        self.pause_button.configure(state="normal" if busy and demo else "disabled", text="Pause")
        self.step_button.configure(state="normal" if busy and demo else "disabled")
        self.replay_button.configure(state="normal" if not busy and self.replay_lookup else "disabled")
        self.export_button.configure(state="normal" if not busy and (self.benchmark or self.demo_runs) else "disabled")

    def run_all(self):
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
        self.demo_runs.clear()
        self.live_snapshots.clear()
        self.live_histories = {a: [] for a in ALGORITHMS}
        self.current_snapshot = None
        self.plot_history = []
        self.clear_display()
        self.token += 1
        token = self.token
        control = self.control = WorkerControl(1/self.speed_var.get())
        self.race_running = True
        self._set_busy(True, demo=True)
        self.notebook.select(self.animation_tab)
        self.search_notebook.select(self.road_tab)
        self.status_var.set("Solving the exact MILP, then comparing three optimizers at equal evaluation counts.")
        def work():
            try:
                optimizers = [Optimizer(scenario, a, seed, settings, capture_search=True) for a in ALGORITHMS]
                while not all(o.done for o in optimizers) and control.wait_turn():
                    snapshots = []
                    # GA elites use fewer fresh evaluations in later generations;
                    # advance to a common batch boundary before publishing maps.
                    boundary = min(settings.budget, min(o.evaluations for o in optimizers)+settings.population)
                    for o in optimizers:
                        while o.evaluations < boundary and not control.cancelled.is_set():
                            snapshots.append(o.advance_to(boundary))
                    self.events.put((token, "race_progress", snapshots))
                    if not all(o.done for o in optimizers):
                        control.cancelled.wait(control.delay)
                self.events.put((token, "race_finished", [o.result() for o in optimizers if o.best]))
            except Exception as error:
                self.events.put((token, "error", str(error)))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

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
        self.search_notebook.select(self.search_view if algorithm in ("PSO", "ACO") else self.road_tab)
        self.status_var.set(f"Solving MILP, then running {ALGORITHM_LABELS[algorithm]} · optimizer {seed} · "
                            f"scenario {scenario.seed} · {settings.constraint_policy}")

        def work():
            try:
                optimizer = Optimizer(scenario, algorithm, seed, settings, capture_search=True)
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
            instances, optimizer_count = int(self.instances_var.get()), int(self.optimizer_count_var.get())
            slack, difficulty = float(self.slack_var.get()), self.difficulty_var.get()
            map_source, osm_city = self.map_source_var.get(), self.osm_city_var.get()
            planning = self._planning_options()
            if instances < 1 or optimizer_count < 1 or not math.isfinite(slack) or slack <= -1:
                raise ValueError("Instance and optimizer counts must be positive; slack must be finite and greater than -1.")
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
        self.benchmark_progress["maximum"] = 3 * instances * optimizer_count * 3
        self.status_var.set(f"Starting {3 * instances * optimizer_count * 3}-run comparison · {settings.constraint_policy}.")

        def emit(kind, payload):
            # Benchmark rendering is intentionally throttled; complete histories
            # remain in RunResult. Tk never waits for or times queue delivery.
            if kind != "progress" or payload.evaluations % 300 < settings.population or payload.evaluations == settings.budget:
                self.events.put((token, "benchmark_" + kind, payload))

        def work():
            try:
                run_benchmark(settings, radius=radius, on_event=emit,
                              cancelled=control.cancelled.is_set, instances_per_size=instances,
                              optimizer_seeds=range(1, optimizer_count + 1), capacity_slack=slack, difficulty=difficulty,
                              capture_search=True, map_source=map_source, osm_city=osm_city, **planning)
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
            if kind == "basemap_loaded":
                self._basemap_raster, self._basemap_origin = payload
                self.basemap_button.configure(state="normal")
                self._map_cache_key = None
                self.draw_map()
                self.status_var.set("OSM raster loaded; markers and shortest-road paths are geographically aligned.")
                continue
            if kind == "basemap_error":
                self.basemap_button.configure(state="normal")
                self.status_var.set("OSM background unavailable; using the rasterized OSM road graph. " + payload)
                continue
            if token != self.token:
                continue
            if kind == "race_progress":
                for snapshot in payload:
                    self.live_snapshots[snapshot.algorithm] = snapshot
                    self.live_histories[snapshot.algorithm].append((snapshot.evaluations, snapshot.best.objective if snapshot.best.feasible else None))
                snapshot = self.live_snapshots.get(self.algorithm_var.get())
                if snapshot:
                    self.plot_history = self.live_histories[snapshot.algorithm][:]
                    self.display_snapshot(snapshot)
                    self.status_var.set(f"GA / PSO / ACO: {snapshot.evaluations:,} evaluations each. Switch View algorithm to inspect allocations and routes.")
            elif kind == "race_finished":
                self.demo_runs = {run.algorithm: run for run in payload}
                self.race_running = False
                self._set_busy(False)
                self.status_var.set("Three-algorithm comparison complete. Switch algorithms to inspect capacity, distance and gaps; export all three runs.")
            elif kind == "demo_progress":
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
            elif kind == "benchmark_oracle_started":
                self.status_var.set(f"MILP oracle: {payload.name} · scenario seed {payload.seed}")
            elif kind == "benchmark_oracle_finished":
                self.status_var.set(f"MILP {payload['oracle'].status} · optimum {payload['oracle'].optimal_objective}")
            elif kind == "benchmark_run_started":
                self.display_scenario = payload["scenario"]
                self.current_snapshot = None
                self.plot_history = []
                self.status_var.set(f"Run {payload['number']}/{payload['total']}: "
                                    f"{payload['scenario'].name} · scenario {payload['scenario'].seed} · "
                                    f"{ALGORITHM_LABELS[payload['algorithm']]} · optimizer {payload['seed']}")
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
            elif kind == "osm_generated":
                self._set_busy(False)
                self._accept_scenario(payload)
            elif kind == "error":
                self.race_running = False
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
            self.search_view.set_paused(paused)
        elif self.replay_frames:
            self.replay_playing = not self.replay_playing
            self.pause_button.configure(text="Pause replay" if self.replay_playing else "Resume replay")
            self._cancel_replay_timer()
            self.search_view.set_paused(not self.replay_playing)
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
        self.live_snapshots.clear()
        self.race_running = False
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

    def _model_labels(self):
        planning = self.display_scenario is not None and self.display_scenario.objective_metric == "facility_distance"
        overrides = {"Feasibility": "Active SPPGs", "Objective · portion-min": "Weighted score",
                     "Weighted distance · km": "Allocation distance · km", "Weighted ETA · min": "Schools / active SPPG",
                     "Max ETA · min": "Capacity lower bound"} if planning else {}
        for key, label in self.metric_labels.items():
            label.configure(text=overrides.get(key, key))
        self.search_notebook.tab(self.tour_frame, text="School allocation & capacity" if planning else "Vehicle tours & arrivals")
        for key, title in (("arrival", "Road km" if planning else "Arrival min"),
                           ("leave", "" if planning else "Leave min"), ("duration", "Utilization" if planning else "Tour min")):
            self.tour_table.heading(key, text=title)
        return planning

    def clear_display(self):
        self.search_view.clear()
        self._delivery_clock_minutes = 0.
        self._delivery_last_tick = time.monotonic()
        self.tour_table.delete(*self.tour_table.get_children())
        for variable in self.metric_vars.values():
            variable.set("—")
        source = self.display_scenario.road_source if self.display_scenario else {}
        planning = self._model_labels()
        self.map_title.configure(text=("SPPG quantity, capacity and allocation distance" if planning else "SPPG sites and multi-stop tours")
                                 + f" · OSM {source.get('city', 'Ciledug')}")
        self.draw_map()
        self.draw_loads()
        self.draw_curve()

    def display_snapshot(self, snapshot):
        previous = self.current_snapshot
        self.current_snapshot = snapshot
        scenario = self.display_scenario
        self.map_title.configure(text=f"{ALGORITHM_LABELS[snapshot.algorithm]} · {scenario.name} · optimizer {snapshot.seed} · "
                                      f"generation / iteration {snapshot.iteration}" +
                                      (f" · OSM {scenario.road_source['city']}" if scenario.road_source.get("source") == "OpenStreetMap" else ""))
        best = snapshot.best
        self.metric_vars["Feasibility"].set(f"{len(set(best.assignment))}/{scenario.max_active_kitchens or len(scenario.kitchens)} active" if best.feasible else f"Overflow {best.overflow:,}; sites +{best.active_site_violation}")
        self.metric_vars["Objective · portion-min"].set(f"{best.objective:,.1f}" if best.feasible else "Infeasible")
        metrics = delivery_metrics(scenario, best, include_routes=False)
        self.metric_vars["Weighted distance · km"].set(f"{metrics['weighted_average_km']:.3f}" if best.feasible else "—")
        for label, key in (("Weighted ETA · min", "weighted_average_minutes"), ("Max ETA · min", "max_school_eta_minutes")):
            value = metrics[key]
            self.metric_vars[label].set(f"{value:.2f}" if value is not None else "—")
        planning = self._model_labels()
        if planning:
            self.metric_vars["Weighted distance · km"].set(f"{metrics['total_allocation_distance_km']:.3f}" if best.feasible else "—")
            self.metric_vars["Weighted ETA · min"].set(f"{len(scenario.schools)/len(set(best.assignment)):.1f}")
            self.metric_vars["Max ETA · min"].set(f"{metrics['capacity_count_lower_bound']} SPPGs")
        self.metric_vars["Evaluations"].set(f"{snapshot.evaluations:,}")
        self.metric_vars["Solver time"].set(f"{snapshot.solver_seconds:.2f}s")
        self.metric_vars["MILP lower bound"].set(f"{snapshot.oracle_lower_bound:,.1f}" if snapshot.oracle_lower_bound is not None else "Unavailable")
        self.metric_vars["MILP optimum"].set(f"{snapshot.optimal_objective:,.1f}" if snapshot.optimal_objective is not None else snapshot.oracle_status)
        if planning and snapshot.optimal_active_sppg_count is not None:
            self.metric_vars["MILP optimum"].set(f"{snapshot.optimal_objective:,.1f} / {snapshot.optimal_active_sppg_count} sites")
        self.metric_vars["Gap · %"].set(f"{snapshot.optimality_gap_percent:.3f}" if snapshot.optimality_gap_percent is not None else (f"{snapshot.lower_bound_gap_percent:.2f} bound" if snapshot.lower_bound_gap_percent is not None else "Unavailable"))
        self.metric_vars["Repair changed · %"].set(f"{100 * snapshot.diagnostics.get('repair_change_rate', 0):.1f}")
        self.metric_vars["Decoded unique · %"].set(f"{100 * snapshot.diagnostics.get('decoded_unique_ratio', 0):.1f}")
        changed = set()
        if previous and len(previous.best.assignment) == len(best.assignment):
            changed = {i for i, (a, b) in enumerate(zip(previous.best.assignment, best.assignment)) if a != b}
        self.draw_map(changed)
        self.draw_loads()
        self.draw_curve()
        paused = bool((self.busy and self.control and self.control.paused)
                      or (self.replay_frames and not self.replay_playing))
        self.search_view.show(scenario, snapshot, duration=1 / max(2, self.speed_var.get()), animate=not paused)
        self.tour_table.delete(*self.tour_table.get_children())
        names = {record["school_id"]: record["name"] for record in scenario.road_source.get("school_features", [])}
        if planning:
            for j in sorted(set(best.assignment)):
                kitchen = scenario.kitchens[j]
                parent = self.tour_table.insert("", "end", text=f"{kitchen.id}: {best.assignment.count(j)} schools", open=True,
                                               values=(kitchen.id, "", "", f"{best.loads[j]}/{kitchen.capacity}", "", "", f"{100*best.loads[j]/kitchen.capacity:.1f}%"))
                for i, school in enumerate(scenario.schools):
                    if best.assignment[i] == j:
                        self.tour_table.insert(parent, "end", values=(kitchen.id, "", school.id, school.demand,
                                              f"{scenario.road_distances[i][j]:.3f}", "", ""))
        for tour in vehicle_tours(scenario, best):
            parent = self.tour_table.insert("", "end", text=f"Vehicle {tour['kitchen']}", open=True,
                                            values=(tour["kitchen"], "", "", tour["initial_load"], 0, "", f"{tour['duration_minutes']:.2f}"))
            for stop, leg in enumerate(tour["legs"], 1):
                i = leg["school_index"]
                school = scenario.schools[i] if i is not None else None
                label = f"{school.id}: {names.get(school.id, school.id)}" if school else "Return to SPPG"
                self.tour_table.insert(parent, "end", values=(tour["kitchen"], stop if school else "Return", label,
                                      school.demand if school else 0, f"{leg['arrival_minutes']:.2f}",
                                      f"{leg['arrival_minutes']+leg['unloading_minutes']:.2f}", ""))

    def draw_map(self, changed=None):
        canvas = self.map_canvas
        if self._delivery_after_id is not None:
            self.after_cancel(self._delivery_after_id)
            self._delivery_after_id = None
        self._delivery_routes = []
        self._school_markers = []
        scenario = self.display_scenario
        if not scenario:
            canvas.delete("all")
            self._map_cache_key = None
            return
        width, height = max(100, canvas.winfo_width()), max(100, canvas.winfo_height())
        cache_key = (id(scenario), width, height, self.radius_visible.get(), self.full_network_visible.get(), id(self._basemap_raster))
        cached = cache_key == self._map_cache_key
        if cached:
            canvas.delete("map_foreground")
        else:
            canvas.delete("all")
        self._map_cache_key = cache_key
        points = list(scenario.kitchens) + list(scenario.schools) + list(scenario.road_nodes)
        xmin, xmax = min(p.x for p in points), max(p.x for p in points)
        ymin, ymax = min(p.y for p in points), max(p.y for p in points)
        focus = scenario.road_source.get("focus_bbox_south_west_north_east")
        if focus and not self.full_network_visible.get():
            lat0, lon0 = scenario.road_source["projection_origin_lat_lon"]
            xmin = 6371.0088*math.radians(focus[1]-lon0)*math.cos(math.radians(lat0))
            xmax = 6371.0088*math.radians(focus[3]-lon0)*math.cos(math.radians(lat0))
            ymin = 6371.0088*math.radians(focus[0]-lat0)
            ymax = 6371.0088*math.radians(focus[2]-lat0)
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

        if cached:
            road_lookup, road_speeds = self._map_road_tables
        else:
            road_lookup = {node.id: node for node in scenario.road_nodes}
            road_speeds = {}
            background = Image.new("RGB", (width, height), "#f5f3ee")
            if self._basemap_raster is not None and scenario.objective_metric == "facility_distance" and not self.full_network_visible.get():
                lat0, lon0 = scenario.road_source["projection_origin_lat_lon"]
                def lat(y):
                    return lat0+math.degrees(y/6371.0088)
                def lon(x):
                    return lon0+math.degrees(x/(6371.0088*math.cos(math.radians(lat0))))
                bbox = (lat(ymin-yoffset/scale), lon(xmin-xoffset/scale),
                        lat(ymax+yoffset/scale), lon(xmax+xoffset/scale))
                background = paint_view(self._basemap_raster, self._basemap_origin, bbox, (width, height))
            painter = ImageDraw.Draw(background)
            drawn_roads, labeled_roads = set(), set()
            preferred_raya = None
            if focus:
                cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
                raya = [e for e in scenario.road_edges if "ciledug raya" in e.name.lower()]
                if raya:
                    preferred_raya = min(raya, key=lambda e: (road_lookup[e.source].x-cx)**2+(road_lookup[e.source].y-cy)**2)

            for edge in scenario.road_edges:
                a, b = road_lookup[edge.source], road_lookup[edge.target]
                arterial = edge.road_class.removesuffix("_link") in ("motorway", "trunk", "primary", "secondary") if edge.road_class else edge.speed_kph >= 40
                collector = edge.road_class.removesuffix("_link") == "tertiary" if edge.road_class else edge.speed_kph >= 30
                road_color = "#d7be91" if arterial else "#bfc7ce" if collector else "#d7dce0"
                road_width = 4 if arterial else 2.5 if collector else 1.5
                drawing_key = (tuple(sorted((edge.source, edge.target))), edge.road_class, edge.name)
                if drawing_key not in drawn_roads:
                    if self._basemap_raster is None or scenario.objective_metric != "facility_distance" or self.full_network_visible.get():
                        painter.line((xy(a), xy(b)), fill=road_color, width=round(road_width))
                    drawn_roads.add(drawing_key)
                if "ciledug raya" in edge.name.lower() and edge.name not in labeled_roads and (preferred_raya is None or edge == preferred_raya):
                    x, y = xy(a)
                    canvas.create_text(x+6, y-10, text=edge.name, anchor="w", fill="#93672f", font=("Segoe UI", 9, "bold"), tags="osm_street_label")
                    labeled_roads.add(edge.name)
                directions = ((edge.source, edge.target),) if edge.one_way else ((edge.source, edge.target), (edge.target, edge.source))
                for pair in directions:
                    road_speeds[pair] = max(road_speeds.get(pair, 0), edge.speed_kph)
            self._map_road_tables = road_lookup, road_speeds
            self._map_image = ImageTk.PhotoImage(background)
            canvas.create_image(0, 0, anchor="nw", image=self._map_image, tags="map_background")
            canvas.tag_lower("map_background")
        if self.radius_visible.get():
            for j, kitchen in enumerate(scenario.kitchens):
                x, y = xy(kitchen)
                r = scenario.radius * scale
                canvas.create_oval(x-r, y-r, x+r, y+r, outline=COLORS[j % len(COLORS)],
                                   dash=(4, 5), tags="map_foreground")
        assignment = self.current_snapshot.best.assignment if self.current_snapshot else None
        if assignment and scenario.routing_mode == "multi_stop":
            for tour in vehicle_tours(scenario, self.current_snapshot.best):
                j = tour["kitchen_index"]
                segments = []
                for leg in tour["legs"]:
                    path = leg["path"]
                    coordinates = [value for key in path for value in xy(road_lookup[key])]
                    if len(coordinates) >= 4:
                        canvas.create_line(*coordinates, fill=COLORS[j % len(COLORS)], width=2, arrow="last", tags=("delivery_path", "map_foreground"))
                    elapsed = leg["departure_minutes"]
                    for a, b in zip(path, path[1:]):
                        first, last = road_lookup[a], road_lookup[b]
                        duration = math.hypot(first.x-last.x, first.y-last.y)*60/road_speeds[a, b]
                        segments.append((elapsed, duration, xy(first), xy(last)))
                        elapsed += duration
                    if leg["unloading_minutes"]:
                        point = xy(road_lookup[path[-1]])
                        segments.append((elapsed, leg["unloading_minutes"], point, point))
                if self.delivery_visible.get():
                    marker = canvas.create_oval(-4, -4, 4, 4, fill=COLORS[j % len(COLORS)], outline=INK, tags=("delivery_vehicle", "map_foreground"))
                    self._delivery_routes.append((marker, segments, tour["duration_minutes"]))
        elif assignment:
            for i, j in enumerate(assignment):
                path = scenario.paths[i][j] if scenario.paths else ()
                route = [road_lookup[key] for key in path] if path else [scenario.kitchens[j], scenario.schools[i]]
                coordinates = [value for point in route for value in xy(point)]
                canvas.create_line(*coordinates, fill=COLORS[j % len(COLORS)],
                                   width=3 if changed and i in changed else 1, tags=("delivery_path", "map_foreground"))
                if path and self.delivery_visible.get():
                    segments = []
                    elapsed = 0.
                    for a, b in zip(path, path[1:]):
                        first, last = road_lookup[a], road_lookup[b]
                        duration = math.hypot(first.x-last.x, first.y-last.y) * 60 / road_speeds[a, b]
                        segments.append((elapsed, duration, xy(first), xy(last)))
                        elapsed += duration
                    marker = canvas.create_oval(-4, -4, 4, 4, fill=COLORS[j % len(COLORS)], outline=INK, tags=("delivery_vehicle", "map_foreground"))
                    self._delivery_routes.append((marker, segments, elapsed))
        stop_numbers = {}
        if self.current_snapshot and self.current_snapshot.best.visit_order:
            counts = {}
            for i in self.current_snapshot.best.visit_order:
                j = assignment[i]
                counts[j] = counts.get(j, 0)+1
                stop_numbers[i] = counts[j]
        for i, school in enumerate(scenario.schools):
            x, y = xy(school)
            color = COLORS[assignment[i] % len(COLORS)] if assignment else "#a0adba"
            r = 4 + 2 * school.demand / 300
            if changed and i in changed:
                canvas.create_oval(x - r - 4, y - r - 4, x + r + 4, y + r + 4, outline="#19304a", width=2, tags="map_foreground")
            marker = canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=1, tags=("school_marker", "map_foreground"))
            arrival = self.current_snapshot.best.school_arrivals[i] if self.current_snapshot and self.current_snapshot.best.school_arrivals else None
            self._school_markers.append((marker, arrival))
            source = next((record for record in scenario.road_source.get("school_features", []) if record["school_id"] == school.id), {})
            message = f"{school.id}: {source.get('name', 'Mapped school')} | demand {school.demand} | arrival {arrival if arrival is not None else 'pending'} min"
            canvas.tag_bind(marker, "<Button-1>", lambda event, text=message: self.status_var.set(text))
            canvas.tag_bind(marker, "<Enter>", lambda event, text=message: self.status_var.set(text))
            label = source.get("name", school.id) if self.school_names_visible.get() else school.id
            if self.current_snapshot and self.current_snapshot.best.visit_order:
                label += f" #{stop_numbers[i]}"
            canvas.create_text(x + 7, y - 9, text=label, fill=MUTED, anchor="w", font=("Segoe UI", 8), tags=("school_label", "map_foreground"))
        for j, kitchen in enumerate(scenario.kitchens):
            x, y = xy(kitchen)
            active = assignment is not None and j in assignment
            canvas.create_rectangle(x - 8, y - 8, x + 8, y + 8, fill=COLORS[j % len(COLORS)] if active else "white", outline=INK if active else MUTED, width=2, tags=("map_foreground", "selected_sppg" if active else "unused_sppg"))
            label = f"{kitchen.id}: {assignment.count(j)} schools" if active else f"{kitchen.id} off"
            canvas.create_text(x, y + 19, text=label, fill=INK if active else MUTED, font=("Segoe UI", 10, "bold"), tags="map_foreground")
        legend = "Synthetic schools | Filled: active SPPG | Hollow: unused | Lines: separate one-way shortest road paths" if scenario.objective_metric == "facility_distance" else "Filled: active SPPG | Hollow: unused | # stop order | Green: delivered | Lines: returning tours"
        canvas.create_text(12, 12, anchor="nw", text=legend, fill=MUTED, font=("Segoe UI", 9), tags="map_foreground")
        # A true map scale in kilometres.
        scale_km = max(0.5, round(65 / scale * 2) / 2)
        bar = scale_km * scale
        canvas.create_line(15, height - 20, 15 + bar, height - 20, fill=INK, width=2, tags="map_foreground")
        canvas.create_text(15, height - 25, anchor="sw", text=f"{scale_km:g} km", fill=MUTED, font=("Segoe UI", 8), tags="map_foreground")
        if scenario.road_source.get("source") == "OpenStreetMap" and not cached:
            canvas.create_text(width-12, height-32, anchor="se", text="© OpenStreetMap contributors · " + scenario.road_source["city"],
                               fill=MUTED, font=("Segoe UI", 9), tags="osm_attribution")
            def attribution(event):
                import webbrowser
                webbrowser.open("https://www.openstreetmap.org/copyright")
            canvas.tag_bind("osm_attribution", "<Button-1>", attribution)

        if self._delivery_routes:
            canvas.create_text(width-12, height-12, anchor="se", fill=INK, tags=("delivery_clock", "map_foreground"))
            self._animate_deliveries()

    def _animate_deliveries(self):
        if self.closed or not self._delivery_routes:
            return
        maximum = max(route[2] for route in self._delivery_routes)
        now = time.monotonic()
        paused = bool((self.control and self.busy and self.control.paused) or (self.replay_frames and not self.replay_playing))
        if not paused:
            self._delivery_clock_minutes = (self._delivery_clock_minutes+(now-self._delivery_last_tick)*maximum/12) % max(.001, maximum)
        self._delivery_last_tick = now
        minutes = self._delivery_clock_minutes
        for marker, arrival in self._school_markers:
            delivered = arrival is not None and minutes >= arrival
            self.map_canvas.itemconfigure(marker, outline="#15803d" if delivered else "white", width=3 if delivered else 1)
        for marker, segments, eta in self._delivery_routes:
            if not segments:
                continue
            first, last = segments[-1][2:]
            for start, duration, a, b in segments:
                if minutes <= start + duration:
                    fraction = min(1., max(0., (minutes-start) / duration)) if duration else 1.
                    first = (a[0] + fraction*(b[0]-a[0]), a[1] + fraction*(b[1]-a[1]))
                    last = first
                    break
            x, y = last
            self.map_canvas.coords(marker, x-3, y-3, x+3, y+3)
        self.map_canvas.itemconfigure("delivery_clock", text=f"Delivery simulation: {minutes:.1f} min / {maximum:.1f} min (compressed)")
        self._delivery_after_id = self.after(80, self._animate_deliveries)

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
            count = self.current_snapshot.best.assignment.count(j) if self.current_snapshot else 0
            canvas.create_text(width - 7, y + row_height / 2, anchor="e", text=f"{count} schools · {load:,}/{kitchen.capacity:,}", fill=INK, font=("Segoe UI", 8))

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
            value = low + fraction * (high - low)
            tick = f"{value:.2f}" if high < 10 else f"{value:,.0f}"
            canvas.create_text(left - 6, y, anchor="e", text=tick, fill=MUTED, font=("Segoe UI", 8))
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

        planning = bool(self.benchmark.scenarios and self.benchmark.scenarios[0].objective_metric == "facility_distance")
        self.summary_table.heading("eta", text="Mean SPPGs" if planning else "ETA min")
        self.summary_table.heading("max_eta", text="Distance km" if planning else "Max ETA")
        for row in self.benchmark.summaries():
            values = (row["scenario"], "PSO/discrete" if row["algorithm"] == "PSO" else row["algorithm"],
                      row["instance_count"], f"{row['feasible_runs']}/{row['runs']}", fmt(row["mean_optimality_gap_percent"], 3),
                      fmt(row["best_objective"]), fmt(row["mean_objective"]), fmt(row["sample_std"]),
                      fmt(row["mean_solver_seconds"], 3), fmt(row["baseline_objective"]),
                      fmt(row["mean_active_sppg_count"] if planning else row["mean_weighted_average_minutes"], 2),
                      fmt(row["mean_total_allocation_distance_km"] if planning else row["mean_max_school_eta_minutes"], 2))
            self.summary_table.insert("", "end", values=values)
        self.replay_lookup = {f"{run.scenario_name} · scenario {run.scenario_seed} · {ALGORITHM_LABELS[run.algorithm]} · optimizer {run.seed} · {run.settings.constraint_policy}"
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
                instances = {}
                for run in self.benchmark.runs:
                    if run.scenario_name == self.compare_size_var.get() and run.algorithm == algorithm and run.completed:
                        for entry in run.history:
                            instances.setdefault(run.scenario_id, {}).setdefault(entry["evaluations"], []).append(entry.get({"Objective": "objective", "Arrival objective": "objective", "Certified gap": "optimality_gap_percent", "Gap to MILP bound": "lower_bound_gap_percent"}[self.compare_metric_var.get()]))
                checkpoints = {}
                for entries in instances.values():
                    for x, ys in entries.items():
                        valid = [y for y in ys if y is not None]
                        if valid:
                            checkpoints.setdefault(x, []).append(sum(valid) / len(valid))
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
        self.search_notebook.select(self.search_view if run.algorithm in ("PSO", "ACO") else self.road_tab)
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
        self.status_var.set(f"Exported JSON, run/summary CSVs, instance_summary.csv and diagnostics.csv to {directory}")

    def close(self):
        self.closed = True
        self.search_view.cancel()
        if self.control:
            self.control.cancel()
        self._stop_replay()
        self.after_cancel(self._poll_id)
        if self._delivery_after_id is not None:
            self.after_cancel(self._delivery_after_id)
        self.destroy()
