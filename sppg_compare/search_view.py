"""Tk-only replay of recorded optimizer state; never drives the optimizer."""
import time
import tkinter as tk
from tkinter import ttk

COLORS = ("#2878b5", "#d96c22", "#2b9973", "#a05daf", "#ce5265", "#a58b20", "#4678a6", "#7873ba")
INK, MUTED = "#19304a", "#61758b"


class SearchView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=5)
        ttk.Label(toolbar, text="Inspect school:").pack(side="left")
        self.school_var = tk.StringVar()
        self.school_box = ttk.Combobox(toolbar, textvariable=self.school_var, state="readonly", width=12)
        self.school_box.pack(side="left", padx=8)
        self.school_box.bind("<<ComboboxSelected>>", lambda e: self.draw())
        self.matrix_mode = tk.StringVar(value="SPPG allocation")
        mode = ttk.Combobox(toolbar, textvariable=self.matrix_mode, values=("SPPG allocation", "Visit order"), state="readonly", width=16)
        mode.pack(side="left", padx=8)
        mode.bind("<<ComboboxSelected>>", lambda e: self.draw())
        ttk.Label(toolbar, text="Recorded search state · delivery routes are in the Road map tab").pack(side="left")
        self.canvas = tk.Canvas(self, bg="#fbfcfe", highlightthickness=0, height=340)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.snapshot = self.scenario = None
        self.previous = {}
        self.timer = None
        self.phase = 1.
        self.started = 0.
        self.duration = .8
        self.paused = False

    def cancel(self):
        if self.timer is not None:
            self.after_cancel(self.timer)
            self.timer = None

    def clear(self):
        self.cancel()
        self.snapshot = self.scenario = None
        self.previous = {}
        self.draw()

    def show(self, scenario, snapshot, duration=.8, animate=True):
        self.cancel()
        same_run = (self.scenario == scenario and self.snapshot is not None
                    and self.snapshot.algorithm == snapshot.algorithm and self.snapshot.seed == snapshot.seed
                    and snapshot.iteration > self.snapshot.iteration)
        self.previous = self.snapshot.search_state if same_run else {}
        self.scenario, self.snapshot = scenario, snapshot
        self.school_box.configure(values=tuple(s.id for s in scenario.schools))
        if self.school_var.get() not in tuple(s.id for s in scenario.schools):
            self.school_var.set(scenario.schools[0].id)
        self.duration = max(.12, min(2., duration))
        self.started = time.monotonic()
        self.paused = not animate
        self.phase = 0. if animate else 1.
        self.tick()

    def set_paused(self, paused):
        self.cancel()
        self.paused = paused
        if not paused and self.phase < 1:
            self.started = time.monotonic() - self.phase * self.duration
            self.tick()

    def tick(self):
        self.timer = None
        if not self.paused:
            self.phase = min(1., (time.monotonic()-self.started) / self.duration)
        self.draw()
        if not self.paused and self.phase < 1:
            self.timer = self.after(25, self.tick)

    def text(self, x, y, text, **options):
        font = options.pop("font", ("Segoe UI", 9))
        self.canvas.create_text(x, y, text=text, fill=INK, anchor="nw", font=font, **options)

    def draw(self):
        c = self.canvas
        c.delete("all")
        if self.snapshot is None:
            self.text(18, 18, "Run PSO or ACO to inspect how the optimizer searches.")
            return
        state = self.snapshot.search_state
        if not state:
            self.text(18, 18, "No recorded search state for this frame. Run a new PSO/ACO demo or GUI comparison.")
            return
        width, height = max(400, c.winfo_width()), max(180, c.winfo_height())
        split = width * .49
        focus = next((i for i, s in enumerate(self.scenario.schools) if s.id == self.school_var.get()), 0)
        if state["kind"] == "PSO":
            self.draw_swarm(split, height, state)
        else:
            self.draw_ants(split, height, state, focus)
        if self.matrix_mode.get() == "Visit order" and self.scenario.routing_mode == "multi_stop":
            self.draw_route_matrix(split+18, width-16, height, state, focus)
        else:
            self.draw_matrix(split+18, width-16, height, state, focus)

    def draw_route_matrix(self, left, right, height, state, focus):
        """Show the actual route-order state rather than geographic particles."""
        c, schools = self.canvas, self.scenario.schools
        order = self.snapshot.best.visit_order
        j = self.snapshot.best.assignment[focus]
        tour = [i for i in order if self.snapshot.best.assignment[i] == j]
        sequence = " -> ".join(schools[i].id for i in tour)
        self.text(left, height-65, f"Best {self.scenario.kitchens[j].id} tour: {sequence} -> SPPG", width=right-left)
        self.text(left, height-30, f"{schools[focus].id}: stop {tour.index(focus)+1}, arrival {self.snapshot.best.school_arrivals[focus]:.2f} min", width=right-left)
        count = min(10, len(schools))
        start = max(0, min(focus-count//2, len(schools)-count))
        visible = list(range(start, start+count))
        if state["kind"] == "PSO":
            self.text(left, 8, "Global-best route priority keys (smaller is earlier)")
            keys = state.get("route_keys", ())
            for row, i in enumerate(visible):
                y = 45+row*max(18, (height-130)/count)
                self.text(left, y, schools[i].id)
                x = left+45
                c.create_rectangle(x, y, right, y+14, fill="#e5ebf1", outline="")
                if keys:
                    c.create_rectangle(x, y, x+(right-x)*keys[i], y+14, fill="#2878b5", outline="", tags="route_key")
        else:
            self.text(left, 8, "Route pheromone: previous stop -> next school")
            matrix = state.get("route_pheromone", ())
            if not matrix:
                return
            rows = [len(schools)]+visible
            cell_w = (right-left-55)/count
            cell_h = max(14, (height-130)/len(rows))
            for column, i in enumerate(visible):
                self.text(left+55+column*cell_w, 30, schools[i].id, font=("Segoe UI", 8))
            for row, previous in enumerate(rows):
                y = 50+row*cell_h
                self.text(left, y, "SPPG" if previous == len(schools) else schools[previous].id)
                for column, i in enumerate(visible):
                    value = matrix[previous][i]
                    x = left+55+column*cell_w
                    color = f"#{int(235-200*value):02x}{int(244-130*value):02x}{int(252-60*value):02x}"
                    c.create_rectangle(x, y, x+cell_w-2, y+cell_h-2, fill=color, outline="white", tags="route_pheromone")

    def draw_swarm(self, width, height, state):
        c = self.canvas
        self.text(16, 8, f"PSO preference projection · showing {len(state['particles'])}/{state['population']} particles", width=width-25)
        left, right, top, bottom = 48, width-24, 42, height-67
        def point(pair):
            return left + pair[0]*(right-left), bottom-pair[1]*(bottom-top)
        for fraction in (0., .5, 1.):
            x, y = point((fraction, fraction))
            c.create_line(x, top, x, bottom, fill="#e4eaf0")
            c.create_line(left, y, right, y, fill="#e4eaf0")
            self.text(x-8, bottom+3, f"{fraction:g}")
            self.text(20, y-6, f"{fraction:g}")
        old = {p["id"]: p for p in self.previous.get("particles", ())}
        smooth = self.phase*self.phase*(3-2*self.phase)
        for p in state["particles"]:
            color = COLORS[p["id"] % len(COLORS)]
            start = old.get(p["id"], p)["position"]
            current = tuple(a+(b-a)*smooth for a, b in zip(start, p["position"]))
            x, y = point(current)
            px, py = point(p["personal_best"])
            sx, sy = point(start)
            c.create_line(sx, sy, x, y, fill=color, width=2, tags="particle_trail")
            c.create_line(x, y, px, py, fill="#b2bcc7", dash=(2, 3))
            c.create_oval(px-4, py-4, px+4, py+4, outline=color, tags="personal_best")
            vx, vy = point(tuple(max(0., min(1., a+v)) for a, v in zip(current, p["velocity"])))
            c.create_line(x, y, vx, vy, fill=color, arrow="last", width=2, tags="velocity")
            c.create_oval(x-5, y-5, x+5, y+5, fill=color, outline="white", tags="particle")
            self.text(x+6, y-13, f"P{p['id']+1}")
        x, y = point(state["global_best"])
        c.create_polygon(x, y-9, x+9, y, x, y+9, x-9, y, fill="#f2bd49", outline=INK, tags="global_best")
        self.text(15, height-46, "X/Y: mean first/second eligible preference (0–1); projection, not map locations", width=width-25, font=("Segoe UI", 8))
        self.text(15, height-22, "● current   ○ personal best   ◆ global best   arrow: projected velocity", width=width-25, font=("Segoe UI", 8))

    def draw_ants(self, width, height, state, focus):
        c = self.canvas
        trace = state["trace"]
        count = min(len(trace), int(self.phase*len(trace)))
        self.text(15, 8, f"ACO · first evaluated ant · {count}/{len(trace)} school choices")
        self.text(15, 28, f"Evaporation: {100*state['evaporation']:.0f}% per batch · schematic allocation")
        # Show a window around the current construction step, preserving demand order.
        active = min(max(0, count-1), len(trace)-1)
        visible = min(10, max(3, (height-135)//20))
        start = max(0, min(active-visible//2, len(trace)-visible))
        rows = trace[start:start+visible]
        top, bottom = 65, height-70
        row_height = (bottom-top) / max(1, len(rows))
        kitchen_x = width-70
        kitchen_y = {j: top+(j+.5)*(bottom-top)/len(self.scenario.kitchens)
                     for j in range(len(self.scenario.kitchens))}
        for j, kitchen in enumerate(self.scenario.kitchens):
            y = kitchen_y[j]
            c.create_rectangle(kitchen_x-7, y-7, kitchen_x+7, y+7, fill=COLORS[j % len(COLORS)])
            self.text(kitchen_x+12, y-8, kitchen.id)
        for offset, step in enumerate(rows):
            index = start+offset
            i, j = step["school"], step["kitchen"]
            y = top+(offset+.5)*row_height
            self.text(15, y-8, self.scenario.schools[i].id)
            c.create_oval(66, y-4, 74, y+4, fill="#aebac7")
            if index < count:
                color = COLORS[j % len(COLORS)]
                c.create_line(74, y, kitchen_x-8, kitchen_y[j], fill=color, width=3 if index == active else 1, arrow="last", tags="ant_choice")
                if index == active:
                    c.create_oval(68, y-6, 80, y+6, fill=color, outline=INK, tags="ant_marker")
        chosen = trace[active] if trace else None
        if chosen:
            probability = chosen["probabilities"][chosen["choices"].index(chosen["kitchen"])]
            school = self.scenario.schools[chosen["school"]].id
            message = f"{school}: sampled kitchen probability {100*probability:.1f}% (at construction)" if count else "Ready to replay the recorded construction choices."
            self.text(15, height-54, message, width=width-25, font=("Segoe UI", 8))
        changed = sum(a != b for a, b in zip(state["raw_assignment"], state["repaired_assignment"]))
        self.text(15, height-30, f"Repair changed {changed} schools for this ant. Lines show raw choices.")

    def draw_matrix(self, left, right, height, state, focus):
        c = self.canvas
        matrix = state["matrix"]
        name = "Mean swarm preference" if state["kind"] == "PSO" else "Pheromone after evaporation + deposit"
        self.text(left, 8, name)
        count = min(12, len(matrix), max(3, (height-114)//18))
        start = max(0, min(focus-count//2, len(matrix)-count))
        top, bottom = 52, height-62
        cell_h = (bottom-top)/count
        cell_w = (right-left-45)/len(self.scenario.kitchens)
        maximum = max(matrix[i][j] for i in range(len(matrix)) for j in self.scenario.eligible[i])
        for j, kitchen in enumerate(self.scenario.kitchens):
            self.text(left+45+j*cell_w, 30, kitchen.id)
        reveal = int(self.phase*len(state.get("trace", ())))
        choices = {s["school"]: s["kitchen"] for s in state.get("trace", ())[:reveal]}
        for row, i in enumerate(range(start, start+count)):
            y = top+row*cell_h
            self.text(left, y+2, self.scenario.schools[i].id)
            for j in range(len(self.scenario.kitchens)):
                x = left+45+j*cell_w
                eligible = j in self.scenario.eligible[i]
                old_matrix = self.previous.get("matrix")
                previous = old_matrix[i][j] if old_matrix else (1. if state["kind"] == "ACO" else matrix[i][j])
                value = previous + (matrix[i][j]-previous)*self.phase
                scale = max(maximum, max(max(row) for row in old_matrix)) if old_matrix else max(maximum, 1.)
                strength = value / max(scale, 1e-12)
                color = f"#{int(235-200*strength):02x}{int(244-130*strength):02x}{int(252-60*strength):02x}" if eligible else "#e4e7eb"
                c.create_rectangle(x, y, x+cell_w-2, y+cell_h-2, fill=color,
                                   outline="#f2bd49" if i == focus else "white", width=2 if i == focus else 1, tags="search_cell")
                if cell_h >= 17 and cell_w >= 34 and eligible:
                    c.create_text(x+cell_w/2, y+cell_h/2, text=f"{value:.2f}", fill="white" if strength > .65 else INK, font=("Segoe UI", 8))
                if choices.get(i) == j:
                    c.create_oval(x+3, y+3, x+10, y+10, fill="#f2bd49", outline=INK, tags="ant_cell")
        self.text(left, height-48, "Darker = stronger; gray = ineligible. Gold outline = inspected school.", width=right-left)
        i = focus
        if state["kind"] == "PSO":
            changed = sum(p["decoded"][i] != p["phenotype"][i] for p in state["particles"])
            self.text(left, height-19, f"Sampled particles repaired at {self.scenario.schools[i].id}: {changed}/{len(state['particles'])}")
        else:
            raw, repaired = state["raw_assignment"][i], state["repaired_assignment"][i]
            self.text(left, height-19, f"{self.scenario.schools[i].id}: raw {self.scenario.kitchens[raw].id} → repaired {self.scenario.kitchens[repaired].id}")
