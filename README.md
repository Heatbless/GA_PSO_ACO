# SPPG assignment lab

A Python/Tkinter coursework application comparing Genetic Algorithm (GA), Particle
Swarm Optimization (PSO), and Ant Colony Optimization (ACO) on assigning schools
to fixed SPPG kitchens. It includes animated search, repeated benchmarks, replay,
and reproducible JSON/CSV exports.

## Run

Python 3.10 or newer with Tkinter is required. No third-party packages are needed.
The implementation was tested on Windows with Python 3.14 and Tk 9.

From this directory:

```powershell
python -m sppg_compare
```

Alternatively, double-click `run_gui.bat` on Windows. If Python is not found, install
Python with its Tcl/Tk component and add it to PATH. `python -m tkinter` opens the
standard Tkinter installation check.

For the full 90-run benchmark without opening the GUI:

```powershell
python -m sppg_compare --benchmark --output results/full_benchmark
```

For a shorter experiment:

```powershell
python -m sppg_compare --benchmark --sizes Small --seeds 2 --budget 300 --output results/quick
```

Use `python -m sppg_compare --help` for CLI options. CLI solver parameters use the
defaults below; the GUI exposes algorithm-specific parameters.

## Using the interface

1. Choose a scenario size, dataset seed, and service radius. Click **Generate scenario**.
2. Choose GA, PSO, or ACO, set the optimizer seed, evaluation budget, and parameters,
   then click **Run selected**.
3. Watch the best assignment, kitchen loads, and best feasible objective. Squares
   represent kitchens; circles represent schools, with size indicating demand.
   New assignment links are emphasized when the incumbent changes.
4. **Pause** stops at the next generation/iteration boundary. **Step** advances one
   batch while paused. Playback speed controls update frequency; it does not change
   the search result or measured solver time.
5. Run other algorithms on the same scenario, then switch the selector to review
   their saved demo results. Changing numeric settings clears previous results.
6. Click **Compare all · 90 runs** for the complete benchmark. The Comparison tab
   shows progress, summaries, and mean feasible convergence by scenario size.
7. Choose a completed run in that tab and click **Replay selected run** to animate
   its recorded history. Pause, step, and speed also work during replay.
8. **Export results** writes JSON and two CSV files into the chosen directory.
   If a benchmark exists, it is exported; otherwise the saved demo runs are exported.

**Reset / cancel** cancels current work and clears results. Closing the window also
cancels its worker. Scenario and algorithm inputs are locked while work is active.
Changing the radius and regenerating scales the synthetic geography with it;
benchmark scenarios always use the three dataset seeds listed below, irrespective
of the current demo dataset seed. The controls scroll vertically on smaller windows.

## SPPG model

Each kitchen has a fixed `(x, y)` location and integer daily capacity. Each school
has a fixed location and integer daily meal demand. A school must receive its full
demand from exactly one kitchen. School-to-kitchen assignments must fit both daily
capacity and service radius.

The objective is `J = sum(school demand × assigned kitchen distance)` in
portion-kilometres. Lower is better. The GUI also displays `J / total demand`, the
demand-weighted mean distance in kilometres. These are assignment metrics, not
vehicle-tour length or monetary delivery cost.

The default radius is 6 km, motivated by [BGN's explanation of SPPG service
coverage](https://www.bgn.go.id/news/siaran-pers/bgn-gunakan-data-dan-jarak-efektif-cegah-makanan-mbg-basi).
Demo capacities are generated inputs, not regulatory capacity limits. Coordinates
are synthetic and distances are straight-line Euclidean distances. The model is
for coursework; its output does not establish actual road access or travel time.

| Size | Kitchens | Schools | Dataset seed |
| --- | ---: | ---: | ---: |
| Small | 3 | 15 | 101 |
| Medium | 5 | 40 | 202 |
| Large | 8 | 80 | 303 |

Demand ranges from 80 to 300 portions. The generator privately constructs a feasible
assignment and sets each kitchen's capacity to its constructed load plus 5%, rounded
up. It accepts instances with overlapping service areas and an overloaded
unconstrained nearest-kitchen assignment. It retries at most 100 times.
The feasibility witness is never passed to the optimizers or exported.

The `scenarios/` directory contains the three default generated instances. They use
the same dictionary format as `Scenario.to_dict()` and can be reconstructed with
`Scenario.from_dict()`. The GUI generates synthetic scenarios; it does not provide
a real-data import workflow.

## Search methods and constraints

All methods ultimately produce a tuple of kitchen indices, one per school. Their
generation/decoding only allows kitchens inside each school's radius. Before
scoring, the shared repair routine moves schools out of overloaded kitchens to
eligible destinations with sufficient spare capacity. It chooses the relocation
with the smallest increase in objective, with deterministic index tie-breaking.
Repair stops when no relocation is available; it does not guarantee feasibility.

Candidates are ranked by `(radius-violating demand, capacity overflow, objective)`.
For solver-generated candidates, radius-violating demand is always zero. Thus any
feasible candidate beats an infeasible candidate, and lower overflow is preferred
when no feasible candidate has been found. Infeasible runs do not contribute to
feasible-objective statistics or convergence points.

| Method | Representation | Default configuration |
| --- | --- | --- |
| GA | Integer school assignments | 30 individuals; tournament size 3; uniform crossover probability 0.9; mutation probability `1 / school_count` per gene; 2 elites |
| PSO | Continuous school–kitchen preference scores decoded by maximum eligible score | 30 particles; inertia 0.7; cognitive/social coefficients 1.5; score bounds `[0, 1]`; velocity bounds `[-0.2, 0.2]` |
| ACO | School–kitchen pheromone matrix; constructive assignment in descending school demand order | 30 ants; alpha 1; beta 2; evaporation 0.2; initial pheromone 1; pheromone bounds `[0.01, 1]` |

GA mutation chooses another eligible kitchen when available. PSO uses a synchronous
global-best update and scores decoded assignments after common repair; personal
best positions remain the preference vectors that produced those scores. This is
a preference-score adaptation of continuous PSO to a discrete assignment problem.

ACO samples eligible kitchens using `pheromone^alpha × (1 / (1 + distance))^beta`.
It prefers choices with enough residual capacity. If no kitchen fits, it makes an
eligible assignment and leaves overload handling to shared repair. Every batch
evaporates pheromone and deposits on the best feasible assignment with quality
`1 / (1 + demand-weighted mean distance)`.

The greedy baseline visits schools in descending demand order and chooses the
nearest eligible kitchen with spare capacity. If none fits, it chooses the nearest
eligible kitchen and then uses the same repair. It is a reference heuristic, not
an optimality certificate.

## Benchmark and exports

The default comparison runs 3 scenario sizes × 3 algorithms × optimizer seeds 1–10:
**90 runs, with 5,000 evaluated candidates each**. Each size uses one fixed dataset.
Algorithm order rotates across seeds: GA/PSO/ACO, PSO/ACO/GA, ACO/GA/PSO.

Initial candidates and repeated candidates count toward the budget. Stored GA
elites are reused without another evaluation. Final batches may be truncated to
finish exactly at the budget. The equal budget controls scoring opportunities;
the methods can still perform different amounts of work per evaluation.

Solver runtime includes batch construction, decoding, shared repair, scoring, and
algorithm updates. It excludes pauses, animation delays, GUI rendering, and export.
Timing is machine-dependent; identical seeds reproduce assignments and objective
histories, not wall-clock durations.

Summaries include feasible-run counts and percentages, best and mean feasible
objective, sample standard deviation, mean solver runtime, and greedy baseline.
Standard deviation is unavailable for fewer than two feasible runs. Comparison
curves average only runs that are feasible at a given checkpoint; the contributing
set can change. A changing set can make the mean curve non-monotonic even though
each run's incumbent improves monotonically.

Exports:

- `benchmark.json`: schema/application version, environment, settings, scenarios,
  baselines, complete or partial runs, assignments, loads, checkpoint histories,
  animation snapshots, and summary statistics.
- `runs.csv`: one row per run; infeasible objectives are blank, with overflow and
  radius violations retained. Cancelled runs are labelled incomplete.
- `summary.csv`: feasible-only statistics for completed runs, grouped by size and
  algorithm. Partial runs are excluded from summaries.

Results support comparison of these defaults on these instances. They do not
demonstrate that one algorithm is universally better, nor isolate each algorithm
from the effect of the shared repair operator.

## Code and tests

`model.py` owns the scenario, evaluator, generator, and repair. `algorithms.py`
provides the common budgeted `Optimizer.advance()` interface and immutable progress
snapshots. `experiments.py` runs sequential experiments and exports their results.
`gui.py` consumes those interfaces without changing the solver logic.

Workers operate on Python objects and send messages through a queue. Only the main
thread creates or updates Tk widgets; `after()` polls for messages. Each new run has
a token so stale updates after reset are discarded. See [Tkinter's threading
model](https://docs.python.org/3/library/tkinter.html#threading-model).

Run core tests:

```powershell
python -m unittest tests.test_core -v
```

Run GUI smoke tests on a desktop with Tk available:

```powershell
python -m unittest tests.test_gui -v
```

The GUI suite briefly opens a window during its resize check. It exercises all
three algorithms, pause/step, reset and stale-message handling, a reduced-budget
90-run benchmark, replay, invalidation, and validation errors. Core tests cover
hand-calculated scores, exhaustive tiny-instance checks, radius and capacity
constraints, exact budgets, reproducibility, cancellation, and exports.
