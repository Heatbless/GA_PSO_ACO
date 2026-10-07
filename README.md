# Ciledug / Pedurenan SPPG comparison

Tkinter comparison of GA, preference-score/discrete PSO, and ACO using real
OpenStreetMap roads and fixed synthetic school locations. The default view is
[the supplied Pedurenan map](https://www.openstreetmap.org/#map=16/-6.22253/106.69706).
The wider Ciledug graph, including Ciledug Raya, remains available for routing.

## Launch

Python 3.10+ with Tkinter, SciPy and Pillow:

    python -m pip install -r requirements.txt
    python -m sppg_compare

On Windows, double-click `run_gui.bat`. Road data is imported by a background
worker and cached under `data/osm_cache/`. The OSM raster loads asynchronously
for the displayed view; **Load OSM background** retries it. Without tiles, the
same real road graph is rendered into a single background image. No road network
is fabricated. Attribution stays visible. Tile requests identify the application,
cache for at least seven days, and cover only this displayed viewport. Set
`SPPG_TILE_URL` to use another compatible raster tile provider.

## Planning model

The default **Synthetic planning** model optimizes which candidate SPPGs to open
and which schools each serves. Every school must be allocated to exactly one
SPPG. School demand is indivisible; capacity and service radius are constraints.

The menu exposes:

| Input | Default | Meaning |
| --- | ---: | --- |
| School count | 30 | Synthetic schools generated in the displayed area |
| School location seed | 101 | Fixed school locations, independent of optimizer randomness |
| Meals / school | 200 | Synthetic daily demand per school |
| SPPG capacity | 3,000 | Daily meals available at each candidate |
| SPPG opening weight | 10 | Penalty for each active SPPG |
| Distance weight | 1 | Penalty per kilometre of allocation distance |
| Candidate pool | Small | Small: 6, Medium: 10, Large: 16 candidates |
| Candidate-site seed | 101 | Resamples candidate SPPG locations |

School locations are deterministic road nodes, explicitly synthetic rather than
real school records. Increasing the count preserves the existing prefix of
school locations. Schools are identical across algorithms, candidate-site seeds
and pool sizes when the school count and school seed match. Candidates are
sampled separately and exclude school nodes. Geography is independent of radius.
Capacity is fixed rather than scaled automatically with school count.

The objective is:

    J = opening_weight * active_SPPG_count
        + distance_weight * sum(one_way_shortest_road_distance_to_each_school)

Distances follow the directed OSM road graph, including one-way restrictions.
They are shortest **distance** paths, rather than fastest paths. Demand controls
capacity; it does not multiply distance in this objective. The score combines
user-defined weights, so it is not a physical distance or an operating cost.

This is a capacitated facility-location/allocation comparison. The distance is
an allocation measure, not total vehicle mileage: it does not include return
trips, shared multi-stop routes, fleet scheduling, traffic, or tray collection.
Animation illustrates the separate outbound paths and does not assert that one
vehicle per school exists.

With the defaults, one SPPG can serve at most 15 schools. Total demand gives a
capacity lower bound of `ceil(total demand / SPPG capacity)` active SPPGs. This
is not a feasibility guarantee: radius coverage and indivisible demands matter.
**The weighted optimum may open more SPPGs than that lower bound** when reduced
distance offsets the opening penalty. Thus the reported SPPG count is the number
selected under the chosen weights, not a proof of the minimum possible count.

## Map, results and animation

The background is one raster image, avoiding tens of thousands of live Tk road
objects. School markers, candidate sites and colored shortest-road paths are
lightweight overlays. The former S30/S38 zoom is removed. **Full network** fits
the larger routing graph. **Show school names** distinguishes synthetic labels.

**Run GA / PSO / ACO** compares all three on the same scenario at equal evaluation
boundaries. Switch **View algorithm** during or after the run. Pause/resume,
step, playback speed, reset/cancel, saved replay and exports are retained.
**Run selected only** runs a single method.

The main metrics show active SPPGs, weighted score, total allocation distance,
mean schools per active SPPG, capacity lower bound, solver runtime, certified
MILP optimum/gap, repair changes and decoded diversity. The
**School allocation & capacity** tab lists each selected SPPG, its school count,
meal load/capacity/utilization and each school's road distance. Load bars and
convergence appear under **Convergence & capacity**.

**PSO / ACO search** shows PSO preference particles, velocities and personal/global
bests, or ACO construction choices and pheromones. Particle coordinates are a
projection of preferences, not vehicle positions. Animation capture does not
consume random draws or candidate evaluations.

## Experimental validity

The exact allocation MILP uses binary school-assignment and SPPG-activation
variables, the same capacities/radius eligibility, and the same weighted
objective as the shared evaluator. Every algorithm run reports the proven
optimal objective and its gap when certified. If the MILP times out, the status,
incumbent and lower bound are retained; unproven exact gaps remain unavailable.
The oracle's selected count/distance are exported separately when optimal.
MILP and shortest-path preprocessing are excluded from heuristic runtime and
candidate budgets.

Scenario seeds vary candidate locations. Optimizer seeds vary search randomness.
School seeds are separate and fixed across benchmark instances. The benchmark
crosses multiple candidate instances per pool size with multiple optimizer seeds,
rotates algorithm order, and averages instance gaps with equal instance weights.
Candidate pool sizes are different configurations, not different school counts.

GA uses allocation genes, tournament selection, crossover, mutation and elitism.
PSO uses bounded school-to-SPPG preference scores decoded to eligible assignments;
it is a discrete adaptation rather than vanilla continuous PSO. ACO constructs
assignments from allocation pheromones and road-distance heuristics.

Constraint policies:

- `shared_repair`: capacity-neutral construction for all algorithms, followed by
  the same deterministic capacity repair. Recommended main comparison.
- `aco_capacity_aware`: ACO additionally sees residual capacity during construction;
  report separately as an information/fairness sensitivity experiment.
- `no_repair`: invalid candidates retain violations and rank feasibility first.

Repair is heuristic, can change the objective, and can leave an infeasible result.
It does not secretly consolidate valid assignments to improve SPPG count.
All candidate evaluations, including duplicates and initialization, count toward
the same budget; reused GA elites consume no new evaluations. Diagnostics record
repair attempts/changes/objective effects, decoded and repaired assignment
diversity, duplicates, and PSO position/decoded/phenotype changes.

## Reproducible benchmark and exports

    python -m sppg_compare --benchmark --schools 30 --school-seed 101 --school-demand 200 --sppg-capacity 3000 --opening-weight 10 --distance-weight 1 --sizes Small Medium Large --scenario-seeds 101 102 103 --optimizer-seeds 1 2 3 --budget 5000 --output results/planning

Exports retain `benchmark.json`, `runs.csv`, `summary.csv`,
`instance_summary.csv`, and `diagnostics.csv`. They include complete scenarios,
road graphs/source checksums, weights, independent seeds, per-SPPG loads and school
counts, allocation distances, histories, oracle results/gaps and optional captured
search state (`--capture-search`). Exported scenarios can be restored with
`Scenario.from_dict` without a new OSM download.

## Preserved mapped-school tour model

**Mapped school tours** retains the prior model for existing experiments and
exports. It includes all located OSM school features within the original linked
Ciledug area (-6.23118, 106.72340), fixed demand per feature ID, 2*K candidates,
and active limits 3/5/8 for Small/Medium/Large. The Python API's default remains
this legacy model; supplying `school_count` selects the new planning model.
Use `--mapped-school-tours` in the CLI to select it.

The legacy model uses one loaded returning multi-stop tour per selected SPPG,
vehicle capacity equal to SPPG capacity, five minutes unloading per school,
fastest directed paths, and demand-weighted arrival time (portion-minutes).
GA/PSO/ACO also optimize visit order; the exact routing MILP uses the same model.
Legacy difficulty/slack API and CLI options remain available. Main roads use
50 km/h and local streets 25 km/h; these speeds also determine the illustrative
travel times on the new model's shortest-distance paths. They do not alter its
distance objective.

Map data: [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
ODbL. Raster usage follows the
[OSM tile policy](https://operations.osmfoundation.org/policies/tiles/).
Synthetic schools/demand and modeled capacities are not official SPPG records.

## Tests

    python -m unittest discover -s tests -v

Tests use an offline OSM-format fixture. They cover both models, directed paths,
weighted facility MILP versus exhaustive solutions, independent/nested synthetic
schools, capacity infeasibility, budget equality, diagnostics, exports, raster
rendering, GUI controls, synchronized animation, pause/step and replay.
