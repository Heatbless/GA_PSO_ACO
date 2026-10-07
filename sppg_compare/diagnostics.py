"""Deterministic search diagnostics; no random draws or candidate caching."""

from collections import Counter


def hamming_diversity(assignments):
    """Mean pairwise normalized Hamming distance without quadratic pairs."""
    count = len(assignments)
    if count < 2:
        return 0.0
    schools = len(assignments[0])
    differences = sum(count * count - sum(v * v for v in Counter(column).values())
                      for column in zip(*assignments))
    return differences / (count * (count - 1) * schools)


class SearchDiagnostics:
    def __init__(self):
        self.seen_decoded = set()
        self.seen_phenotypes = set()
        self.totals = {key: 0 for key in (
            "duplicate_decoded_evaluations", "duplicate_phenotype_evaluations",
            "pre_repair_feasible", "post_repair_feasible", "repair_attempts", "repair_changes",
            "repair_site_limit_attempts", "repair_active_site_reduction", "repair_successes", "repair_changed_genes", "repair_overflow_reduction",
            "repair_objective_delta_sum", "repair_absolute_objective_delta_sum",
            "pso_updates", "pso_position_changes", "pso_decoded_changes",
            "pso_route_order_changes", "pso_phenotype_route_changes", "pso_phenotype_changes", "pso_decoded_changed_genes", "pso_phenotype_changed_genes")}
        self.new_batch()

    def new_batch(self):
        self.decoded_batch = []
        self.phenotype_batch = []
        self.decoded_candidate_batch = []
        self.phenotype_candidate_batch = []

    def record(self, raw, phenotype, repair_enabled, previous=None, previous_phenotype=None,
               position_changed=None, previous_order=None, previous_phenotype_order=None):
        totals = self.totals
        raw_key = (raw.assignment, raw.visit_order) if raw.visit_order else raw.assignment
        def route_signature(assignment, order):
            return tuple((j, tuple(i for i in order if assignment[i] == j)) for j in sorted(set(assignment)))
        phenotype_key = (phenotype.assignment, route_signature(phenotype.assignment, phenotype.visit_order)) if phenotype.visit_order else phenotype.assignment
        totals["duplicate_decoded_evaluations"] += int(raw_key in self.seen_decoded)
        totals["duplicate_phenotype_evaluations"] += int(phenotype_key in self.seen_phenotypes)
        self.seen_decoded.add(raw_key)
        self.seen_phenotypes.add(phenotype_key)
        self.decoded_candidate_batch.append(raw_key)
        self.phenotype_candidate_batch.append(phenotype_key)
        self.decoded_batch.append(raw.assignment)
        self.phenotype_batch.append(phenotype.assignment)
        totals["pre_repair_feasible"] += int(raw.feasible)
        totals["post_repair_feasible"] += int(phenotype.feasible)
        attempted = repair_enabled and (raw.overflow > 0 or raw.active_site_violation > 0)
        changed = raw.assignment != phenotype.assignment
        totals["repair_attempts"] += int(attempted)
        totals["repair_site_limit_attempts"] += int(repair_enabled and raw.active_site_violation > 0)
        totals["repair_active_site_reduction"] += raw.active_site_violation - phenotype.active_site_violation
        totals["repair_changes"] += int(changed)
        totals["repair_successes"] += int(attempted and phenotype.feasible)
        totals["repair_changed_genes"] += sum(a != b for a, b in zip(raw.assignment, phenotype.assignment))
        totals["repair_overflow_reduction"] += raw.overflow - phenotype.overflow
        delta = phenotype.objective - raw.objective
        totals["repair_objective_delta_sum"] += delta
        totals["repair_absolute_objective_delta_sum"] += abs(delta)
        if previous is not None:
            decoded_changes = sum(a != b for a, b in zip(previous, raw.assignment))
            phenotype_changes = sum(a != b for a, b in zip(previous_phenotype, phenotype.assignment))
            totals["pso_updates"] += 1
            totals["pso_position_changes"] += int(position_changed)
            route_changed = previous_order is not None and tuple(previous_order) != raw.visit_order
            phenotype_route_changed = previous_phenotype_order is not None and route_signature(previous_phenotype, previous_phenotype_order) != route_signature(phenotype.assignment, phenotype.visit_order)
            totals["pso_route_order_changes"] += int(route_changed)
            totals["pso_phenotype_route_changes"] += int(phenotype_route_changed)
            totals["pso_decoded_changes"] += int(decoded_changes > 0 or route_changed)
            totals["pso_phenotype_changes"] += int(phenotype_changes > 0 or phenotype_route_changed)
            totals["pso_decoded_changed_genes"] += decoded_changes
            totals["pso_phenotype_changed_genes"] += phenotype_changes

    def snapshot(self, evaluations):
        values = dict(self.totals)
        count = len(self.decoded_batch)
        updates = values["pso_updates"]
        changes = values["repair_changes"]
        values.update({
            "evaluated_batch_size": count,
            "decoded_candidate_unique_ratio": len(set(self.decoded_candidate_batch))/count if count else 0,
            "phenotype_candidate_unique_ratio": len(set(self.phenotype_candidate_batch))/count if count else 0,
            "decoded_unique_ratio": len(set(self.decoded_batch)) / count if count else 0,
            "phenotype_unique_ratio": len(set(self.phenotype_batch)) / count if count else 0,
            "decoded_hamming_diversity": hamming_diversity(self.decoded_batch),
            "phenotype_hamming_diversity": hamming_diversity(self.phenotype_batch),
            "unique_decoded_assignments": len(self.seen_decoded),
            "unique_phenotypes": len(self.seen_phenotypes),
            "duplicate_decoded_rate": values["duplicate_decoded_evaluations"] / evaluations,
            "duplicate_phenotype_rate": values["duplicate_phenotype_evaluations"] / evaluations,
            "repair_attempt_rate": values["repair_attempts"] / evaluations,
            "repair_change_rate": changes / evaluations,
            "repair_mean_objective_delta": values["repair_objective_delta_sum"] / changes if changes else None,
            "repair_mean_absolute_objective_delta": values["repair_absolute_objective_delta_sum"] / changes if changes else None,
            "pso_decoded_change_rate": values["pso_decoded_changes"] / updates if updates else None,
            "pso_phenotype_change_rate": values["pso_phenotype_changes"] / updates if updates else None,
        })
        return values
