"""Observed-allocation historical discrepancy experiment.

This deliberately replaces historical allocation with country/category visa
budgets. It is a calibration stress test, not an independently validated model
or a change to either projection's statutory allocation rule.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .empirical_params import COUNTRIES, RECONSTRUCTION_END_YEAR
from .models import EBCategory


def read_historical_allocations(path):
    targets = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            category = row["category"]
            nationality = row["nationality"]
            if category == "Overall" or nationality not in COUNTRIES:
                continue
            year = int(row["year"])
            key = (year, EBCategory(category), nationality)
            if key in targets:
                raise ValueError(f"Duplicate historical allocation target {key}")
            raw_value = row.get("visas_issued", row.get("visas", ""))
            try:
                value = Decimal(raw_value)
            except (InvalidOperation, TypeError):
                raise ValueError(f"Invalid historical allocation target {key}: {raw_value!r}") from None
            if not value.is_finite() or value < 0 or value != value.to_integral_value():
                raise ValueError(f"Historical allocation target must be a finite nonnegative integer {key}: {raw_value!r}")
            targets[key] = int(value)
    return targets


class HistoricalAllocationReplay:
    """Proxy the standard allocator after the common reconstruction period."""
    def __init__(self, original, targets):
        self.original = original
        self.targets = targets
        self.replay_audit = []

    def process_conversions(self, annual_limit, annual_eb_caps,
                            category_nationality_queues, per_country_caps_by_category,
                            worker_lookup, child_processor, current_year,
                            active_worker_ids=None):
        if current_year > RECONSTRUCTION_END_YEAR:
            return self.original.process_conversions(
                annual_limit, annual_eb_caps, category_nationality_queues,
                per_country_caps_by_category, worker_lookup, child_processor,
                current_year, active_worker_ids=active_worker_ids)
        costs = self.original._build_visa_cost_cache(
            worker_lookup, child_processor, active_worker_ids)
        ids, by_country, by_category = set(), defaultdict(int), {cat: 0 for cat in EBCategory}
        visas, spouses = 0, 0
        cells = defaultdict(int)
        for category in EBCategory:
            for nationality in COUNTRIES:
                target_key = (current_year, category, nationality)
                if target_key not in self.targets:
                    raise ValueError(f"Missing historical target {target_key}")
                budget = self.targets[target_key]
                candidates = [wid for wid in category_nationality_queues[(category, nationality)]
                              if worker_lookup[wid].is_temporary]
                candidates.sort(key=lambda wid: (worker_lookup[wid].entry_year, wid))
                candidates = self.original._shuffle_applicants_within_cohorts(candidates, worker_lookup)
                cell = (category, nationality)
                for wid in candidates:
                    if cells[cell] >= budget or visas >= annual_limit:
                        break
                    cost = costs[wid]
                    if cells[cell] + cost > budget or visas + cost > annual_limit:
                        continue
                    worker = worker_lookup[wid]
                    worker.convert_to_permanent(current_year)
                    ids.add(wid)
                    by_country[nationality] += 1
                    by_category[category] += 1
                    cells[cell] += cost
                    visas += cost
                    spouses += worker.spouse_count
                self.replay_audit.append({
                    "year": current_year, "category": category.value,
                    "nationality": nationality, "target_visas": budget,
                    "used_visas": cells[cell], "unfilled_target": budget - cells[cell],
                })
        return (len(ids), dict(by_country), by_category, ids, visas, spouses,
                dict(cells), {cat: 0 for cat in EBCategory})
