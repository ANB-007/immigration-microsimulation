"""Resumable, isolated paired experiments with per-seed provenance.

Each completed pair is written atomically. Existing records are reused only
when the code, inputs, environment, horizon, seed and parameters agree.
Simulation and rendering are deliberately separate.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .sim import Simulation
from .states import SimulationConfig
from .models import EBCategory
from . import empirical_params as ep

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2


def source_fingerprint():
    scientific_modules = (
        "__init__.py",
        "sim.py", "models.py", "states.py", "child_processor.py",
        "visa_processor.py", "empirical_params.py", "ci_runner.py",
        "experiments.py", "historical_replay.py", "retirement.py", "input_data.py",
        "sensitivity_runner.py",
    )
    paths = [ROOT / "simulation" / name for name in scientific_modules]
    paths += sorted((ROOT / "data" / "petitions").glob("*.json"))
    paths += sorted((ROOT / "data" / "demographics").glob("*_distributions*.json"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {
        "schema_version": SCHEMA_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "files": hashes,
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


CATCHALL_KEYS = frozenset({"catchall_median_age", "catchall_near_zero_age",
                           "catchall_tail_probability", "catchall_categories"})


def _catchall_configuration(overrides):
    """Resolve the named total-hazard mode before any simulation is started."""
    specification = overrides.get("exit_specification")
    if specification not in (None, "legacy", "catchall_retention"):
        raise ValueError(f"Unknown exit_specification: {specification}")
    supplied = {key for key in overrides if key.startswith("catchall_")}
    if supplied - CATCHALL_KEYS:
        raise ValueError(f"Unknown catchall parameters: {sorted(supplied - CATCHALL_KEYS)}")
    if specification != "catchall_retention":
        if supplied:
            raise ValueError("Catchall parameters require exit_specification='catchall_retention'")
        return None
    if "AGE_EXIT_ENABLED" in overrides and overrides["AGE_EXIT_ENABLED"] is not False:
        raise ValueError("Catchall retention disables the legacy AGE_EXIT_ENABLED floor")
    if any(key.startswith("AGE_EXIT_") and key != "AGE_EXIT_ENABLED" for key in overrides):
        raise ValueError("Legacy age-floor parameters do not apply to catchall retention")
    age_keys = {"catchall_median_age", "catchall_near_zero_age"}
    if supplied & age_keys and not age_keys.issubset(supplied):
        raise ValueError("Declare both catchall_median_age and catchall_near_zero_age")
    return {"median_age": overrides.get("catchall_median_age", 60.0),
            "near_zero_age": overrides.get("catchall_near_zero_age", 65.0),
            "tail_probability": overrides.get("catchall_tail_probability", .05),
            "categories": overrides.get("catchall_categories", ("EB-1", "EB-2", "EB-3", "EB-4", "EB-5"))}


@contextlib.contextmanager
def parameter_override(overrides):
    """Apply a named exit specification and explicit overrides in one worker.

    ``exit_specification='catchall_retention'`` selects a total catchall hazard,
    with central age parameters 60/65/.05 and default EB-1 through EB-5 coverage. It
    replaces the old age floor; it does not stack independent death or
    withdrawal components. Omitting the name preserves historical behavior.
    """
    catchall = _catchall_configuration(overrides)
    original = {}
    try:
        with contextlib.ExitStack() as stack:
            for key, value in overrides.items():
                if key == "exit_specification" or key in CATCHALL_KEYS:
                    continue
                if key == "historical_allocation_path":
                    continue
                if not key.startswith("AGE_EXIT_") or not hasattr(ep, key):
                    raise ValueError(f"Unsupported experiment override: {key}")
                original[key] = getattr(ep, key)
                setattr(ep, key, value)
            if catchall is not None:
                from .retirement import catchall_retention_exit_rates
                stack.enter_context(catchall_retention_exit_rates(**catchall))
            yield
    finally:
        for key, value in original.items():
            setattr(ep, key, value)


def validate_states(states):
    """Check independent stock/flow and family identities before export."""
    previous_stock = 0
    previous_ageouts = 0
    for state in states:
        if state.visas_consumed_this_year > state.visas_available_this_year:
            raise ValueError(
                f"Annual visa pool exceeded in {state.year}: "
                f"{state.visas_consumed_this_year} > {state.visas_available_this_year}"
            )
        expected = previous_stock + state.new_temporary - state.converted_temps - state.exited_this_year
        if expected != state.temporary_workers:
            raise ValueError(f"Principal balance failed in {state.year}: {expected} != {state.temporary_workers}")
        if sum(state.exited_by_nationality.values()) != state.exited_this_year:
            raise ValueError(f"Nationality exit balance failed in {state.year}")
        if sum(state.exited_by_eb_category.values()) != state.exited_this_year:
            raise ValueError(f"Category exit balance failed in {state.year}")
        family_visas = state.converted_temps + state.converted_spouses + state.children_saved_this_year
        if family_visas != state.visas_consumed_this_year:
            raise ValueError(f"Family visa balance failed in {state.year}")
        if sum(state.visas_consumed_by_category_nationality.values()) != family_visas:
            raise ValueError(f"Visa cell balance failed in {state.year}")
        if state.cumulative_children_aged_out != previous_ageouts + state.children_aged_out_this_year:
            raise ValueError(f"Age-out balance failed in {state.year}")
        if sum(state.aged_out_by_nationality.values()) != state.cumulative_children_aged_out:
            raise ValueError(f"Nationality age-out balance failed in {state.year}")
        previous_stock = state.temporary_workers
        previous_ageouts = state.cumulative_children_aged_out


def _cohorts(sim, scenario):
    rows = []
    outcomes = sim.child_processor.get_outcomes_by_entry_year_nationality_eb()
    for (year, nationality, category), counts in sorted(outcomes.items()):
        total = counts["total_created"]
        if not total:
            continue
        remaining = total - counts["aged_out"] - counts["saved"] - counts["exited"]
        if remaining < 0:
            raise ValueError("Negative unresolved cohort population")
        rows.append({
            "scenario": scenario, "entry_year": year, "nationality": nationality,
            "eb_category": category, **counts, "still_dependent": remaining,
            "ageout_pct": counts["aged_out"] / total * 100,
            "save_pct": counts["saved"] / total * 100,
            "exit_pct": counts["exited"] / total * 100,
        })
    return rows


def _state_cells(states, scenario):
    rows = []
    for state in states:
        for category in EBCategory:
            for nationality in ep.COUNTRIES:
                cell = (category, nationality)
                rows.append({
                    "scenario": scenario, "year": state.year,
                    "category": category.value, "nationality": nationality,
                    "principal_backlog": state.queue_backlog_by_eb_category_nationality.get(cell, 0),
                    "annual_visas": state.visas_consumed_by_category_nationality.get(cell, 0),
                    "annual_principal_exits": state.exited_by_category_nationality.get(cell, 0),
                })
    return rows


def _queue_timing(sim, scenario):
    """Aggregate model-entry tenure, explicitly not legal priority-date waits."""
    from collections import Counter
    counts = Counter()
    final_year = sim.current_year - 1
    for worker in sim.worker_lookup.values():
        if worker.is_temporary:
            end_year, outcome = final_year, "still_waiting"
        elif worker.is_permanent:
            end_year, outcome = worker.conversion_year, "converted"
        else:
            end_year, outcome = worker.exit_year, "exited"
        counts[(worker.nationality, worker.eb_category.value, worker.entry_year,
                outcome, end_year - worker.entry_year)] += 1
    return [
        {"scenario": scenario, "nationality": key[0], "category": key[1],
         "entry_year": key[2], "outcome": key[3], "model_entry_tenure_years": key[4],
         "count": count}
        for key, count in sorted(counts.items())
    ]


def _entry_inventory(sim, scenario, year):
    """Record observed-year model entry cohorts; do not label them priority dates."""
    from collections import Counter
    principals, people = Counter(), Counter()
    for wid in sim.temp_worker_ids:
        worker = sim.worker_lookup[wid]
        key = (worker.nationality, worker.eb_category.value, worker.entry_year)
        principals[key] += 1
        people[key] += 1 + worker.spouse_count + len(sim.child_processor.children_by_parent_id.get(wid, ()))
    return [{"scenario": scenario, "year": year, "nationality": key[0],
             "category": key[1], "model_entry_year": key[2],
             "principal_backlog": count, "principal_and_active_derivative_backlog": people[key]}
             for key, count in sorted(principals.items())]


def _age_profile(sim, scenario):
    """Final waiting age and age at an absorbing event, without new RNG draws.

    The engine ages every stored worker, including those already removed.
    Recover age at conversion/exit from its year and the end-of-step age.
    Remaining principals use end-of-step age; event ages precede that year's
    aging step because conversion and queue exit occur earlier in the step.
    """
    from collections import Counter
    counts = Counter()
    final_year = sim.current_year - 1
    for worker in sim.worker_lookup.values():
        if worker.is_temporary:
            outcome, year = "still_waiting", final_year
            age = worker.age
        elif worker.is_permanent:
            outcome, year = "converted", worker.conversion_year
            age = worker.age - (final_year - year + 1)
        else:
            outcome, year = "exited", worker.exit_year
            age = worker.age - (final_year - year + 1)
        counts[(outcome, year, worker.nationality, worker.eb_category.value, age)] += 1
    return [{"scenario": scenario, "outcome": outcome, "observation_or_event_year": year,
             "nationality": nationality, "category": category, "age": age, "principals": count}
            for (outcome, year, nationality, category, age), count in sorted(counts.items())]


def _run_job(job):
    from .ci_runner import _extract_run
    logging.disable(logging.WARNING)
    path = Path(job["output"]) / "runs" / f"seed_{job['seed']}.json"
    identity = job["identity"]
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("identity") != identity:
            raise ValueError(f"Incompatible existing result: {path}; use a new experiment directory")
        if existing.get("complete") is not True:
            raise ValueError(f"Incomplete result: {path}")
        return str(path), True
    start = time.perf_counter()
    states_by_policy = {}
    cohorts, cells, timings, inventories, replay_audits, age_profiles = [], [], [], [], [], []
    with parameter_override(job["overrides"]):
        for scenario, capped in (("Uncapped", False), ("Capped", True)):
            config = SimulationConfig(
                years=job["years"], seed=job["seed"], output_path=job["output"],
                country_cap_enabled=capped, debug=False, start_year=job["start_year"],
            )
            sim = Simulation(config)
            if job["overrides"].get("historical_allocation_path"):
                from .historical_replay import HistoricalAllocationReplay, read_historical_allocations
                sim.visa_processor = HistoricalAllocationReplay(
                    sim.visa_processor,
                    read_historical_allocations(job["overrides"]["historical_allocation_path"]))
            for _ in range(config.years):
                state = sim.step()
                if state.year in (2023, 2024):
                    inventories.extend(_entry_inventory(sim, scenario, state.year))
            states = sim.states
            validate_states(states)
            states_by_policy[scenario] = states
            cohorts.extend(_cohorts(sim, scenario))
            cells.extend(_state_cells(states, scenario))
            timings.extend(_queue_timing(sim, scenario))
            age_profiles.extend(_age_profile(sim, scenario))
            if hasattr(sim.visa_processor, "replay_audit"):
                replay_audits.extend({"scenario": scenario, **row} for row in sim.visa_processor.replay_audit)
            del sim
            gc.collect()
    raw = _extract_run(states_by_policy["Uncapped"], states_by_policy["Capped"])
    historical = raw["years"] <= ep.RECONSTRUCTION_END_YEAR
    for key in list(raw):
        if key.startswith("annual_") and key.endswith("_unc"):
            counterpart = key[:-4] + "_cap"
            if counterpart in raw and not np.array_equal(raw[key][historical], raw[counterpart][historical]):
                raise ValueError(f"Policies have different historical results: {key}")
    serial = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in raw.items()}
    atomic_json(path, {
        "identity": identity, "complete": True, "seed": job["seed"],
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_unit": "bytes" if sys.platform == "darwin" else "KiB",
        "raw": serial, "cohorts": cohorts, "annual_cells": cells, "queue_timing": timings,
        "historical_entry_inventory": inventories, "historical_replay_audit": replay_audits,
        "principal_age_profile": age_profiles,
    })
    return str(path), False


def run_campaign(config, n_runs, workers=1, overrides=None):
    if n_runs < 1 or workers < 1:
        raise ValueError("runs and workers must be positive")
    output = Path(config.output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    overrides = dict(overrides or {})
    _catchall_configuration(overrides)
    fingerprint = source_fingerprint()
    if overrides.get("historical_allocation_path"):
        external = Path(overrides["historical_allocation_path"]).resolve()
        overrides["historical_allocation_path"] = str(external)
        fingerprint["historical_allocation_sha256"] = hashlib.sha256(external.read_bytes()).hexdigest()
    jobs = []
    for i in range(n_runs):
        seed = config.seed + i
        identity = {
            **fingerprint, "start_year": config.start_year, "years": config.years,
            "seed": seed, "overrides": overrides, "policy_branch_year": ep.RECONSTRUCTION_END_YEAR + 1,
        }
        jobs.append({"seed": seed, "years": config.years, "start_year": config.start_year,
                     "output": str(output), "overrides": overrides, "identity": identity})
    atomic_json(output / "requested_campaign.json", {
        "fingerprint": fingerprint, "requested_seeds": [j["seed"] for j in jobs],
        "years": config.years, "start_year": config.start_year, "overrides": overrides,
        "workers": workers, "status": "requested",
    })
    if workers == 1:
        for job in jobs:
            path, reused = _run_job(job)
            print(f"{'Reused' if reused else 'Completed'} {path}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                path, reused = future.result()
                print(f"{'Reused' if reused else 'Completed'} {path}", flush=True)
    records = [json.loads((output / "runs" / f"seed_{job['seed']}.json").read_text()) for job in jobs]
    runs = []
    for record in records:
        runs.append({key: np.asarray(value) if isinstance(value, list) else float(value)
                     for key, value in record["raw"].items()})
    import pandas as pd
    for field in ("annual_cells", "queue_timing", "historical_entry_inventory", "historical_replay_audit", "principal_age_profile"):
        rows = [{**row, "run_index": i, "seed": record["seed"]}
                for i, record in enumerate(records) for row in record[field]]
        pd.DataFrame(rows).to_csv(output / f"{field}.csv", index=False)
    atomic_json(output / "completed_campaign.json", {
        "fingerprint": fingerprint, "seeds": [r["seed"] for r in records],
        "overrides": overrides, "years": config.years, "start_year": config.start_year,
        "elapsed_seconds_by_seed": [r["elapsed_seconds"] for r in records],
        "status": "complete", "uncertainty_scope": "stochastic, conditional on named model inputs",
    })
    return runs, [record["cohorts"] for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2014)
    parser.add_argument("--years", type=int, default=32)
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overrides", default="{}", help="JSON object of supported scalar sensitivity parameters")
    args = parser.parse_args()
    cfg = SimulationConfig(args.years, args.seed, args.output, False, False, args.start_year)
    runs, cohorts = run_campaign(cfg, args.runs, workers=args.workers, overrides=json.loads(args.overrides))
    from .ci_runner import save_raw_runs_to_csv, save_raw_cohorts_to_csv
    save_raw_runs_to_csv(runs, args.output, args.seed)
    save_raw_cohorts_to_csv(cohorts, args.output, args.seed)


if __name__ == "__main__":
    main()
