"""
Ceteris-paribus sensitivity scenario runner.

Regenerates each sensitivity scenario by varying EXACTLY ONE structural
parameter from the published baseline while holding every other parameter at
its standard state, then exporting cohort age-out segmentation to
    outputs/exploratory/sensitivity/scenarios/<name>/children_aged_out_segmentation.csv
which simulation/sensitivity.py aggregates into the robustness table.

Each scenario is run as a paired (uncapped, capped) simulation with the same
seed, so the only thing that changes across the pair is the per-country cap and
the only thing that changes across scenarios is the single varied parameter.

Baseline standard state (see simulation/empirical_params.py):
    India emigration multiplier = 0.7
    EB emigration multipliers   = {EB1 .25, EB2 .50, EB3 .75, EB4 1.0, EB5 .125}
    spouse-probability mult     = 1.0
    child entry-age mult        = 1.0
    total annual visa supply    = 167,394

Scenario grid (paper Appendix values):
    base
    low/high_attrition_india      India emigration mult -> 0.4 / 1.0
    uniform_eb_exit_rates         EB emigration mults   -> all 1.0
    low/high_spouse_prob          spouse prob mult      -> 0.8 / 1.2
    low/high_child_ages           child entry-age mult  -> 0.8 / 1.2
    low/high_allocation           projection visa supply-> 140,000 / 200,000

Runtime: one paired 32-year run per scenario (several minutes each); the full
sweep of 10 scenarios takes roughly 1-1.5 hours. The exit specification is
loaded from the primary case in data/experiment.json. Historical submitted
results cannot substitute for this current-model sweep.

Usage:
    python -m simulation.sensitivity_runner                 # regenerate all scenarios
    python -m simulation.sensitivity_runner --only base low_attrition_india
    python -m simulation.sensitivity_runner --dest /tmp/scen_check   # write elsewhere
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import shlex
from pathlib import Path

import simulation.empirical_params as ep
import simulation.child_processor as cp
from simulation.models import EBCategory
from simulation.sim import Simulation
from simulation.states import SimulationConfig
from simulation.__main__ import export_children_aged_out_segmentation_csv
from simulation.campaign import experiment_cases
from simulation.experiments import parameter_override, source_fingerprint, atomic_json, validate_states

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "outputs" / "exploratory" / "sensitivity" / "scenarios"
YEARS = 32          # FY2009-FY2040
START_YEAR = 2009
PROJECTION_YEARS = range(2025, START_YEAR + YEARS + 22)  # cover long tails safely


# One-parameter override context managers (restore baseline on exit)
@contextlib.contextmanager
def _noop():
    yield


@contextlib.contextmanager
def _india_mult(m):
    old = ep.NATIONALITY_EMIGRATION_MULTIPLIERS["India"]
    ep.NATIONALITY_EMIGRATION_MULTIPLIERS["India"] = m
    try:
        yield
    finally:
        ep.NATIONALITY_EMIGRATION_MULTIPLIERS["India"] = old


@contextlib.contextmanager
def _uniform_eb():
    old = dict(ep.CATEGORY_EMIGRATION_MULTIPLIERS)
    for k in list(ep.CATEGORY_EMIGRATION_MULTIPLIERS):
        ep.CATEGORY_EMIGRATION_MULTIPLIERS[k] = 1.0
    try:
        yield
    finally:
        ep.CATEGORY_EMIGRATION_MULTIPLIERS.clear()
        ep.CATEGORY_EMIGRATION_MULTIPLIERS.update(old)


@contextlib.contextmanager
def _spouse_mult(mult):
    snap = copy.deepcopy(ep.SPOUSE_PRESENCE_EMPIRICAL)
    for pathway in ep.SPOUSE_PRESENCE_EMPIRICAL.values():
        for nat in pathway.values():
            probs = nat.get("probabilities")
            if probs and len(probs) >= 2:
                p1 = min(1.0, max(0.0, probs[1] * mult))
                nat["probabilities"] = [1.0 - p1, p1]
    try:
        yield
    finally:
        ep.SPOUSE_PRESENCE_EMPIRICAL.clear()
        ep.SPOUSE_PRESENCE_EMPIRICAL.update(snap)


@contextlib.contextmanager
def _child_age_mult(mult):
    # Wrap the sampler child_processor actually calls; scale the sampled age
    # and clamp to the valid [0, 20] entry range (21 = aged out).
    orig = cp.sample_child_entry_age

    def wrapped(*args, **kwargs):
        return int(min(20, max(0, round(orig(*args, **kwargs) * mult))))

    cp.sample_child_entry_age = wrapped
    try:
        yield
    finally:
        cp.sample_child_entry_age = orig


@contextlib.contextmanager
def _visa_supply(total):
    # Change projection-period supply only (2025+); leave 2009-2024 actuals.
    snap = dict(ep.HISTORICAL_EB_VISA_POOL)
    for y in PROJECTION_YEARS:
        ep.HISTORICAL_EB_VISA_POOL[y] = total
    try:
        yield
    finally:
        ep.HISTORICAL_EB_VISA_POOL.clear()
        ep.HISTORICAL_EB_VISA_POOL.update(snap)


# Scenario registry: name -> (override factory, parameter description)
SCENARIOS = {
    "base": (lambda: _noop(), "Published baseline (no change)"),
    "low_attrition_india": (lambda: _india_mult(0.4), "India emigration mult 0.7 -> 0.4"),
    "high_attrition_india": (lambda: _india_mult(1.0), "India emigration mult 0.7 -> 1.0"),
    "uniform_eb_exit_rates": (lambda: _uniform_eb(), "EB emigration mults -> all 1.0"),
    "low_spouse_prob": (lambda: _spouse_mult(0.8), "Spouse prob mult 1.0 -> 0.8"),
    "high_spouse_prob": (lambda: _spouse_mult(1.2), "Spouse prob mult 1.0 -> 1.2"),
    "low_child_ages": (lambda: _child_age_mult(0.8), "Child entry-age mult 1.0 -> 0.8"),
    "high_child_ages": (lambda: _child_age_mult(1.2), "Child entry-age mult 1.0 -> 1.2"),
    "low_allocation": (lambda: _visa_supply(140_000), "Projection visa supply -> 140,000/yr"),
    "high_allocation": (lambda: _visa_supply(200_000), "Projection visa supply -> 200,000/yr"),
}


def _run_scenario(name: str, dest_root: Path) -> str:
    override_factory, desc = SCENARIOS[name]
    declaration = json.loads((PROJECT_ROOT / "data/experiment.json").read_text())
    primary = next(case for case in experiment_cases(
        declaration, PROJECT_ROOT / "data/validation/dos_visa_consumption_reconciled.csv"
    ) if case["role"] == "primary")
    fingerprint = source_fingerprint()
    out_dir = dest_root / name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Preserving existing scenario results; choose a new --dest: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[sensitivity] {name}: {desc}")
    # Apply multipliers before the catchall context snapshots its rate table.
    with override_factory(), parameter_override(primary["overrides"]):
        states = {}
        for label, capped in [("uncapped", False), ("capped", True)]:
            cfg = SimulationConfig(
                years=YEARS,
                seed=declaration["seed"],
                output_path=str(out_dir),
                country_cap_enabled=capped,
                debug=False,
                start_year=declaration["start_year"],
            )
            states[label] = Simulation(cfg).run()
            validate_states(states[label])
        if source_fingerprint() != fingerprint:
            raise RuntimeError("Sources or inputs changed during the sensitivity run")
        csv_path = export_children_aged_out_segmentation_csv(
            states["uncapped"], states["capped"], out_dir
        )
    atomic_json(out_dir / "run_metadata.json", {
        "complete": True, "scenario": name, "parameter_varied": desc,
        "seed": declaration["seed"], "start_year": declaration["start_year"],
        "years": YEARS, "primary_overrides": primary["overrides"],
        "source_fingerprint": fingerprint,
    })
    print(f"[sensitivity] {name}: wrote {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate ceteris-paribus sensitivity scenarios.")
    parser.add_argument("--only", nargs="+", choices=list(SCENARIOS), help="Subset of scenarios to run.")
    parser.add_argument("--dest", type=str, default=str(DEFAULT_DEST), help="Destination root for scenario dirs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-year engine logging.")
    args = parser.parse_args()

    if args.quiet:
        logging.disable(logging.WARNING)

    dest_root = Path(args.dest)
    names = args.only or list(SCENARIOS)
    print(f"Regenerating {len(names)} scenario(s) into {dest_root}")
    for name in names:
        _run_scenario(name, dest_root)
    print("\nDone. Summarize this sweep with: python -m simulation.sensitivity --scenarios "
          + shlex.quote(str(dest_root)) + " --output "
          + shlex.quote(str(dest_root.parent / "sensitivity_ageout_summary.csv")))


if __name__ == "__main__":
    main()
