"""
Monte Carlo confidence interval runner for immigration microsimulation.

Executes N paired (uncapped + capped) simulation runs and aggregates
95% empirical confidence intervals for all headline policy metrics.

DESIGN PRINCIPLES
Paired runs
    Each run i uses the same seed for both the uncapped and capped scenario.
    Because worker arrivals, queue exits, and child demographics are drawn
    from identical RNG states (via SeedSequence), the CI on the policy
    difference (capped - uncapped) is a tight matched-pairs estimate.
    This is the correct estimator for a policy paper arguing that caps
    cause additional age-outs over and above baseline stochastic variation.
    An independent-samples design would produce CIs 2-4x wider on the
    difference, obscuring a real policy signal.

Empirical percentile CI method
    We use np.percentile([2.5, 97.5]) rather than mean +/- 1.96*sigma. This
    makes no normality assumption and is valid even if age-out counts have
    a skewed distribution across runs (plausible given rare-event dynamics
    in the EB-3 India queue).

Lightweight extraction
    Only scalar and dict fields from SimulationState are stored per run.
    children_aged_out_this_year_list (raw AgedOutChild objects) is never
    stored. Per-nationality annual counts are recovered from cumulative
    counters via np.diff -- this is correct because aged_out_by_nationality
    is built from child_processor.aged_out_children (the full historical
    list) inside ChildAgeoutStatistics.calculate(), making it a running
    cumulative total at every year's snapshot.
    Per-EB-category annual counts are tallied by iterating the per-year
    list during extraction (read-only, objects are not stored in row).
    Per-nationality queue-exit counts follow the same cumulative-diff
    pattern using exited_by_nationality. Per-EB-category queue-exit counts
    follow the same cumulative-diff pattern using exited_by_eb_category.

No side effects in the inner loop
    Diagnostics, CSV exports, console summaries, and visualizations are
    entirely suppressed during the N runs. They execute once on the
    aggregated CIResults object.

Seed strategy
    Run i uses seed = base_seed + i. Because Simulation.__init__ passes
    this through np.random.SeedSequence, incrementing by 1 produces a
    cryptographically independent stream, not a permutation of the same
    one. The four child streams (worker creation, queue exits, child
    processor, visa processor) are all re-derived from seed_i inside each
    fresh Simulation instance -- there is zero cross-run state.
"""

from __future__ import annotations


import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np


from .sim import Simulation
from .states import SimulationConfig
from .empirical_params import COUNTRIES
from .models import EBCategory


logger = logging.getLogger(__name__)


# Project root -- anchors output paths so CI artifacts land in the repo's
# outputs/ci/ regardless of the current working directory (matches the
# anchored readers in mc_calc.py and ci_figures.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Ordered list of EB categories -- used in extraction, aggregation, and export
EB_CATS = list(EBCategory)  # [EBCategory.EB1, EBCategory.EB2, ...]


# Type aliases


# Scalar 95% CI: (mean, p2.5, p97.5)
ScalarCI = Tuple[float, float, float]


# Time-series 95% CI: three arrays each of shape (n_years,)
TimeSeriesCI = Tuple[np.ndarray, np.ndarray, np.ndarray]


# Progress bar helper


def _progress_bar(
    i: int,
    n_runs: int,
    elapsed: float,
    seed_i: int,
    bar_width: int = 28,
) -> None:
    """
    Overwrite the current terminal line with a live progress bar.

    Format:
        Run  42/200  [████████████░░░░░░░░░░░░░░░░]  21.0%  0:02:15 < 0:08:30  seed=2056

    Args:
        i:          Zero-based index of the run just completed.
        n_runs:     Total number of runs.
        elapsed:    Wall-clock seconds since the loop started.
        seed_i:     Seed used for this run (for reproducibility tracing).
        bar_width:  Number of characters inside the [ ] block.
    """
    completed = i + 1
    pct = completed / n_runs

    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)

    elapsed_str = _fmt_seconds(elapsed)
    if completed > 1:
        eta_seconds = (elapsed / completed) * (n_runs - completed)
        eta_str = _fmt_seconds(eta_seconds)
    else:
        eta_str = "--:--"

    line = f"  Run {completed:>{len(str(n_runs))}}/{n_runs}  " f"[{bar}]  {pct*100:5.1f}%  " f"{elapsed_str} < {eta_str}  " f"seed={seed_i}"

    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


def _fmt_seconds(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# Lightweight per-run extraction


def _extract_run(states_unc, states_cap) -> Dict:
    """
    Extract a flat dict of lightweight numeric arrays from one paired run.

    All arrays are float64. Per-nationality annual aged-out counts are
    recovered from cumulative snapshots via np.diff(prepend=0).
    Per-EB-category annual aged-out counts are tallied from
    children_aged_out_this_year_list (read-only; objects not stored).
    Per-nationality annual queue-exit counts are recovered from cumulative
    snapshots of exited_by_nationality via np.diff(prepend=0).
    Per-EB-category annual queue-exit counts are recovered from cumulative
    snapshots of exited_by_eb_category via np.diff(prepend=0),
    mirroring the exited_by_nationality pattern. Keys follow the pattern:
        annual_<metric>_<scenario>      time-series array, shape (n_years,)
        cumul_<metric>_<scenario>       time-series array, shape (n_years,)
        final_<metric>_<scenario>       float scalar
        diff_<metric>                   float scalar  (capped - uncapped)
    """

    # Scalar field helpers
    def ts(states, attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in states], dtype=np.float64)

    def dict_ts(states, attr: str, key) -> np.ndarray:
        return np.array([getattr(s, attr).get(key, 0) for s in states], dtype=np.float64)

    f_unc = states_unc[-1]
    f_cap = states_cap[-1]

    row: Dict = {
        "years": np.array([s.year for s in states_unc], dtype=np.int32),
        # Annual aged-out (direct per-year field)
        "annual_aged_out_unc": ts(states_unc, "children_aged_out_this_year"),
        "annual_aged_out_cap": ts(states_cap, "children_aged_out_this_year"),
        # Cumulative aged-out
        "cumul_aged_out_unc": ts(states_unc, "cumulative_children_aged_out"),
        "cumul_aged_out_cap": ts(states_cap, "cumulative_children_aged_out"),
        # Annual total backlog (sum across all nationalities)
        "annual_backlog_total_unc": np.array(
            [sum(s.queue_backlog_by_country.values()) for s in states_unc],
            dtype=np.float64,
        ),
        "annual_backlog_total_cap": np.array(
            [sum(s.queue_backlog_by_country.values()) for s in states_cap],
            dtype=np.float64,
        ),
        # Annual principal conversions
        "annual_conversions_unc": ts(states_unc, "converted_temps"),
        "annual_conversions_cap": ts(states_cap, "converted_temps"),
        # Annual visa consumption (principals + spouses + children)
        "annual_visas_consumed_unc": ts(states_unc, "visas_consumed_this_year"),
        "annual_visas_consumed_cap": ts(states_cap, "visas_consumed_this_year"),
        # Annual queue exits
        "annual_exited_unc": ts(states_unc, "exited_this_year"),
        "annual_exited_cap": ts(states_cap, "exited_this_year"),
        # Final-state scalars
        "final_cumul_aged_out_unc": float(f_unc.cumulative_children_aged_out),
        "final_cumul_aged_out_cap": float(f_cap.cumulative_children_aged_out),
        "final_backlog_total_unc": float(sum(f_unc.queue_backlog_by_country.values())),
        "final_backlog_total_cap": float(sum(f_cap.queue_backlog_by_country.values())),
        "final_cumul_conversions_unc": float(f_unc.cumulative_conversions),
        "final_cumul_conversions_cap": float(f_cap.cumulative_conversions),
        "final_cumul_visas_unc": float(f_unc.cumulative_visas_consumed),
        "final_cumul_visas_cap": float(f_cap.cumulative_visas_consumed),
        "final_cumul_exited_unc": float(f_unc.cumulative_exited),
        "final_cumul_exited_cap": float(f_cap.cumulative_exited),
        # Paired policy-difference scalars (tight matched-pairs estimator)
        "diff_cumul_aged_out": float(f_cap.cumulative_children_aged_out - f_unc.cumulative_children_aged_out),
        "diff_backlog_total": float(sum(f_cap.queue_backlog_by_country.values()) - sum(f_unc.queue_backlog_by_country.values())),
    }

    # Per-nationality fields
    for nat in COUNTRIES:
        cumul_ao_unc = dict_ts(states_unc, "aged_out_by_nationality", nat)
        cumul_ao_cap = dict_ts(states_cap, "aged_out_by_nationality", nat)
        row[f"annual_aged_out_{nat}_unc"] = np.diff(cumul_ao_unc, prepend=0.0)
        row[f"annual_aged_out_{nat}_cap"] = np.diff(cumul_ao_cap, prepend=0.0)

        row[f"annual_backlog_{nat}_unc"] = dict_ts(states_unc, "queue_backlog_by_country", nat)
        row[f"annual_backlog_{nat}_cap"] = dict_ts(states_cap, "queue_backlog_by_country", nat)

        row[f"final_backlog_{nat}_unc"] = float(f_unc.queue_backlog_by_country.get(nat, 0))
        row[f"final_backlog_{nat}_cap"] = float(f_cap.queue_backlog_by_country.get(nat, 0))
        row[f"final_aged_out_{nat}_unc"] = float(f_unc.aged_out_by_nationality.get(nat, 0))
        row[f"final_aged_out_{nat}_cap"] = float(f_cap.aged_out_by_nationality.get(nat, 0))
        row[f"final_ageout_pct_{nat}_unc"] = float(f_unc.children_ageout_percentage_by_nationality.get(nat, 0.0))
        row[f"final_ageout_pct_{nat}_cap"] = float(f_cap.children_ageout_percentage_by_nationality.get(nat, 0.0))

        row[f"diff_aged_out_{nat}"] = float(f_cap.aged_out_by_nationality.get(nat, 0) - f_unc.aged_out_by_nationality.get(nat, 0))
        row[f"diff_ageout_pct_{nat}"] = float(f_cap.children_ageout_percentage_by_nationality.get(nat, 0.0) - f_unc.children_ageout_percentage_by_nationality.get(nat, 0.0))
        row[f"diff_backlog_{nat}"] = float(f_cap.queue_backlog_by_country.get(nat, 0) - f_unc.queue_backlog_by_country.get(nat, 0))

        # Queue exits by nationality
        cumul_ex_unc = dict_ts(states_unc, "exited_by_nationality", nat)
        cumul_ex_cap = dict_ts(states_cap, "exited_by_nationality", nat)
        row[f"annual_exited_{nat}_unc"] = np.diff(cumul_ex_unc, prepend=0.0)
        row[f"annual_exited_{nat}_cap"] = np.diff(cumul_ex_cap, prepend=0.0)
        row[f"final_exited_{nat}_unc"] = float(f_unc.exited_by_nationality.get(nat, 0))
        row[f"final_exited_{nat}_cap"] = float(f_cap.exited_by_nationality.get(nat, 0))
        row[f"diff_exited_{nat}"] = float(f_cap.exited_by_nationality.get(nat, 0) - f_unc.exited_by_nationality.get(nat, 0))

    # Per-EB-category fields
    # Aged-out tallied from children_aged_out_this_year_list in a single pass
    # per year (one iteration over the list regardless of how many EB cats).
    # Queue exits read from exited_by_eb_category (cumulative dict) via np.diff,
    # mirroring the exited_by_nationality pattern.
    annual_unc_by_eb = {eb: np.zeros(len(states_unc), dtype=np.float64) for eb in EB_CATS}
    annual_cap_by_eb = {eb: np.zeros(len(states_cap), dtype=np.float64) for eb in EB_CATS}
    for yr_idx, (s_u, s_c) in enumerate(zip(states_unc, states_cap)):
        for child in s_u.children_aged_out_this_year_list:
            if child.parent_eb_category in annual_unc_by_eb:
                annual_unc_by_eb[child.parent_eb_category][yr_idx] += 1
        for child in s_c.children_aged_out_this_year_list:
            if child.parent_eb_category in annual_cap_by_eb:
                annual_cap_by_eb[child.parent_eb_category][yr_idx] += 1

    for eb_cat in EB_CATS:
        key = eb_cat.value.replace("-", "")  # "EB1", "EB2", ...
        annual_unc = annual_unc_by_eb[eb_cat]
        annual_cap = annual_cap_by_eb[eb_cat]
        cumul_ex_unc = dict_ts(states_unc, "exited_by_eb_category", eb_cat)
        cumul_ex_cap = dict_ts(states_cap, "exited_by_eb_category", eb_cat)
        row[f"annual_aged_out_{key}_unc"] = annual_unc
        row[f"annual_aged_out_{key}_cap"] = annual_cap
        row[f"final_aged_out_{key}_unc"] = float(annual_unc.sum())
        row[f"final_aged_out_{key}_cap"] = float(annual_cap.sum())
        row[f"diff_aged_out_{key}"] = float(annual_cap.sum() - annual_unc.sum())
        row[f"annual_exited_{key}_unc"] = np.diff(cumul_ex_unc, prepend=0.0)
        row[f"annual_exited_{key}_cap"] = np.diff(cumul_ex_cap, prepend=0.0)
        row[f"final_exited_{key}_unc"] = float(cumul_ex_unc[-1])
        row[f"final_exited_{key}_cap"] = float(cumul_ex_cap[-1])
        row[f"diff_exited_{key}"] = float(cumul_ex_cap[-1] - cumul_ex_unc[-1])

    return row


# NEW (UPDATED)
def _extract_cohort_run(sim_unc: Simulation, sim_cap: Simulation) -> List[Dict]:
    """
    Extract per-(entry_year, nationality, eb_category) cohort outcome counts.

    Calls child_processor.get_outcomes_by_entry_year_nationality_eb() on each
    completed Simulation object. Returns a flat list of dicts -- one per
    (scenario, entry_year, nationality, eb_category) combination -- suitable
    for direct row-by-row writing to ci_raw_cohorts.csv.

    Args:
        sim_unc: Completed uncapped Simulation object (after .run()).
        sim_cap: Completed capped Simulation object (after .run()).

    Returns:
        List of dicts, each with keys:
            scenario, entry_year, nationality, eb_category,
            total_created, aged_out, saved, exited, still_dependent,
            ageout_pct, save_pct, exit_pct
    """
    records = []
    for scenario_name, sim in [("Uncapped", sim_unc), ("Capped", sim_cap)]:
        outcomes = sim.child_processor.get_outcomes_by_entry_year_nationality_eb()  # NEW
        for (entry_year, nationality, eb_category), counts in sorted(outcomes.items()):  # NEW
            total = counts["total_created"]
            if total == 0:
                continue
            aged_out = counts["aged_out"]
            saved = counts["saved"]
            exited = counts["exited"]
            still_dependent = total - aged_out - saved - exited
            records.append(
                {
                    "scenario": scenario_name,
                    "entry_year": entry_year,
                    "nationality": nationality,
                    "eb_category": eb_category,  # NEW
                    "total_created": total,
                    "aged_out": aged_out,
                    "saved": saved,
                    "exited": exited,
                    "still_dependent": still_dependent,
                    "ageout_pct": round(aged_out / total * 100, 6),
                    "save_pct": round(saved / total * 100, 6),
                    "exit_pct": round(exited / total * 100, 6),
                }
            )
    return records


# END NEW


# CI aggregation helpers


def _ts_ci(runs: List[Dict], key: str) -> TimeSeriesCI:
    """
    Stack a time-series array across all runs.

    Returns (mean, p2.5, p97.5), each of shape (n_years,).
    Uses empirical percentile method -- no normality assumption.
    """
    mat = np.stack([r[key] for r in runs], axis=0)  # (n_runs, n_years)
    return (
        mat.mean(axis=0),
        np.percentile(mat, 2.5, axis=0),
        np.percentile(mat, 97.5, axis=0),
    )


def _sc_ci(runs: List[Dict], key: str) -> ScalarCI:
    """
    Stack a scalar across all runs.

    Returns (mean, p2.5, p97.5) as floats.
    Uses empirical percentile method.
    """
    vals = np.array([r[key] for r in runs], dtype=np.float64)
    return (
        float(vals.mean()),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


# CIResults dataclass


@dataclass
class CIResults:
    """
    Aggregated 95% confidence interval results from N paired Monte Carlo runs.

    Naming convention
    *_unc / *_cap      uncapped / capped scenario
    TimeSeriesCI       (mean_array, lo_array, hi_array), each shape (n_years,)
    ScalarCI           (mean, lo_2.5, hi_97.5) as plain floats
    diff_*             capped - uncapped (matched-pairs estimator)
    *_by_nat_*         Dict[nationality_str -> TimeSeriesCI or ScalarCI]
    *_by_eb_*          Dict[eb_key_str -> TimeSeriesCI or ScalarCI]
                       eb_key_str = EBCategory.value.replace("-","")
                       e.g. "EB1", "EB2", "EB3", "EB4", "EB5"

    All percentiles are empirical (np.percentile), not normal-approximation.
    """

    n_runs: int
    years: np.ndarray  # shape (n_years,), dtype int32

    # Annual time-series
    annual_aged_out_unc: TimeSeriesCI
    annual_aged_out_cap: TimeSeriesCI
    annual_backlog_total_unc: TimeSeriesCI
    annual_backlog_total_cap: TimeSeriesCI
    annual_conversions_unc: TimeSeriesCI
    annual_conversions_cap: TimeSeriesCI
    annual_visas_consumed_unc: TimeSeriesCI
    annual_visas_consumed_cap: TimeSeriesCI
    annual_exited_unc: TimeSeriesCI
    annual_exited_cap: TimeSeriesCI

    # Cumulative time-series
    cumul_aged_out_unc: TimeSeriesCI
    cumul_aged_out_cap: TimeSeriesCI

    # Per-nationality annual time-series
    annual_aged_out_by_nat_unc: Dict[str, TimeSeriesCI]
    annual_aged_out_by_nat_cap: Dict[str, TimeSeriesCI]
    annual_backlog_by_nat_unc: Dict[str, TimeSeriesCI]
    annual_backlog_by_nat_cap: Dict[str, TimeSeriesCI]
    annual_exited_by_nat_unc: Dict[str, TimeSeriesCI]
    annual_exited_by_nat_cap: Dict[str, TimeSeriesCI]

    # Per-EB-category annual time-series
    annual_aged_out_by_eb_unc: Dict[str, TimeSeriesCI]
    annual_aged_out_by_eb_cap: Dict[str, TimeSeriesCI]
    annual_exited_by_eb_unc: Dict[str, TimeSeriesCI]
    annual_exited_by_eb_cap: Dict[str, TimeSeriesCI]

    # Final-state scalars
    final_cumul_aged_out_unc: ScalarCI
    final_cumul_aged_out_cap: ScalarCI
    final_backlog_total_unc: ScalarCI
    final_backlog_total_cap: ScalarCI
    final_cumul_conversions_unc: ScalarCI
    final_cumul_conversions_cap: ScalarCI
    final_cumul_visas_unc: ScalarCI
    final_cumul_visas_cap: ScalarCI
    final_cumul_exited_unc: ScalarCI
    final_cumul_exited_cap: ScalarCI

    # Paired policy-difference CIs (matched-pairs, tightest estimator)
    diff_cumul_aged_out: ScalarCI
    diff_backlog_total: ScalarCI

    # Per-nationality final-state scalars
    final_backlog_by_nat_unc: Dict[str, ScalarCI]
    final_backlog_by_nat_cap: Dict[str, ScalarCI]
    final_aged_out_by_nat_unc: Dict[str, ScalarCI]
    final_aged_out_by_nat_cap: Dict[str, ScalarCI]
    final_ageout_pct_by_nat_unc: Dict[str, ScalarCI]
    final_ageout_pct_by_nat_cap: Dict[str, ScalarCI]
    final_exited_by_nat_unc: Dict[str, ScalarCI]
    final_exited_by_nat_cap: Dict[str, ScalarCI]

    # Per-nationality paired policy-difference CIs
    diff_aged_out_by_nat: Dict[str, ScalarCI]
    diff_ageout_pct_by_nat: Dict[str, ScalarCI]
    diff_backlog_by_nat: Dict[str, ScalarCI]
    diff_exited_by_nat: Dict[str, ScalarCI]

    # Per-EB-category final-state scalars
    final_aged_out_by_eb_unc: Dict[str, ScalarCI]
    final_aged_out_by_eb_cap: Dict[str, ScalarCI]
    final_exited_by_eb_unc: Dict[str, ScalarCI]
    final_exited_by_eb_cap: Dict[str, ScalarCI]

    # Per-EB-category paired policy-difference CIs
    diff_aged_out_by_eb: Dict[str, ScalarCI]
    diff_exited_by_eb: Dict[str, ScalarCI]


# Main entry point


def run_ci_analysis(
    config: SimulationConfig,
    n_runs: int = 200,
    base_seed: Optional[int] = None,
) -> CIResults:
    """
    Run N paired (uncapped, capped) simulations and return 95% CI results.

    Args:
        config:     Base SimulationConfig. years, start_year, and output_path
                    are forwarded unchanged. country_cap_enabled is overridden
                    per scenario. seed is used as base_seed if not provided.
        n_runs:     Number of Monte Carlo iterations.
                    100  -> fast validation / sanity check (~10 min typical)
                    200  -> publication quality
                    500  -> very tight bands if runtime permits
        base_seed:  Seed for run 0. Run i uses base_seed + i.
                    Defaults to config.seed.

    Returns:
        CIResults with means and empirical 95% CI bands for all metrics.

    Raises:
        ValueError: If n_runs < 2 (cannot compute a meaningful percentile CI).
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be >= 2 to compute CI bounds; got {n_runs}.")

    if base_seed is None:
        base_seed = config.seed

    runs: List[Dict] = []
    cohort_runs: List[List[Dict]] = []  # existing

    print(f"\n{'=' * 64}")
    print(f"  CI MODE -- {n_runs} paired Monte Carlo runs")
    print(f"  Base seed  : {base_seed}")
    print(f"  Period     : {config.start_year}-" f"{config.start_year + config.years - 1}  ({config.years} years)")
    print(f"  Scenarios  : Uncapped + 7% Cap  (paired per run)")
    print(f"{'=' * 64}")

    print()

    loop_start = time.perf_counter()

    for i in range(n_runs):
        seed_i = base_seed + i

        cfg_unc = SimulationConfig(
            years=config.years,
            seed=seed_i,
            output_path=config.output_path,
            country_cap_enabled=False,
            debug=False,
            start_year=config.start_year,
        )
        cfg_cap = SimulationConfig(
            years=config.years,
            seed=seed_i,
            output_path=config.output_path,
            country_cap_enabled=True,
            debug=False,
            start_year=config.start_year,
        )

        # store sim objects so child_processor is accessible
        sim_unc = Simulation(cfg_unc)
        states_unc = sim_unc.run()
        sim_cap = Simulation(cfg_cap)
        states_cap = sim_cap.run()

        runs.append(_extract_run(states_unc, states_cap))
        cohort_runs.append(_extract_cohort_run(sim_unc, sim_cap))
        del sim_unc, sim_cap  # release queue/child objects immediately

        _progress_bar(i, n_runs, time.perf_counter() - loop_start, seed_i)

    print()
    total_elapsed = time.perf_counter() - loop_start
    print(f"\n  [OK] All {n_runs} runs complete in {_fmt_seconds(total_elapsed)} -- aggregating CIs...")

    years = runs[0]["years"]

    # Time-series CIs
    annual_aged_out_unc = _ts_ci(runs, "annual_aged_out_unc")
    annual_aged_out_cap = _ts_ci(runs, "annual_aged_out_cap")
    cumul_aged_out_unc = _ts_ci(runs, "cumul_aged_out_unc")
    cumul_aged_out_cap = _ts_ci(runs, "cumul_aged_out_cap")
    annual_backlog_total_unc = _ts_ci(runs, "annual_backlog_total_unc")
    annual_backlog_total_cap = _ts_ci(runs, "annual_backlog_total_cap")
    annual_conversions_unc = _ts_ci(runs, "annual_conversions_unc")
    annual_conversions_cap = _ts_ci(runs, "annual_conversions_cap")
    annual_visas_consumed_unc = _ts_ci(runs, "annual_visas_consumed_unc")
    annual_visas_consumed_cap = _ts_ci(runs, "annual_visas_consumed_cap")
    annual_exited_unc = _ts_ci(runs, "annual_exited_unc")
    annual_exited_cap = _ts_ci(runs, "annual_exited_cap")

    # Scalar CIs
    final_cumul_aged_out_unc = _sc_ci(runs, "final_cumul_aged_out_unc")
    final_cumul_aged_out_cap = _sc_ci(runs, "final_cumul_aged_out_cap")
    final_backlog_total_unc = _sc_ci(runs, "final_backlog_total_unc")
    final_backlog_total_cap = _sc_ci(runs, "final_backlog_total_cap")
    final_cumul_conversions_unc = _sc_ci(runs, "final_cumul_conversions_unc")
    final_cumul_conversions_cap = _sc_ci(runs, "final_cumul_conversions_cap")
    final_cumul_visas_unc = _sc_ci(runs, "final_cumul_visas_unc")
    final_cumul_visas_cap = _sc_ci(runs, "final_cumul_visas_cap")
    final_cumul_exited_unc = _sc_ci(runs, "final_cumul_exited_unc")
    final_cumul_exited_cap = _sc_ci(runs, "final_cumul_exited_cap")
    diff_cumul_aged_out = _sc_ci(runs, "diff_cumul_aged_out")
    diff_backlog_total = _sc_ci(runs, "diff_backlog_total")

    # Per-nationality
    annual_aged_out_by_nat_unc: Dict[str, TimeSeriesCI] = {}
    annual_aged_out_by_nat_cap: Dict[str, TimeSeriesCI] = {}
    annual_backlog_by_nat_unc: Dict[str, TimeSeriesCI] = {}
    annual_backlog_by_nat_cap: Dict[str, TimeSeriesCI] = {}
    annual_exited_by_nat_unc: Dict[str, TimeSeriesCI] = {}
    annual_exited_by_nat_cap: Dict[str, TimeSeriesCI] = {}
    final_backlog_by_nat_unc: Dict[str, ScalarCI] = {}
    final_backlog_by_nat_cap: Dict[str, ScalarCI] = {}
    final_aged_out_by_nat_unc: Dict[str, ScalarCI] = {}
    final_aged_out_by_nat_cap: Dict[str, ScalarCI] = {}
    final_ageout_pct_by_nat_unc: Dict[str, ScalarCI] = {}
    final_ageout_pct_by_nat_cap: Dict[str, ScalarCI] = {}
    final_exited_by_nat_unc: Dict[str, ScalarCI] = {}
    final_exited_by_nat_cap: Dict[str, ScalarCI] = {}
    diff_aged_out_by_nat: Dict[str, ScalarCI] = {}
    diff_ageout_pct_by_nat: Dict[str, ScalarCI] = {}
    diff_backlog_by_nat: Dict[str, ScalarCI] = {}
    diff_exited_by_nat: Dict[str, ScalarCI] = {}

    for nat in COUNTRIES:
        annual_aged_out_by_nat_unc[nat] = _ts_ci(runs, f"annual_aged_out_{nat}_unc")
        annual_aged_out_by_nat_cap[nat] = _ts_ci(runs, f"annual_aged_out_{nat}_cap")
        annual_backlog_by_nat_unc[nat] = _ts_ci(runs, f"annual_backlog_{nat}_unc")
        annual_backlog_by_nat_cap[nat] = _ts_ci(runs, f"annual_backlog_{nat}_cap")
        annual_exited_by_nat_unc[nat] = _ts_ci(runs, f"annual_exited_{nat}_unc")
        annual_exited_by_nat_cap[nat] = _ts_ci(runs, f"annual_exited_{nat}_cap")
        final_backlog_by_nat_unc[nat] = _sc_ci(runs, f"final_backlog_{nat}_unc")
        final_backlog_by_nat_cap[nat] = _sc_ci(runs, f"final_backlog_{nat}_cap")
        final_aged_out_by_nat_unc[nat] = _sc_ci(runs, f"final_aged_out_{nat}_unc")
        final_aged_out_by_nat_cap[nat] = _sc_ci(runs, f"final_aged_out_{nat}_cap")
        final_ageout_pct_by_nat_unc[nat] = _sc_ci(runs, f"final_ageout_pct_{nat}_unc")
        final_ageout_pct_by_nat_cap[nat] = _sc_ci(runs, f"final_ageout_pct_{nat}_cap")
        final_exited_by_nat_unc[nat] = _sc_ci(runs, f"final_exited_{nat}_unc")
        final_exited_by_nat_cap[nat] = _sc_ci(runs, f"final_exited_{nat}_cap")
        diff_aged_out_by_nat[nat] = _sc_ci(runs, f"diff_aged_out_{nat}")
        diff_ageout_pct_by_nat[nat] = _sc_ci(runs, f"diff_ageout_pct_{nat}")
        diff_backlog_by_nat[nat] = _sc_ci(runs, f"diff_backlog_{nat}")
        diff_exited_by_nat[nat] = _sc_ci(runs, f"diff_exited_{nat}")

    # Per-EB-category
    annual_aged_out_by_eb_unc: Dict[str, TimeSeriesCI] = {}
    annual_aged_out_by_eb_cap: Dict[str, TimeSeriesCI] = {}
    annual_exited_by_eb_unc: Dict[str, TimeSeriesCI] = {}
    annual_exited_by_eb_cap: Dict[str, TimeSeriesCI] = {}
    final_aged_out_by_eb_unc: Dict[str, ScalarCI] = {}
    final_aged_out_by_eb_cap: Dict[str, ScalarCI] = {}
    final_exited_by_eb_unc: Dict[str, ScalarCI] = {}
    final_exited_by_eb_cap: Dict[str, ScalarCI] = {}
    diff_aged_out_by_eb: Dict[str, ScalarCI] = {}
    diff_exited_by_eb: Dict[str, ScalarCI] = {}

    for eb_cat in EB_CATS:
        key = eb_cat.value.replace("-", "")
        annual_aged_out_by_eb_unc[key] = _ts_ci(runs, f"annual_aged_out_{key}_unc")
        annual_aged_out_by_eb_cap[key] = _ts_ci(runs, f"annual_aged_out_{key}_cap")
        annual_exited_by_eb_unc[key] = _ts_ci(runs, f"annual_exited_{key}_unc")
        annual_exited_by_eb_cap[key] = _ts_ci(runs, f"annual_exited_{key}_cap")
        final_aged_out_by_eb_unc[key] = _sc_ci(runs, f"final_aged_out_{key}_unc")
        final_aged_out_by_eb_cap[key] = _sc_ci(runs, f"final_aged_out_{key}_cap")
        final_exited_by_eb_unc[key] = _sc_ci(runs, f"final_exited_{key}_unc")
        final_exited_by_eb_cap[key] = _sc_ci(runs, f"final_exited_{key}_cap")
        diff_aged_out_by_eb[key] = _sc_ci(runs, f"diff_aged_out_{key}")
        diff_exited_by_eb[key] = _sc_ci(runs, f"diff_exited_{key}")

    ci_root = PROJECT_ROOT / "outputs" / "ci"
    save_raw_runs_to_csv(runs, ci_root, base_seed)
    save_raw_cohorts_to_csv(cohort_runs, ci_root, base_seed)

    print(f"  [OK] Aggregation complete.\n")

    return CIResults(
        n_runs=n_runs,
        years=years,
        annual_aged_out_unc=annual_aged_out_unc,
        annual_aged_out_cap=annual_aged_out_cap,
        cumul_aged_out_unc=cumul_aged_out_unc,
        cumul_aged_out_cap=cumul_aged_out_cap,
        annual_backlog_total_unc=annual_backlog_total_unc,
        annual_backlog_total_cap=annual_backlog_total_cap,
        annual_conversions_unc=annual_conversions_unc,
        annual_conversions_cap=annual_conversions_cap,
        annual_visas_consumed_unc=annual_visas_consumed_unc,
        annual_visas_consumed_cap=annual_visas_consumed_cap,
        annual_exited_unc=annual_exited_unc,
        annual_exited_cap=annual_exited_cap,
        annual_aged_out_by_nat_unc=annual_aged_out_by_nat_unc,
        annual_aged_out_by_nat_cap=annual_aged_out_by_nat_cap,
        annual_backlog_by_nat_unc=annual_backlog_by_nat_unc,
        annual_backlog_by_nat_cap=annual_backlog_by_nat_cap,
        annual_exited_by_nat_unc=annual_exited_by_nat_unc,
        annual_exited_by_nat_cap=annual_exited_by_nat_cap,
        annual_aged_out_by_eb_unc=annual_aged_out_by_eb_unc,
        annual_aged_out_by_eb_cap=annual_aged_out_by_eb_cap,
        annual_exited_by_eb_unc=annual_exited_by_eb_unc,
        annual_exited_by_eb_cap=annual_exited_by_eb_cap,
        final_cumul_aged_out_unc=final_cumul_aged_out_unc,
        final_cumul_aged_out_cap=final_cumul_aged_out_cap,
        final_backlog_total_unc=final_backlog_total_unc,
        final_backlog_total_cap=final_backlog_total_cap,
        final_cumul_conversions_unc=final_cumul_conversions_unc,
        final_cumul_conversions_cap=final_cumul_conversions_cap,
        final_cumul_visas_unc=final_cumul_visas_unc,
        final_cumul_visas_cap=final_cumul_visas_cap,
        final_cumul_exited_unc=final_cumul_exited_unc,
        final_cumul_exited_cap=final_cumul_exited_cap,
        diff_cumul_aged_out=diff_cumul_aged_out,
        diff_backlog_total=diff_backlog_total,
        final_backlog_by_nat_unc=final_backlog_by_nat_unc,
        final_backlog_by_nat_cap=final_backlog_by_nat_cap,
        final_aged_out_by_nat_unc=final_aged_out_by_nat_unc,
        final_aged_out_by_nat_cap=final_aged_out_by_nat_cap,
        final_ageout_pct_by_nat_unc=final_ageout_pct_by_nat_unc,
        final_ageout_pct_by_nat_cap=final_ageout_pct_by_nat_cap,
        final_exited_by_nat_unc=final_exited_by_nat_unc,
        final_exited_by_nat_cap=final_exited_by_nat_cap,
        diff_aged_out_by_nat=diff_aged_out_by_nat,
        diff_ageout_pct_by_nat=diff_ageout_pct_by_nat,
        diff_backlog_by_nat=diff_backlog_by_nat,
        diff_exited_by_nat=diff_exited_by_nat,
        final_aged_out_by_eb_unc=final_aged_out_by_eb_unc,
        final_aged_out_by_eb_cap=final_aged_out_by_eb_cap,
        final_exited_by_eb_unc=final_exited_by_eb_unc,
        final_exited_by_eb_cap=final_exited_by_eb_cap,
        diff_aged_out_by_eb=diff_aged_out_by_eb,
        diff_exited_by_eb=diff_exited_by_eb,
    )


# Console summary


def print_ci_summary(ci: CIResults) -> None:
    """
    Print a concise console summary of all headline CI results.

    Useful for a quick sanity check before running visualizations,
    and for embedding the key numbers in pipeline logs.
    """

    def fmt(sc: ScalarCI, decimals: int = 0) -> str:
        f = f"{{:,.{decimals}f}}"
        return f"{f.format(sc[0])}  [95% CI: {f.format(sc[1])} - {f.format(sc[2])}]"

    W = 72
    print("\n" + "=" * W)
    print(f"  CI SUMMARY  ({ci.n_runs} paired Monte Carlo runs, empirical percentile)")
    print("=" * W)

    print("\n-- Cumulative Children Aged Out (final year) ----------------------")
    print(f"  No Cap     : {fmt(ci.final_cumul_aged_out_unc)}")
    print(f"  7% Cap     : {fmt(ci.final_cumul_aged_out_cap)}")
    print(f"  Delta cap-unc  : {fmt(ci.diff_cumul_aged_out)}")

    print("\n-- By Nationality -- Aged-Out Count (final year) -------------------")
    for nat in COUNTRIES:
        print(f"  {nat:<8}  No Cap : {fmt(ci.final_aged_out_by_nat_unc[nat])}")
        print(f"            7% Cap : {fmt(ci.final_aged_out_by_nat_cap[nat])}")
        print(f"            Delta      : {fmt(ci.diff_aged_out_by_nat[nat])}")

    print("\n-- By Nationality -- Age-Out Percentage (final year) ---------------")
    for nat in COUNTRIES:
        print(f"  {nat:<8}  No Cap : {fmt(ci.final_ageout_pct_by_nat_unc[nat], 2)}%")
        print(f"            7% Cap : {fmt(ci.final_ageout_pct_by_nat_cap[nat], 2)}%")
        print(f"            Delta      : {fmt(ci.diff_ageout_pct_by_nat[nat], 2)} pp")

    print("\n-- By EB Category -- Aged-Out Count (cumulative) -------------------")
    for eb_cat in EB_CATS:
        key = eb_cat.value.replace("-", "")
        label = eb_cat.value  # "EB-1", "EB-2", ...
        print(f"  {label:<5}  No Cap : {fmt(ci.final_aged_out_by_eb_unc[key])}")
        print(f"         7% Cap : {fmt(ci.final_aged_out_by_eb_cap[key])}")
        print(f"         Delta      : {fmt(ci.diff_aged_out_by_eb[key])}")

    print("\n-- Queue Backlog -- Principals Only (final year) -------------------")
    print(f"  No Cap     : {fmt(ci.final_backlog_total_unc)}")
    print(f"  7% Cap     : {fmt(ci.final_backlog_total_cap)}")
    print(f"  Delta cap-unc  : {fmt(ci.diff_backlog_total)}")

    print("\n-- By Nationality -- Backlog (final year) ---------------------------")
    for nat in COUNTRIES:
        print(f"  {nat:<8}  No Cap : {fmt(ci.final_backlog_by_nat_unc[nat])}")
        print(f"            7% Cap : {fmt(ci.final_backlog_by_nat_cap[nat])}")
        print(f"            Delta      : {fmt(ci.diff_backlog_by_nat[nat])}")

    print("\n-- Conversions & Visa Consumption (final year, cumulative) --------")
    print(f"  Conversions  No Cap : {fmt(ci.final_cumul_conversions_unc)}")
    print(f"  Conversions  7% Cap : {fmt(ci.final_cumul_conversions_cap)}")
    print(f"  Visas issued No Cap : {fmt(ci.final_cumul_visas_unc)}")
    print(f"  Visas issued 7% Cap : {fmt(ci.final_cumul_visas_cap)}")
    print(f"  Queue exits  No Cap : {fmt(ci.final_cumul_exited_unc)}")
    print(f"  Queue exits  7% Cap : {fmt(ci.final_cumul_exited_cap)}")

    print("\n-- By Nationality -- Queue Exits (cumulative) -----------------------")
    for nat in COUNTRIES:
        print(f"  {nat:<8}  No Cap : {fmt(ci.final_exited_by_nat_unc[nat])}")
        print(f"            7% Cap : {fmt(ci.final_exited_by_nat_cap[nat])}")
        print(f"            Delta      : {fmt(ci.diff_exited_by_nat[nat])}")

    print("\n-- By EB Category -- Queue Exits (cumulative) -----------------------")
    for eb_cat in EB_CATS:
        key = eb_cat.value.replace("-", "")
        label = eb_cat.value
        print(f"  {label:<5}  No Cap : {fmt(ci.final_exited_by_eb_unc[key])}")
        print(f"         7% Cap : {fmt(ci.final_exited_by_eb_cap[key])}")
        print(f"         Delta      : {fmt(ci.diff_exited_by_eb[key])}")

    print("=" * W + "\n")


# CSV export for reproducible downstream analysis


def save_raw_runs_to_csv(runs: List[Dict], output_dir, base_seed: int) -> Dict[str, str]:
    """
    Persist the raw per-run results to two CSVs for offline CI reanalysis.

    ci_raw_scalars.csv
        One row per run. Columns: run_index, seed, then every final_* and
        diff_* scalar key from _extract_run.

    ci_raw_timeseries.csv
        One row per (run_index, year). Columns: run_index, year, then every
        annual_* and cumul_* array key from _extract_run.

    Args:
        runs:       Raw list of dicts from the Monte Carlo loop.
        output_dir: Path-like. Created if it does not exist.
        base_seed:  Seed used for run 0; run i used base_seed + i.

    Returns:
        Dict mapping label -> absolute file path string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    scalar_keys = sorted(k for k, v in runs[0].items() if isinstance(v, float))
    array_keys = sorted(k for k, v in runs[0].items() if isinstance(v, np.ndarray) and k != "years")

    # ci_raw_scalars.csv
    sc_path = output_dir / "ci_raw_scalars.csv"
    with open(sc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "seed"] + scalar_keys)
        for i, run in enumerate(runs):
            writer.writerow([i, base_seed + i] + [round(run[k], 6) for k in scalar_keys])
    files["ci_raw_scalars"] = str(sc_path)

    # ci_raw_timeseries.csv
    ts_path = output_dir / "ci_raw_timeseries.csv"
    years = runs[0]["years"]
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "year"] + array_keys)
        for i, run in enumerate(runs):
            for yr_idx, year in enumerate(years):
                writer.writerow([i, int(year)] + [round(float(run[k][yr_idx]), 6) for k in array_keys])
    files["ci_raw_timeseries"] = str(ts_path)

    print(f"  [OK] ci_raw_scalars.csv    ->  {sc_path}  ({len(runs)} rows)")
    print(f"  [OK] ci_raw_timeseries.csv ->  {ts_path}  ({len(runs) * len(years)} rows)")
    return files


def save_raw_cohorts_to_csv(
    cohort_runs: List[List[Dict]],
    output_dir,
    base_seed: int,
) -> str:
    """
    Persist per-run entry-year cohort outcomes to ci_raw_cohorts.csv.

    Structure mirrors ci_raw_timeseries.csv: one row per
    (run_index, scenario, entry_year, nationality, eb_category), enabling
    downstream groupby -> np.percentile([2.5, 97.5]) to produce cohort-level
    CIs for ageout_pct, save_pct, and exit_pct across N Monte Carlo runs.

    Columns
    -------
    run_index       : int   -- zero-based run index
    seed            : int   -- base_seed + run_index
    scenario        : str   -- "Uncapped" or "Capped"
    entry_year      : int   -- year parent entered the queue
    nationality     : str   -- e.g. "India", "China", "Other"
    eb_category     : str   -- e.g. "EB-1", "EB-2", "EB-3", "Unknown"
    total_created   : int   -- children created in this cohort
    aged_out        : int   -- children who aged out
    saved           : int   -- children whose parent converted
    exited          : int   -- children whose parent exited without converting
    still_dependent : int   -- children still in queue at simulation end
    ageout_pct      : float -- aged_out / total_created x 100
    save_pct        : float -- saved    / total_created x 100
    exit_pct        : float -- exited   / total_created x 100

    Args:
        cohort_runs: List (one element per run) of lists of record dicts
                     produced by _extract_cohort_run().
        output_dir:  Path-like. Created if it does not exist.
        base_seed:   Seed used for run 0; run i used base_seed + i.

    Returns:
        Absolute file path string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / "ci_raw_cohorts.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_index",
                "seed",
                "scenario",
                "entry_year",
                "nationality",
                "eb_category",  # NEW
                "total_created",
                "aged_out",
                "saved",
                "exited",
                "still_dependent",
                "ageout_pct",
                "save_pct",
                "exit_pct",
            ]
        )
        for i, records in enumerate(cohort_runs):
            for rec in records:
                writer.writerow(
                    [
                        i,
                        base_seed + i,
                        rec["scenario"],
                        rec["entry_year"],
                        rec["nationality"],
                        rec["eb_category"],  # NEW
                        rec["total_created"],
                        rec["aged_out"],
                        rec["saved"],
                        rec["exited"],
                        rec["still_dependent"],
                        rec["ageout_pct"],
                        rec["save_pct"],
                        rec["exit_pct"],
                    ]
                )

    total_rows = sum(len(r) for r in cohort_runs)
    print(f"  [OK] ci_raw_cohorts.csv    ->  {path}  ({total_rows} rows)")
    return str(path)


def save_ci_results_to_csv(ci: CIResults, output_dir) -> Dict[str, str]:
    """
    Persist CI results to two CSVs for reproducible downstream use.

    ci_timeseries.csv
        One row per year. Columns: mean, lo_2.5, hi_97.5 for every annual
        and cumulative time-series metric, including per-nationality and
        per-EB-category aged-out and queue-exit series.

    ci_scalars.csv
        One row per (metric, scenario). Columns: Mean, CI_Lo_2.5, CI_Hi_97.5.
        Covers final-state scalars and paired policy-difference CIs.

    Args:
        ci:         CIResults from run_ci_analysis()
        output_dir: Path-like. Created if it does not exist.

    Returns:
        Dict mapping label -> absolute file path string.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}

    # ci_timeseries.csv
    ts_path = output_dir / "ci_timeseries.csv"

    ts_headers = ["Year"]
    ts_series = [
        ("annual_aged_out_unc", ci.annual_aged_out_unc),
        ("annual_aged_out_cap", ci.annual_aged_out_cap),
        ("cumul_aged_out_unc", ci.cumul_aged_out_unc),
        ("cumul_aged_out_cap", ci.cumul_aged_out_cap),
        ("annual_backlog_total_unc", ci.annual_backlog_total_unc),
        ("annual_backlog_total_cap", ci.annual_backlog_total_cap),
        ("annual_conversions_unc", ci.annual_conversions_unc),
        ("annual_conversions_cap", ci.annual_conversions_cap),
        ("annual_visas_consumed_unc", ci.annual_visas_consumed_unc),
        ("annual_visas_consumed_cap", ci.annual_visas_consumed_cap),
        ("annual_exited_unc", ci.annual_exited_unc),
        ("annual_exited_cap", ci.annual_exited_cap),
    ]
    for nat in COUNTRIES:
        ts_series += [
            (f"annual_aged_out_{nat}_unc", ci.annual_aged_out_by_nat_unc[nat]),
            (f"annual_aged_out_{nat}_cap", ci.annual_aged_out_by_nat_cap[nat]),
            (f"annual_backlog_{nat}_unc", ci.annual_backlog_by_nat_unc[nat]),
            (f"annual_backlog_{nat}_cap", ci.annual_backlog_by_nat_cap[nat]),
            (f"annual_exited_{nat}_unc", ci.annual_exited_by_nat_unc[nat]),
            (f"annual_exited_{nat}_cap", ci.annual_exited_by_nat_cap[nat]),
        ]
    for eb_cat in EB_CATS:
        key = eb_cat.value.replace("-", "")
        ts_series += [
            (f"annual_aged_out_{key}_unc", ci.annual_aged_out_by_eb_unc[key]),
            (f"annual_aged_out_{key}_cap", ci.annual_aged_out_by_eb_cap[key]),
            (f"annual_exited_{key}_unc", ci.annual_exited_by_eb_unc[key]),
            (f"annual_exited_{key}_cap", ci.annual_exited_by_eb_cap[key]),
        ]

    for label, _ in ts_series:
        ts_headers += [f"{label}_mean", f"{label}_lo", f"{label}_hi"]

    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ts_headers)
        for idx, year in enumerate(ci.years):
            row = [int(year)]
            for _, (mean, lo, hi) in ts_series:
                row += [round(float(mean[idx]), 4), round(float(lo[idx]), 4), round(float(hi[idx]), 4)]
            writer.writerow(row)

    files["ci_timeseries"] = str(ts_path)

    # ci_scalars.csv
    sc_path = output_dir / "ci_scalars.csv"

    def sc_rows(label: str, unc: ScalarCI, cap: ScalarCI, diff: Optional[ScalarCI] = None, pct: bool = False) -> List[List]:
        suffix = "%" if pct else ""
        out = [
            [label, "Uncapped", f"{unc[0]:.4f}{suffix}", f"{unc[1]:.4f}{suffix}", f"{unc[2]:.4f}{suffix}"],
            [label, "Capped", f"{cap[0]:.4f}{suffix}", f"{cap[1]:.4f}{suffix}", f"{cap[2]:.4f}{suffix}"],
        ]
        if diff is not None:
            out.append([label, "Capped-Uncapped", f"{diff[0]:.4f}{suffix}", f"{diff[1]:.4f}{suffix}", f"{diff[2]:.4f}{suffix}"])
        return out

    with open(sc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Scenario", "Mean", "CI_Lo_2.5", "CI_Hi_97.5"])

        for row in sc_rows("cumulative_aged_out", ci.final_cumul_aged_out_unc, ci.final_cumul_aged_out_cap, ci.diff_cumul_aged_out):
            writer.writerow(row)

        for row in sc_rows("final_backlog_total", ci.final_backlog_total_unc, ci.final_backlog_total_cap, ci.diff_backlog_total):
            writer.writerow(row)

        for row in sc_rows("cumul_conversions", ci.final_cumul_conversions_unc, ci.final_cumul_conversions_cap):
            writer.writerow(row)

        for row in sc_rows("cumul_visas_consumed", ci.final_cumul_visas_unc, ci.final_cumul_visas_cap):
            writer.writerow(row)

        for row in sc_rows("cumul_queue_exits", ci.final_cumul_exited_unc, ci.final_cumul_exited_cap):
            writer.writerow(row)

        for nat in COUNTRIES:
            for row in sc_rows(f"aged_out_{nat}", ci.final_aged_out_by_nat_unc[nat], ci.final_aged_out_by_nat_cap[nat], ci.diff_aged_out_by_nat[nat]):
                writer.writerow(row)

            for row in sc_rows(f"ageout_pct_{nat}", ci.final_ageout_pct_by_nat_unc[nat], ci.final_ageout_pct_by_nat_cap[nat], ci.diff_ageout_pct_by_nat[nat], pct=True):
                writer.writerow(row)

            for row in sc_rows(f"backlog_{nat}", ci.final_backlog_by_nat_unc[nat], ci.final_backlog_by_nat_cap[nat], ci.diff_backlog_by_nat[nat]):
                writer.writerow(row)

            for row in sc_rows(f"exited_{nat}", ci.final_exited_by_nat_unc[nat], ci.final_exited_by_nat_cap[nat], ci.diff_exited_by_nat[nat]):
                writer.writerow(row)

        for eb_cat in EB_CATS:
            key = eb_cat.value.replace("-", "")
            for row in sc_rows(f"aged_out_{key}", ci.final_aged_out_by_eb_unc[key], ci.final_aged_out_by_eb_cap[key], ci.diff_aged_out_by_eb[key]):
                writer.writerow(row)

            for row in sc_rows(f"exited_{key}", ci.final_exited_by_eb_unc[key], ci.final_exited_by_eb_cap[key], ci.diff_exited_by_eb[key]):
                writer.writerow(row)

    files["ci_scalars"] = str(sc_path)

    print(f"  [OK] ci_timeseries.csv  ->  {ts_path}")
    print(f"  [OK] ci_scalars.csv     ->  {sc_path}")
    return files
