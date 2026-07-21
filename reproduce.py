#!/usr/bin/env python3
"""
One-command reproduction driver for the EB immigration microsimulation.

Runs the analysis pipeline behind the paper. Subcommands:

    standard     FY2009-FY2040 paired capped/uncapped run
                 -> outputs/  (core figures + datasets)
    validation   model-vs-DOS fit metrics + validation figures
                 -> validation/results/
    cohorts      FY2061 cohort-resolution run + entry-year cohort density figures
                 -> outputs/cohorts/
    montecarlo   N paired Monte Carlo runs + 95% CI summary stats + CI figures
                 -> outputs/ci/
    sensitivity  regenerate ceteris-paribus scenarios + robustness table
                 -> outputs/sensitivity/
    all          run standard, validation, cohorts, montecarlo, sensitivity in order

Examples:
    python reproduce.py all
    python reproduce.py montecarlo --ci-runs 50
    python reproduce.py sensitivity
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd, cwd=ROOT):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        sys.exit(f"\n[reproduce] step failed ({result.returncode}): {' '.join(cmd)}")


def standard(args):
    run([PY, "-m", "simulation", "--years", "32", "--seed", "2014", "--quiet", "--output", "outputs/"])


def validation(args):
    # Compares the model's visa issuances against official DOS data (both shipped
    # under validation/). Writes results/validation_metrics_report.txt and the
    # per-nationality / per-category fit figures. Scripts use cwd-relative paths.
    val = ROOT / "validation"
    run([PY, "metrics.py"], cwd=val)
    run([PY, "plots.py"], cwd=val)


def cohorts(args):
    run([PY, "-m", "simulation", "--years", "53", "--seed", "2014", "--quiet",
         "--output", "outputs/cohorts"])
    run([PY, "simulation/densities.py"])


def montecarlo(args):
    # ci_figures.py also reads outputs/cohorts/children_aged_out_segmentation.csv,
    # which is produced by the `cohorts` stage. `all` runs `cohorts` first; if you
    # run `montecarlo` on its own, run `cohorts` first (or keep the shipped file).
    run([PY, "-m", "simulation", "--years", "32", "--seed", "2014", "--quiet",
         "--ci", "--ci-runs", str(args.ci_runs), "--output", "outputs/montecarlo"])
    run([PY, "simulation/mc_calc.py"])
    run([PY, "simulation/ci_figures.py"])


def sensitivity(args):
    run([PY, "-m", "simulation.sensitivity_runner", "--quiet"])
    run([PY, "simulation/sensitivity.py"])


def all_steps(args):
    standard(args)
    validation(args)
    cohorts(args)
    montecarlo(args)
    sensitivity(args)


STEPS = {
    "standard": standard,
    "validation": validation,
    "cohorts": cohorts,
    "montecarlo": montecarlo,
    "sensitivity": sensitivity,
    "all": all_steps,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=list(STEPS), help="Which part of the pipeline to run.")
    parser.add_argument("--ci-runs", type=int, default=50,
                        help="Monte Carlo paired runs for `montecarlo`/`all` (default: 50).")
    args = parser.parse_args()
    STEPS[args.step](args)
    print("\n[reproduce] done.")


if __name__ == "__main__":
    main()
