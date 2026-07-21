# Employment-Based Immigration Microsimulation

This repository implements a petitioner-level microsimulation of the U.S. employment-based (EB) immigration system. It reconstructs the EB green card queue from FY2009-FY2024 and projects outcomes through FY2040 under two policy regimes:

- **Capped scenario**: Current law with a 7% per-country cap.
- **Uncapped scenario**: Counterfactual that removes the per-country cap while holding total EB visas constant.

The model is designed to quantify how statutory rules—especially the per-country cap—redistribute wait times, age-out risk, and intergenerational exclusion across national-origin groups and EB preference categories.

---

## Quick Start

```bash
git clone https://github.com/ANB-007/immigration-microsimulation.git
cd immigration-microsimulation
pip install .                 # numpy, pandas, matplotlib, seaborn (pinned; Python >= 3.10)
python reproduce.py all       # regenerate every figure, table, and dataset behind the paper
```

The repository **ships all precomputed outputs under `outputs/`**, so every result can be inspected immediately without re-running anything. `reproduce.py all` regenerates them; see [Reproducing the Paper](#reproducing-the-paper-replication-package) for individual stages and runtimes. No external data download is required (see [Data Availability](#data-availability)).

## Repository Layout

```
simulation/      core microsimulation engine + Monte Carlo / cohort / sensitivity modules
validation/      model-vs-DOS validation (metrics.py, plots.py); real_data/ + model_data/ inputs
demographics/    IPUMS ACS extractors (ages/, counts/) that build the cached distributions
outputs/         precomputed paper outputs: figures, tables, datasets
reproduce.py     one-command reproduction driver
pyproject.toml   package metadata + pinned dependencies
CITATION.cff     citation metadata
LICENSE          MIT
```

---

## Key Features

- **Petitioner-level agent-based model** of EB-1 through EB-5 queues with nationality- and category-specific first-in-first-out queues.
- **Hybrid reconstruction-projection design**:
  - 2009-2024: Historical reconstruction calibrated to DOS/USCIS data with caps always enforced.
  - 2025+: Scenario-specific projections with capped vs. uncapped allocation.
- **End-to-end family modeling**:
  - Empirical spouse presence and child count distributions by nationality and EB category.
  - Child entry-age distributions and full dependent lifecycle (saved, aged out, exited).
- **Visa allocation engine**:
  - Category caps (28.6% for EB-1/2/3, 7.1% for EB-4/5).
  - 7% per-country ceilings, cascade spillovers, and historical demand constraints.
- **Validation against DOS data**:
  - Visa issuance by nationality and EB category, with quantitative fit metrics and comparison plots.
- **Rich outputs for analysis**:
  - Yearly state snapshots, backlog decompositions, visa consumption by category-nationality cell, and age-out cohorts by entry year.

---

## Data Availability

**Everything required to reproduce the paper ships with this repository—no downloads are needed.** External microdata is required only to rebuild the cached empirical distributions from source (the optional step at the end).

### Shipped with the repository

| Data | Location | Source |
|---|---|---|
| Cached empirical distributions (principal age, child entry-age, spouse presence, child counts) | `simulation/distributions/*.json` | Estimated from IPUMS ACS (below); these are the inputs the model consumes |
| DOS employment-based visa issuances, FY2009-FY2024 | `validation/real_data/dos_visa_consumption_data.csv` | U.S. Department of State, *Report of the Visa Office* (`dos_visaoffice2024`) |
| USCIS petition approvals: I-140 (EB-1/2/3), I-360 (EB-4), I-526/I-526E (EB-5) | `validation/real_data/<year>/*.pdf` | USCIS employment-based petition statistics (`uscisEB5`, `uscis_data`) |
| Model visa issuances used for validation | `validation/model_data/visa_consumption.csv` | This model (the paper's run) |
| Precomputed paper outputs (all figures, tables, datasets) | `outputs/` | This model |

Because the ACS-derived distributions are cached under `simulation/distributions/`, the simulation runs end-to-end **without any IPUMS download**. Full bibliographic citations for the DOS and USCIS sources are in the paper's References.

### Optional: regenerating the empirical distributions from IPUMS ACS

The distributions in `simulation/distributions/` were estimated from **IPUMS USA American Community Survey (ACS)** microdata. This is needed only to rebuild them from scratch—not to reproduce any published result.

1. At <https://usa.ipums.org/usa/>, create a free account and build an extract of the **ACS 1-year samples for 2014-2024, excluding 2020**, with these variables:

   `YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, STRATA, GQ, PERNUM, PERWT, MOMLOC, POPLOC, NCHILD, NCHLT5, RELATE, RELATED, SEX, AGE, MARST, BPL, BPLD, CITIZEN, YRNATUR, YRIMMIG, YRSUSA1, EDUC, EDUCD, CLASSWKR, CLASSWKRD, OCC, OCC2010, IND, WKSWORK2, UHRSWORK, INCWAGE`

2. Save the extract as `demographics/acs_2014-2024.csv` (several GB; **gitignored**—it is never committed and must not be included in a release archive).

3. Rebuild the four distribution JSONs, then copy them into `simulation/distributions/` (each script reads `../acs_2014-2024.csv` and writes its JSON next to itself):

   ```bash
   cd demographics/ages   && python age_extractor.py      # principal-age + child-entry-age distributions
   cd ../counts           && python counts_extractor.py   # spouse-presence + child-count distributions
   ```

---

## Core Modules

This section documents the main modules a reader or collaborator needs to understand to run and extend the model.

### `models.py`

Defines the **core data models** for the simulation:

- `WorkerStatus`: Temporary, permanent, and exited statuses.
- `EBCategory`: Enum for EB-1 through EB-5.
- `Worker`: Principal applicant in the EB queue, including:
  - Status, nationality, EB category, queue entry date, and spouse count.
  - Utility methods for status transitions and computing family visa cost.
- `DependentChild`, `AgedOutChild`, `SavedChild`, `ExitedChild`:
  - Typed records for child lifecycle states and outcomes, including timing and parental queue tenure.

### `empirical_params.py`

Centralizes **all empirical and statutory parameters** used by the simulation, including:

- EB visa pool:
  - Historical EB visa totals by fiscal year.
  - Statutory cap and spillover assumptions.
- Per-country caps and category shares:
  - 7% per-country limit.
  - Base shares for EB-1 through EB-5 and cascade flow structure.
- Nationality groups:
  - Aggregated into `India`, `China`, and `ROW`.
- Family structure:
  - Age-out age (21).
  - Spouse presence distributions.
  - Child count and child entry-age distributions by nationality and category.
- Queue exit parameters:
  - Tenure-, nationality-, and category-specific exit rates.

### `child_processor.py`

Implements the **child lifecycle engine**:

- Creates dependent children when a temporary worker enters the queue, using empirical distributions for:
  - Number of children (conditional on nationality and marital status).
  - Entry age (conditional on nationality and pathway).
- Tracks all children across four states:
  1. Dependent (under 21, parent still in queue).
  2. Saved (parent converts before child ages out).
  3. Aged out (child turns 21 while parent still waiting).
  4. Exited (parent leaves the queue without converting).
- Provides:
  - Annual processing of age-outs.
  - Batch removal of children when parents convert or exit.
  - Cohort-level outcome summaries by entry year, nationality, and EB category.
  - Aggregate statistics needed for analysis and visualization (e.g., age-out percentages by nationality).

### `sim.py`

Implements the **main microsimulation engine**:

- Initializes:
  - RNG streams for worker creation, queue exits, child generation, and visa allocation.
  - Category-nationality first-in-first-out queues for EB-1 through EB-5 and {India, China, ROW}.
- Annual step logic:
  1. Add new workers by pathway (using `get_pathway_totals` and nationality distributions).
  2. Allocate visas via `VisaProcessor` (capped or uncapped, depending on period and scenario).
  3. Save children of converted parents.
  4. Process queue exits (tenure-, age-, nationality-, and category-dependent).
  5. Update family-adjusted visa consumption by category.
  6. Age workers and children; process new age-outs at 21.
  7. Aggregate statistics into a `SimulationState` snapshot.
- Supports:
  - A **hybrid reconstruction-projection** regime:
    - 2009-2024 always enforce caps for realism.
    - 2025+ toggles caps based on scenario configuration.
  - Rich time-series outputs for both capped and uncapped runs.

### `visa_processor.py`

Implements the **visa allocation engine** invoked each simulated year:

- Applies statutory category shares (28.6% for EB-1/2/3, 7.1% for EB-4/5) and the 7% per-country ceiling.
- Runs the two-pass allocation: a first per-country-limited pass, then a spillover/cascade pass (EB-4/5 → EB-1 → EB-2 → EB-3) that redistributes unused numbers, honoring historical demand constraints during the reconstruction period.
- Returns per category-nationality allocations that `sim.py` uses to convert principals and their families.

### `states.py`

Defines typed **configuration and state containers**:

- `SimulationConfig`:
  - Years to simulate, seed, start year, output path, and whether caps are enabled in the projection period.
- `SimulationState`:
  - Per-year snapshot including:
    - Worker counts by status.
    - New entries and conversions.
    - Children aged out, saved, and at risk.
    - Backlogs by EB category and nationality.
    - Visa consumption by category-nationality cell.
    - Queue exits by EB category and nationality.
    - Pass 2 allocations for oversubscribed countries.

### `backlog_analyzer.py`

Constructs **end-of-horizon backlog statistics** from completed simulation runs:

- `BacklogAnalysis` dataclass:
  - Principal-only backlogs by:
    - Nationality.
    - EB category.
    - Category-nationality pairs (e.g., EB-2 India).
  - Family-adjusted backlogs (principals + spouses + children) with the same breakdowns.

### `visa_consumption_exporter.py`

Exports **visa consumption statistics** to CSV:

- Aggregates visas used by principals, spouses, and children by:
  - Year,
  - Scenario (capped vs. uncapped),
  - EB category,
  - Nationality.
- Produces a tidy `visa_consumption.csv` that supports external validation and downstream statistics.

### `visualization.py`

Creates **publication-quality plots** from simulation outputs:

- Age-outs:
  - By nationality and EB category.
  - Annual and cumulative comparisons across capped and uncapped regimes.
- EB conversions:
  - Category-specific conversion time series for both scenarios.
- Applicant type composition:
  - Principals vs. spouses vs. children over time.

### `validation/metrics.py`

Computes **quantitative fit metrics** comparing model outputs to DOS data:

- Time-series diagnostics:
  - MAPE, RMSE, bias, correlation, turning-point accuracy.
- Aggregated metrics:
  - Overall fits, plus nationality- and category-specific performance.
- Includes a CLI entry point that:
  - Loads model-generated `visa_consumption.csv` and DOS reference data.
  - Writes a validation report to `results/validation_metrics_report.txt`.

### `validation/plots.py`

Generates **visual validation plots** comparing simulated and actual DOS series:

- For each nationality (India, China, ROW):
  - Line plots of model vs. DOS total EB visa issuances over time.
- For each EB category (EB-1 through EB-5):
  - Line plots comparing modeled and actual category-specific issuances.

Outputs PNGs into `results/nationality/` and `results/category/` to support visual inspection of model fit.

### `__main__.py`

Provides the **command-line entry point**:

- Parses arguments such as:
  - `--reconstruction` (historical period only).
  - `--years N` (length of reconstruction + projection years, inclusive of current year).
  - `--seed` (for reproducibility; default is 2014).
  - `--output` (output directory).
  - `--debug` (verbose logging).
- Constructs a `SimulationConfig` and runs:
  - Paired capped vs. uncapped simulations.
  - Backlog analysis, CSV exports, and core visualizations.
- Usage (from the project root) could look like:

```bash
python -m simulation --years 32 --seed 12345 --output outputs/ --debug
```

Add `--ci` (with optional `--ci-runs N`, default 50) to append a Monte Carlo confidence-interval pass after the standard run; see the Monte Carlo modules and the Replication section below.

---

## Monte Carlo, Cohort, and Sensitivity Modules

These modules implement the Monte Carlo confidence intervals, the long-horizon (FY2061) cohort-resolution run, and the robustness analysis reported in the paper.

### `ci_runner.py`

Monte Carlo **confidence-interval engine**. Runs `N` paired *(uncapped, capped)* simulations using common random numbers—each pair `i` draws from the same seed (`base_seed + i`), so the capped-uncapped difference is a tight matched-pairs estimate—and aggregates **empirical 2.5th-97.5th percentile** confidence intervals (no normality assumption) for every headline metric by year, nationality, and EB category. Invoked via `python -m simulation --ci --ci-runs N`. Writes per-run raw data (`ci_raw_scalars.csv`, `ci_raw_timeseries.csv`, `ci_raw_cohorts.csv`) and aggregated CIs (`ci_scalars.csv`, `ci_timeseries.csv`) to `outputs/ci/`.

### `mc_calc.py`

Reads `outputs/ci/ci_raw_timeseries.csv` and computes **publication summary statistics** (means with 95% CIs) for backlog, visas, queue exits, and age-outs across the reconstruction (2009-2024), projection (2025-2040), and full (2009-2040) windows, disaggregated by nationality and EB category. Writes `ci_summary_stats.csv` and `ci_annual_table.csv`.

### `ci_figures.py`

Renders the **CI-backed policy figures** (annual and cumulative age-outs with confidence bands, totals, capped-uncapped differential, backlog, EB-category) from `outputs/ci/` into `outputs/ci/policy_figures/`, including `annual_age_outs_by_scenario.png` used in the paper. It also reads `outputs/cohorts/children_aged_out_segmentation.csv` (produced by the `cohorts` stage), so run `cohorts` before `montecarlo` when running stages individually; `reproduce.py all` already orders them correctly.

### `densities.py`

100% stacked-area **cohort-outcome density charts** (Saved / Aged Out / Exited / Still Dependent) by entry year, computed from the long-horizon (FY2061) run and clipped to entry cohorts ≤ FY2040. Reads `outputs/cohorts/outcomes_by_entry_year.csv`, writes `outputs/cohorts/entry_year_density/`. Produces the per-category-nationality figures (e.g., `eb2_india_capped.png`) used in the paper. The FY2061 horizon lets every cohort present through FY2040 fully resolve (age out or convert) before its outcome shares are tallied.

### `sensitivity.py`

**Robustness analysis.** Aggregates cohort age-outs (FY2025-2040) across parameter-varied scenarios—India attrition, child entry ages, spouse probability, visa allocation, and uniform EB exit rates—stored under `outputs/sensitivity/scenarios/`, and reports capped vs. uncapped totals, the differential, and nationality composition per scenario. Confirms capped age-outs exceed uncapped in **every** scenario. Writes `outputs/sensitivity/sensitivity_ageout_summary.csv`.

### `sensitivity_runner.py`

**Ceteris-paribus scenario generator.** Regenerates each sensitivity scenario by varying **exactly one** structural parameter from the published baseline while holding all others at their standard state, exporting each scenario's cohort segmentation for `sensitivity.py` to aggregate. Each scenario is a paired (uncapped, capped) run at seed 2014, so the only difference within a pair is the per-country cap and the only difference across scenarios is the one varied parameter. Overrides are applied to live `empirical_params` values and restored after each run: India emigration multiplier (0.4 / 1.0), EB multipliers flattened to 1.0, spouse-probability multiplier (0.8 / 1.2), child entry-age multiplier (0.8 / 1.2), and projection visa supply (140,000 / 200,000). Run with `python -m simulation.sensitivity_runner`.

---

## Reproducing the Paper (Replication Package)

### Environment

```bash
pip install .        # numpy, pandas, matplotlib, seaborn (versions pinned in pyproject.toml)
# or, for a live checkout:  pip install -e .
```

Developed and validated with Python 3.10.18; the pinned dependencies require Python >= 3.10. Dependencies and their pinned versions are declared in `pyproject.toml`; IPUMS ACS microdata are **not** needed to reproduce the paper outputs—the estimated empirical distributions are cached under `simulation/distributions/`.

### One-command reproduction

The `reproduce.py` driver runs the whole pipeline (or any single stage) with one command:

```bash
python reproduce.py all            # standard + validation + cohorts + montecarlo + sensitivity
python reproduce.py standard       # FY2009-FY2040 paired run           -> outputs/
python reproduce.py validation     # model-vs-DOS fit metrics + figures -> validation/results/
python reproduce.py cohorts        # FY2061 cohort-resolution run       -> outputs/cohorts/
python reproduce.py montecarlo     # 50 paired MC runs + CI stats/figs  -> outputs/ci/
python reproduce.py sensitivity    # regenerate scenarios + robustness  -> outputs/sensitivity/
python reproduce.py montecarlo --ci-runs 100   # more runs -> tighter CI bands
```

Each stage is just a thin wrapper over the underlying commands, which can also be run directly:

```bash
# standard
python -m simulation --years 32 --seed 2014 --quiet --output outputs/
# validation: model-vs-DOS fit metrics + figures (scripts use paths relative to validation/)
cd validation && python metrics.py && python plots.py && cd ..
# cohort-resolution run to FY2061 (lets cohorts through FY2040 fully resolve), then density figures
python -m simulation --years 53 --seed 2014 --quiet --output outputs/cohorts
python simulation/densities.py
# Monte Carlo 50 paired runs, then CI summary stats and CI figures
python -m simulation --years 32 --seed 2014 --quiet --ci --ci-runs 50 --output outputs/montecarlo
python simulation/mc_calc.py
python simulation/ci_figures.py
# sensitivity: regenerate ceteris-paribus scenarios, then the robustness table
python -m simulation.sensitivity_runner --quiet
python simulation/sensitivity.py
```

**Runtime.** The model is compute-intensive in the projection years (the queue exceeds ~1.5M by the 2030s): a single 32-year paired run takes several minutes; `montecarlo` (50 runs) and `sensitivity` (10 scenarios) can each take 1-2 hours on a laptop. **The repository ships the exact outputs behind the paper under `outputs/`, so every figure and table can be inspected without re-running.** Regenerating from the clean engine reproduces the published findings within Monte-Carlo / seed tolerance (roughly +/-1% on totals at seed 2014).

### Output layout

```
outputs/
  age_outs/, conversions/       standard-run figures
  *.csv                         standard-run datasets (states, backlog, outcomes, ...)
  ci/                           Monte Carlo confidence-interval artifacts
    ci_raw_timeseries.csv         per-run annual metrics, 50 runs (paper data)
    ci_raw_scalars.csv            per-run end-state scalars, 50 runs
    ci_timeseries.csv             aggregated per-year 95% CIs
    ci_summary_stats.csv          headline means + 95% CIs (mc_calc)
    ci_annual_table.csv           annual age-outs table with CIs
    policy_figures/               CI-backed figures (incl. annual_age_outs_by_scenario.png)
  cohorts/                      cohort-resolution run (FY2061 horizon)
    outcomes_by_entry_year.csv
    entry_year_density/           cohort-outcome density figures
  sensitivity/
    scenarios/<name>/             parameter-varied scenario outputs
    sensitivity_ageout_summary.csv
```

### Which command produces each paper figure

| Paper figure(s) | `reproduce.py` stage (underlying script) |
|---|---|
| `india/china/row_validation.png` | `validation` (`validation/plots.py`) |
| `applicant_type_{capped,uncapped}.png`, `{capped,uncapped}_by_nationality.png` | `standard` (`visualization.py`) |
| `annual_age_outs_by_scenario.png` | `montecarlo` (`ci_figures.py`) |
| `eb2_india`, `eb3_india`, `eb2_china`, `eb4_other_{capped,uncapped}.png` | `cohorts` (`densities.py`) |

### Note on Monte Carlo provenance

The 95% intervals reported in the paper are the 2.5th-97.5th percentiles across **50 paired Monte Carlo runs**. `outputs/ci/ci_raw_timeseries.csv` is the exact 50-run set underlying the paper; regenerating via step 3 above produces statistically equivalent results.

---

## Citation

If you use this microsimulation, its outputs, or this code, please cite the accompanying SSRN preprint: Balamurugan, A. N. (2026). *A Microsimulation of Age-Out Dynamics in United States Employment-Based Immigration.* SSRN. [doi:10.2139/ssrn.5841103](https://doi.org/10.2139/ssrn.5841103). To cite the software specifically, use the metadata in [`CITATION.cff`](CITATION.cff) (archived at [doi:10.5281/zenodo.21465739](https://doi.org/10.5281/zenodo.21465739)). If the article is published in the *International Journal of Microsimulation*, please cite the published version instead.

## License

Released under the MIT License—see [`LICENSE`](LICENSE). Copyright (c) 2026 Adhithiya Narayanan Balamurugan.
