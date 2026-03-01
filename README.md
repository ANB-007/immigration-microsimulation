# Employment-Based Immigration Microsimulation

This repository implements a petitioner-level microsimulation of the U.S. employment-based (EB) immigration system. It reconstructs the EB green card queue from FY2009–FY2024 and projects outcomes through FY2040 under two policy regimes:

- **Capped scenario**: Current law with a 7% per-country cap.
- **Uncapped scenario**: Counterfactual that removes the per-country cap while holding total EB visas constant.

The model is designed to quantify how statutory rules—especially the per-country cap—redistribute wait times, age-out risk, and intergenerational exclusion across national-origin groups and EB preference categories.

---

## Key Features

- **Petitioner-level agent-based model** of EB-1 through EB-5 queues with nationality- and category-specific first-in-first-out queues.
- **Hybrid reconstruction–projection design**:
  - 2009–2024: Historical reconstruction calibrated to DOS/USCIS data with caps always enforced.
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
  - Yearly state snapshots, backlog decompositions, visa consumption by category–nationality cell, and age-out cohorts by entry year.

---

## Data Inputs

### IPUMS ACS Microdata (age_extractor.py, counts_extractor.py)

The empirical family and age distributions are estimated from IPUMS ACS microdata (FY2014–FY2024, skipping FY2020) using the following variables:

`YEAR, SAMPLE, SERIAL, CBSERIAL, HHWT, CLUSTER, STRATA, GQ, PERNUM, PERWT, MOMLOC, POPLOC, NCHILD, NCHLT5, RELATE, RELATED, SEX, AGE, MARST, BPL, BPLD, CITIZEN, YRNATUR, YRIMMIG, YRSUSA1, EDUC, EDUCD, CLASSWKR, CLASSWKRD, OCC, OCC2010, IND, WKSWORK2, UHRSWORK, INCWAGE`

These files are **not** included in the repository and must be obtained separately from IPUMS.

The extractors:

- `age_extractor.py`  
  Builds empirical principal age distributions and child entry-age distributions by EB-like category (EB-1, EB-2, EB-3, Other_EB4, EB-5) and nationality (India, China, ROW), and writes them to JSON for use by the simulation.

- `counts_extractor.py`  
  Derives nationality- and category-specific distributions for spouse presence (married vs. unmarried) and the number of foreign-born dependent children per married principal, and writes them to JSON for use by the simulation. 

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
  - Category–nationality first-in-first-out queues for EB-1 through EB-5 and {India, China, ROW}.
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
    - 2009–2024 always enforce caps for realism.
    - 2025+ toggles caps based on scenario configuration.
  - Rich time-series outputs for both capped and uncapped runs.

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
    - Visa consumption by category–nationality cell.
    - Queue exits by EB category and nationality.
    - Pass 2 allocations for oversubscribed countries.

### `backlog_analyzer.py`

Constructs **end-of-horizon backlog statistics** from completed simulation runs:

- `BacklogAnalysis` dataclass:
  - Principal-only backlogs by:
    - Nationality.
    - EB category.
    - Category–nationality pairs (e.g., EB-2 India).
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

### `metrics.py`

Computes **quantitative fit metrics** comparing model outputs to DOS data:

- Time-series diagnostics:
  - MAPE, RMSE, bias, correlation, turning-point accuracy.
- Aggregated metrics:
  - Overall fits, plus nationality- and category-specific performance.
- Includes a CLI entry point that:
  - Loads model-generated `visa_consumption.csv` and DOS reference data.
  - Writes a validation report to `results/validation_metrics_report.txt`.

### `plots.py`

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
python -m src.simulation --years 32 --seed 12345 --output outputs/ --debug
