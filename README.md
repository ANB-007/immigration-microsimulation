# Employment-Based Immigration Microsimulation

This repository contains the petitioner-level model used in *A Microsimulation of Age-Out Dynamics in United States Employment-Based Immigration*. It reconstructs the U.S. employment-based immigration queue from FY2009–FY2024, then projects visa allocations, backlogs and child age-outs through FY2040 under two policies:

- **Capped:** the modeled 7% per-country cap remains in place.
- **Uncapped:** country allowances are removed, holding total visa supply and category allocations fixed.

The purpose is to quantify how the cap changes age-out risk across nationalities and preference categories. Individual waiting times and age-outs are not directly validated; the estimates depend on the model's allocation, demographic and queue-exit assumptions.

## Results

The [main results](outputs/main/) and [sensitivity results](outputs/sensitivity/) can be read without running the model. Start with [metric_summary.csv](outputs/main/metric_summary.csv) for the primary estimates and [replay_historical_outcomes_summary.csv](outputs/sensitivity/replay_historical_outcomes_summary.csv) for the FY2009–FY2024 historical-allocation comparison. Manuscript and submission files are maintained separately from this code repository.

The latest completed experiment contains **55 capped/uncapped pairs** using the corrected annual-CPI demographic inputs. These comprise 50 primary pairs, one pair for each alternative age schedule, one historical-allocation replay pair and two cohort-follow-up pairs. Primary annual estimates cover FY2025–FY2040. The historical replay compares FY2009–FY2024 age-outs and the end-FY2024 backlog using seed 2014. The age-schedule comparisons also use seed 2014. The follow-up continues through FY2061 to resolve children entering through FY2040. Confidence intervals describe Monte Carlo precision in simulation means.

The five parameter sensitivity tests have also been rerun with the same corrected demographic inputs and primary exit specification. They comprise ten additional capped/uncapped pairs at seed 2014. The seven parameter-sensitivity tables in the appendix use these updated results.

The paper discusses an exploratory comparison with historical Visa Bulletin
dates and why it was not adopted as a waiting-time validation measure. The
comparison showed timing differences under the assumptions examined, but the
available data do not link applicant histories well enough to identify their
causes. Matching annual issuance totals does not establish accurate individual
waiting times.

The latest main campaign is selected by `data/current-results.json` and lives in `data/runs/inflation-corrected-20260923/`. The parameter-sensitivity records live in `data/runs/parameters-inflation-corrected-20260923/`. Earlier campaigns retain their original results and checksums. Aggregate demographic summaries and the family-omission audit are available in `outputs/demographics/`.

## Running the model

Clone the repository and use Python **3.10 or newer** on macOS or Linux. From its root:

```sh
python -m pip install -e .
python -m simulation --years 32 --output .local/exploratory
```

This runs a capped/uncapped comparison from FY2009 through FY2040. The output stays under the ignored `.local/` directory so an exploratory run cannot be mistaken for the curated results in `outputs/`.

The author's local checkout can rebuild the curated tables and figures from the saved paired runs without starting new simulations:

```sh
python reproduce.py --analysis-only
```

That authoring workflow requires the unversioned manuscript source because it verifies every table and narrative value against the document. It can also compile the manuscript and update the response's page references:

```sh
python reproduce.py
```

PDF compilation additionally requires Tectonic with its TeX packages cached, Pandoc, and the Times New Roman and Arial fonts. Generated numerical fragments and typesetting intermediates are kept out of Git.

To rerun the simulations:

```sh
python reproduce.py --run --workers 3
```

New results are saved separately in `.local/runs/`. Repeat the command after interruption to resume completed pairs. To use a different destination, add `--run-output /path/to/new-run`; to analyze an existing run, use `--records /path/to/run`. A full campaign takes several hours; rebuilding from saved records takes minutes.

After regenerating demographic inputs, the five parameter sensitivity tests
also need a new sweep, using the same primary exit specification:

```sh
python -m simulation.sensitivity_runner --dest .local/runs/inflation-corrected-parameters --quiet
python -m simulation.sensitivity --scenarios .local/runs/inflation-corrected-parameters --output .local/runs/inflation-corrected-parameters-summary.csv
```

This runs ten capped/uncapped pairs with seed 2014: one baseline and nine
parameter variants. It is separate from the main campaign. The published appendix already uses the corrected sweep. Future changes require a new sweep and a corresponding table refresh.

Use `reproduce.py` for the current paper. The lower-level `python -m simulation` interface retains the original model defaults and does not select the paper's exit specification automatically.

## Repository layout

```text
reproduce.py       private authoring build and paired-run campaign driver
simulation/        model and paired-run execution
  analysis/        tables, figures and Monte Carlo summaries used in the paper
data/              all model inputs, experiment settings and saved paired runs
outputs/           numerical result tables and figures
```

Start with [simulation/sim.py](simulation/sim.py) for the model or [data/experiment.json](data/experiment.json) for the experiment settings.

The singular `output/` path is not used. Curated public results live in `outputs/`; local runs and build products live under `.local/`. Manuscripts, submission files, tests, previews, superseded experiments and development checks are retained locally and ignored by Git.

## Data availability

All inputs needed for replication are supplied in [data/](data/README.md). Petition tables are in `data/petitions/`; demographic distributions and their extraction scripts are in `data/demographics/`. No input datasets are stored inside `simulation/`.

Each saved campaign contains a compressed JSON-lines data file, its experiment declaration and checksums. `data/current-results.json` selects the latest completed campaign in `data/runs/all-category-age/`; the bundle directly under `data/runs/` is an earlier campaign. It contains simulation results and provenance, not another copy of the project. Compression keeps more than 1 GB of numerical records to approximately 36 MB. These records preserve the earlier results unchanged; the code reorganization does not reassign them a new source identity.

The raw ACS microdata are not redistributed. To regenerate demographic distributions, obtain the IPUMS USA ACS extract for 2014–2019 and 2021–2024 with these variables:

```text
YEAR SAMPLE SERIAL CBSERIAL HHWT CLUSTER STRATA GQ PERNUM PERWT
MOMLOC POPLOC NCHILD NCHLT5 RELATE RELATED SEX AGE MARST BPL BPLD
CITIZEN YRNATUR YRIMMIG YRSUSA1 EDUC EDUCD CLASSWKR CLASSWKRD
OCC OCC2010 IND WKSWORK2 UHRSWORK INCWAGE
```

Save it as `data/demographics/acs_2014-2024.csv`, then run `python data/demographics/ages.py` and `python data/demographics/counts.py`. These overwrite the four cached distribution files. The raw extract is ignored by Git.

## Citation

If you use this microsimulation, its outputs or this code, please cite: Balamurugan, A. N. (2026). *A Microsimulation of Age-Out Dynamics in United States Employment-Based Immigration.* SSRN. [doi:10.2139/ssrn.5841103](https://doi.org/10.2139/ssrn.5841103). To cite the software, use [CITATION.cff](CITATION.cff). The [previous software release](https://doi.org/10.5281/zenodo.21465739) predates this revision. If the article is published in the *International Journal of Microsimulation*, please cite the published version instead.

## License

Released under the MIT License; see [LICENSE](LICENSE). Copyright (c) 2026 Adhithiya Narayanan Balamurugan.
