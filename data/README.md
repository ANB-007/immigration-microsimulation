# Data

All model inputs live here. No ACS download is needed to reproduce the paper.

| Location | Contents |
|---|---|
| `experiment.json` | Fixed experiment settings: 50 primary pairs, one per exit alternative, one historical replay pair and two follow-up pairs |
| `petitions/` | Historical petition totals and nationality distributions, with source notes |
| `demographics/` | Four cached demographic distributions, ACS extraction scripts and supporting family audits |
| `validation/` | Observed visa allocations and original sensitivity results |
| `current-results.json` | Selects the latest completed campaign and pins its manifest hash |
| `runs/inflation-corrected-20260923/` | Current 55 paired results with corrected annual-CPI demographic inputs |
| `runs/parameters-inflation-corrected-20260923/` | Ten current parameter-sensitivity pairs and their provenance |
| `runs/all-category-age/` | Earlier 82-pair campaign, preserved with its original inputs and checksums |
| `runs/` | Earlier saved campaign, retained with its original declaration and checksums |

Each campaign’s `paired_runs.jsonl.gz` is a compressed data file, with one case and paired simulation record per line. It contains numerical outputs and their original run identities; it is not a project archive or a second copy of the code. `reproduce.py` reads it directly, without extracting a duplicate tree. Its manifest verifies the file and each paired record.

The earlier saved records were produced before the directory reorganization. Their original identities remain intact. Moving the input tables and removing unused experiment options did not alter the current model's numerical behavior; equivalence was checked against the preceding implementation. New simulations record the current source and input hashes.

To regenerate demographic distributions, place the IPUMS ACS extract at `demographics/acs_2014-2024.csv` and run `python data/demographics/ages.py` and `python data/demographics/counts.py` from the repository root. These overwrite the four cached JSON files. The raw extract is ignored by Git.

The extractors share `demographics/income_adjustment.py`: nominal `INCWAGE` is
multiplied by September 2024 CPI-U divided by survey-year annual-average CPI-U,
using published index values without rounding the ratios. This assumes annual
ACS samples with unadjusted monetary variables. Missing income codes are excluded
from wage filters; unsupported survey years raise an error.

The four cached demographic distributions and selected simulation results now use the corrected annual-average survey-year CPI adjustment. Before applying the correction, the extraction reproduced all four previous distributions exactly. The demographic summary tables and family-omission percentages were refreshed from the corrected extraction.

The parameter sensitivity tests have been rerun with the same inputs and primary exit specification as the selected campaign. Their seed-2014 baseline matches the main campaign exactly. The historical sensitivity CSV in `validation/` remains an archive of the submitted results and is not used for the current appendix.
