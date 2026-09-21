# Saved paired simulations

`../current-results.json` selects `inflation-corrected-20260923/` and pins its manifest hash. This campaign uses the corrected annual-CPI demographic inputs and contains 55 paired simulations. There are 50 primary pairs, one pair per alternative age schedule, one historical replay pair and two follow-up pairs.

Each campaign contains `paired_runs.jsonl.gz`, `plan.json` and `manifest.json`. The manifest verifies the saved files and each paired record. The plan records the seeds, settings, source hashes and input hashes.

`parameters-inflation-corrected-20260923/` contains ten additional seed-2014 parameter pairs using the same corrected inputs and primary exit specification. Each scenario includes its age-out segmentation CSV and run metadata. Its manifest records file checksums and the matching main-campaign manifest. The parameter baseline matches the main seed-2014 pair exactly.

Earlier campaigns in this directory and in `all-category-age/` retain their original identities and checksums. They are preserved for comparison and are not selected for the current paper.

Use `python reproduce.py` to rebuild the publication from the selected main campaign. To summarize the published parameter sweep, run `python -m simulation.sensitivity --scenarios data/runs/parameters-inflation-corrected-20260923 --output outputs/sensitivity/parameter_summary.csv`.
