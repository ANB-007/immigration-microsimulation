"""
Sensitivity / robustness analysis for the age-out microsimulation.

Aggregates cohort age-out outcomes (FY2025-FY2040) across a set of
parameter-varied scenarios and reports, for each scenario, capped vs uncapped
total age-outs, the capped-minus-uncapped differential, and the nationality
composition. This substantiates the paper's robustness claim that capped
scenarios produce MORE age-outs than the otherwise-identical uncapped scenario
across a wide range of structural parameterizations.

Each scenario is a full simulation run in which ONE structural assumption was
varied from the base (published) parameterization, with all cohort outcomes
exported to
    outputs/sensitivity/scenarios/<name>/children_aged_out_segmentation.csv

Parameter varied per scenario (see SCENARIOS below). To regenerate a scenario,
apply the corresponding change in simulation/empirical_params.py, re-run
    python -m simulation --years 32 --output outputs/sensitivity/scenarios/<name>
and then re-run this script.

Reads:  outputs/sensitivity/scenarios/<name>/children_aged_out_segmentation.csv
Writes: outputs/sensitivity/sensitivity_ageout_summary.csv
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCEN_DIR = PROJECT_ROOT / "outputs" / "sensitivity" / "scenarios"
OUT_CSV = PROJECT_ROOT / "outputs" / "sensitivity" / "sensitivity_ageout_summary.csv"

REPORT_WINDOW = (2025, 2040)  # projection period reported in the paper
EB_CATS = ["EB-1", "EB-2", "EB-3", "EB-4", "EB-5"]
NATIONALITIES = ["India", "China", "Other"]

# (directory name, human-readable label, parameter varied from the base case)
SCENARIOS = [
    ("base", "Base Case", "Published parameterization (no change)"),
    ("low_attrition_india", "Low India Attrition", "NATIONALITY_EMIGRATION_MULTIPLIERS['India'] decreased"),
    ("high_attrition_india", "High India Attrition", "NATIONALITY_EMIGRATION_MULTIPLIERS['India'] increased"),
    ("low_child_ages", "Younger Child Ages", "Derivative-child entry-age distribution shifted younger"),
    ("high_child_ages", "Older Child Ages", "Derivative-child entry-age distribution shifted older"),
    ("low_spouse_prob", "Low Spouse Probability", "Spouse-presence probability decreased (fewer children)"),
    ("high_spouse_prob", "High Spouse Probability", "Spouse-presence probability increased (more children)"),
    ("low_allocation", "Low Visa Allocation", "Total annual visa allocation decreased"),
    ("high_allocation", "High Visa Allocation", "Total annual visa allocation increased"),
    ("uniform_eb_exit_rates", "Uniform EB Exit Rates", "CATEGORY_EMIGRATION_MULTIPLIERS flattened to 1.0"),
]


def summarise(df_sub):
    """Total age-outs and nationality composition for one scenario+regime slice."""
    nat_mask = (df_sub["EB_Category"] == "Overall") & (df_sub["Nationality"].isin(NATIONALITIES))
    nat = df_sub[nat_mask].groupby("Nationality")["Count"].sum()
    eb_mask = (df_sub["Nationality"] == "Overall") & (df_sub["EB_Category"].isin(EB_CATS))
    eb = df_sub[eb_mask].groupby("EB_Category")["Count"].sum()
    out = {"total": int(sum(nat.get(n, 0) for n in NATIONALITIES))}
    for n in NATIONALITIES:
        out[n] = int(nat.get(n, 0))
    for c in EB_CATS:
        out[c] = int(eb.get(c, 0))
    return out


def load_scenario(dir_name):
    path = SCEN_DIR / dir_name / "children_aged_out_segmentation.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Normalize the rest-of-world label: the engine emits "ROW", some scenario
    # exports use "Other". Treat them as the same nationality group.
    df["Nationality"] = df["Nationality"].replace({"ROW": "Other"})
    return df[(df["Year"] >= REPORT_WINDOW[0]) & (df["Year"] <= REPORT_WINDOW[1])]


def _pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else 0.0


def main():
    rows, missing = [], []
    robustness_holds = True

    for dir_name, label, knob in SCENARIOS:
        df = load_scenario(dir_name)
        if df is None:
            missing.append(dir_name)
            continue
        cap = summarise(df[df["Scenario"] == "Capped"])
        unc = summarise(df[df["Scenario"] == "Uncapped"])
        diff = cap["total"] - unc["total"]
        if diff <= 0:
            robustness_holds = False
        row = {
            "scenario": label,
            "parameter_varied": knob,
            "capped_total": cap["total"],
            "uncapped_total": unc["total"],
            "differential_cap_minus_unc": diff,
            "capped_India_pct": _pct(cap["India"], cap["total"]),
            "uncapped_India_pct": _pct(unc["India"], unc["total"]),
        }
        for c in EB_CATS:
            row[f"capped_{c}"] = cap[c]
        rows.append(row)

    summary = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print("=" * 100)
    print("SENSITIVITY / ROBUSTNESS ANALYSIS  (cohort age-outs, FY2025-FY2040)")
    print("=" * 100)
    cols = [
        "scenario", "parameter_varied", "capped_total", "uncapped_total",
        "differential_cap_minus_unc", "capped_India_pct", "uncapped_India_pct",
    ]
    if not summary.empty:
        print(summary[cols].to_string(index=False))
    print()
    if missing:
        print(f"WARNING: {len(missing)} scenario dir(s) missing, skipped: {', '.join(missing)}")
    print(
        "Robustness check - capped age-outs exceed uncapped in EVERY scenario: "
        f"{'PASS' if robustness_holds and not summary.empty else 'FAIL'}"
    )
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
