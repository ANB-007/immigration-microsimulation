import pandas as pd
import numpy as np
import os
import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from simulation.result_schema import normalize_nationality_columns, check_annual_exit_partitions
EXPLORATORY_OUTPUT = Path.cwd() / "outputs" / "exploratory"
FILEPATH = str(EXPLORATORY_OUTPUT / "ci" / "ci_raw_timeseries.csv")
OUTPATH = str(EXPLORATORY_OUTPUT / "ci" / "ci_summary_stats.csv")


def compute_all_stats(filepath):
    df = normalize_nationality_columns(pd.read_csv(filepath))
    check_annual_exit_partitions(df)
    if df.duplicated(["run_index", "year"]).any():
        raise ValueError("Duplicate run/year rows")
    expected_years = set(range(2009, 2041))
    if any(set(group["year"]) != expected_years for _, group in df.groupby("run_index")):
        raise ValueError("Paper summaries require complete FY2009–FY2040 runs")

    # Current-schema subgroup exits are annual flows. Legacy archived exports
    # were first-differenced incorrectly and require an explicit migration;
    # the accounting check above rejects them rather than silently changing data.

    RECON = (2009, 2024)
    PROJ = (2025, 2040)
    FULL = (2009, 2040)

    def period(df_sub, y0, y1):
        return df_sub[(df_sub["year"] >= y0) & (df_sub["year"] <= y1)]

    def pct(series, label):
        return {
            "label": label,
            "mean": round(series.mean(), 1),
            "lo": round(np.percentile(series, 2.5), 1),
            "hi": round(np.percentile(series, 97.5), 1),
        }

    results = {}

    # 1. Final backlog at FY2040
    last = df[df["year"] == 2040]
    results["backlog_unc"] = pct(last["annual_backlog_total_unc"], "Final_Backlog_FY2040_Uncapped")
    results["backlog_cap"] = pct(last["annual_backlog_total_cap"], "Final_Backlog_FY2040_Capped")
    diff_bl = last["annual_backlog_total_cap"].values - last["annual_backlog_total_unc"].values
    results["backlog_diff"] = pct(pd.Series(diff_bl), "Final_Backlog_FY2040_Cap_minus_Unc")

    # 2. Cumulative visas consumed 2009-2040
    for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
        s = df.groupby("run_index")[f"annual_visas_consumed_{sc}"].sum()
        results[f"visas_{sc}"] = pct(s, f"Cumulative_Visas_Consumed_2009_2040_{tag}")

    # 2b. Total cumulative queue exits 2009-2040 and 2025-2040
    for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
        s_full = df.groupby("run_index")[f"annual_exited_{sc}"].sum()
        results[f"exited_total_{sc}"] = pct(s_full, f"Total_Queue_Exits_2009_2040_{tag}")

        sub_proj = period(df, PROJ[0], PROJ[1])
        s_proj = sub_proj.groupby("run_index")[f"annual_exited_{sc}"].sum()
        results[f"exited_total_proj_{sc}"] = pct(s_proj, f"Total_Queue_Exits_2025_2040_{tag}")

    diff_ex_full = (
        df.groupby("run_index")["annual_exited_cap"].sum()
        - df.groupby("run_index")["annual_exited_unc"].sum()
    )
    results["exited_total_diff"] = pct(diff_ex_full, "Total_Queue_Exits_2009_2040_Cap_minus_Unc")

    sub_proj = period(df, PROJ[0], PROJ[1])
    diff_ex_proj = (
        sub_proj.groupby("run_index")["annual_exited_cap"].sum()
        - sub_proj.groupby("run_index")["annual_exited_unc"].sum()
    )
    results["exited_total_proj_diff"] = pct(diff_ex_proj, "Total_Queue_Exits_2025_2040_Cap_minus_Unc")

    # 3. Nationality exits 2009-2040 and 2025-2040
    for nat in ["India", "China", "ROW"]:
        for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
            s_full = df.groupby("run_index")[f"annual_exited_{nat}_{sc}"].sum()
            results[f"exits_{nat}_{sc}"] = pct(s_full, f"Exits_{nat}_2009_2040_{tag}")

            sub_proj = period(df, PROJ[0], PROJ[1])
            s_proj = sub_proj.groupby("run_index")[f"annual_exited_{nat}_{sc}"].sum()
            results[f"exits_{nat}_proj_{sc}"] = pct(s_proj, f"Exits_{nat}_2025_2040_{tag}")

        diff_full = (
            df.groupby("run_index")[f"annual_exited_{nat}_cap"].sum()
            - df.groupby("run_index")[f"annual_exited_{nat}_unc"].sum()
        )
        results[f"exits_{nat}_diff"] = pct(diff_full, f"Exits_{nat}_2009_2040_Cap_minus_Unc")

        sub_proj = period(df, PROJ[0], PROJ[1])
        diff_proj = (
            sub_proj.groupby("run_index")[f"annual_exited_{nat}_cap"].sum()
            - sub_proj.groupby("run_index")[f"annual_exited_{nat}_unc"].sum()
        )
        results[f"exits_{nat}_proj_diff"] = pct(diff_proj, f"Exits_{nat}_2025_2040_Cap_minus_Unc")

    # 3b. EB category exits 2009-2040 and 2025-2040
    for eb in ["EB1", "EB2", "EB3", "EB4", "EB5"]:
        for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
            s_full = df.groupby("run_index")[f"annual_exited_{eb}_{sc}"].sum()
            results[f"exits_{eb}_{sc}"] = pct(s_full, f"Exits_{eb}_2009_2040_{tag}")

            sub_proj = period(df, PROJ[0], PROJ[1])
            s_proj = sub_proj.groupby("run_index")[f"annual_exited_{eb}_{sc}"].sum()
            results[f"exits_{eb}_proj_{sc}"] = pct(s_proj, f"Exits_{eb}_2025_2040_{tag}")

        diff_full = (
            df.groupby("run_index")[f"annual_exited_{eb}_cap"].sum()
            - df.groupby("run_index")[f"annual_exited_{eb}_unc"].sum()
        )
        results[f"exits_{eb}_diff"] = pct(diff_full, f"Exits_{eb}_2009_2040_Cap_minus_Unc")

        sub_proj = period(df, PROJ[0], PROJ[1])
        diff_proj = (
            sub_proj.groupby("run_index")[f"annual_exited_{eb}_cap"].sum()
            - sub_proj.groupby("run_index")[f"annual_exited_{eb}_unc"].sum()
        )
        results[f"exits_{eb}_proj_diff"] = pct(diff_proj, f"Exits_{eb}_2025_2040_Cap_minus_Unc")

    # 4. Cumulative age-outs by period
    period_map = {"recon": RECON, "proj": PROJ, "full": FULL}
    period_label = {"recon": "2009_2024", "proj": "2025_2040", "full": "2009_2040"}
    for pname, (y0, y1) in period_map.items():
        sub = period(df, y0, y1)
        for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
            s = sub.groupby("run_index")[f"annual_aged_out_{sc}"].sum()
            results[f"ageout_{pname}_{sc}"] = pct(s, f"Total_AgeOuts_{period_label[pname]}_{tag}")
        diff = (
            sub.groupby("run_index")[f"annual_aged_out_cap"].sum()
            - sub.groupby("run_index")[f"annual_aged_out_unc"].sum()
        )
        results[f"ageout_{pname}_diff"] = pct(diff, f"Total_AgeOuts_{period_label[pname]}_Cap_minus_Unc")

    # 5. Annual temporal evolution FY2025-2040
    annual_rows = []
    for yr in range(2025, 2041):
        yr_df = df[df["year"] == yr]
        row = {"year": yr}
        for sc in ["unc", "cap"]:
            col = f"annual_aged_out_{sc}"
            row[f"{sc}_mean"] = round(yr_df[col].mean(), 1)
            row[f"{sc}_lo"] = round(np.percentile(yr_df[col], 2.5), 1)
            row[f"{sc}_hi"] = round(np.percentile(yr_df[col], 97.5), 1)
        row["diff"] = round(row["cap_mean"] - row["unc_mean"], 1)
        annual_rows.append(row)
    results["annual"] = annual_rows

    # 6. Nationality age-outs (full and projection)
    for nat in ["India", "China", "ROW"]:
        for pname, (y0, y1) in [("full", FULL), ("proj", PROJ)]:
            plbl = period_label[pname]
            sub = period(df, y0, y1)
            for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
                s = sub.groupby("run_index")[f"annual_aged_out_{nat}_{sc}"].sum()
                results[f"ageout_{nat}_{pname}_{sc}"] = pct(s, f"AgeOuts_{nat}_{plbl}_{tag}")
            diff = (
                sub.groupby("run_index")[f"annual_aged_out_{nat}_cap"].sum()
                - sub.groupby("run_index")[f"annual_aged_out_{nat}_unc"].sum()
            )
            results[f"ageout_{nat}_{pname}_diff"] = pct(diff, f"AgeOuts_{nat}_{plbl}_Cap_minus_Unc")

    # 7. EB category age-outs (full and projection)
    for eb in ["EB1", "EB2", "EB3", "EB4", "EB5"]:
        for pname, (y0, y1) in [("full", FULL), ("proj", PROJ)]:
            plbl = period_label[pname]
            sub = period(df, y0, y1)
            for sc, tag in [("unc", "Uncapped"), ("cap", "Capped")]:
                s = sub.groupby("run_index")[f"annual_aged_out_{eb}_{sc}"].sum()
                results[f"ageout_{eb}_{pname}_{sc}"] = pct(s, f"AgeOuts_{eb}_{plbl}_{tag}")
            diff = (
                sub.groupby("run_index")[f"annual_aged_out_{eb}_cap"].sum()
                - sub.groupby("run_index")[f"annual_aged_out_{eb}_unc"].sum()
            )
            results[f"ageout_{eb}_{pname}_diff"] = pct(diff, f"AgeOuts_{eb}_{plbl}_Cap_minus_Unc")

    return results


def print_and_save(results, outpath):
    W = 60  # label column width for console

    def fmt(d):
        return f"  {d['label']:<{W}} mean={d['mean']:>14,.1f}   " f"lo={d['lo']:>14,.1f}   hi={d['hi']:>14,.1f}"

    # Section 1
    print("=" * 100)
    print("1. BACKLOG, VISAS, AND TOTAL EXITS")
    print("=" * 100)
    for k in [
        "backlog_unc",
        "backlog_cap",
        "backlog_diff",
        "visas_unc",
        "visas_cap",
        "exited_total_unc",
        "exited_total_cap",
        "exited_total_diff",
        "exited_total_proj_unc",
        "exited_total_proj_cap",
        "exited_total_proj_diff",
    ]:
        print(fmt(results[k]))
    print()

    # Section 2: Nationality exits
    print("=" * 100)
    print("2. NATIONALITY EXITS (2009-2040 AND 2025-2040)")
    print("=" * 100)
    for nat in ["India", "China", "ROW"]:
        for sfx in ["unc", "cap", "diff", "proj_unc", "proj_cap", "proj_diff"]:
            print(fmt(results[f"exits_{nat}_{sfx}"]))
        print()

    # Section 2b: EB category exits
    print("=" * 100)
    print("2b. EB CATEGORY EXITS (2009-2040 AND 2025-2040)")
    print("=" * 100)
    for eb in ["EB1", "EB2", "EB3", "EB4", "EB5"]:
        for sfx in ["unc", "cap", "diff", "proj_unc", "proj_cap", "proj_diff"]:
            print(fmt(results[f"exits_{eb}_{sfx}"]))
        print()

    # Section 3: Aggregate age-outs
    print("=" * 100)
    print("3. AGGREGATE AGE-OUTS BY PERIOD")
    print("=" * 100)
    for k in [
        "ageout_recon_unc",
        "ageout_recon_cap",
        "ageout_recon_diff",
        "ageout_proj_unc",
        "ageout_proj_cap",
        "ageout_proj_diff",
        "ageout_full_unc",
        "ageout_full_cap",
        "ageout_full_diff",
    ]:
        print(fmt(results[k]))
    print()

    # Section 4: Annual age-outs
    print("=" * 100)
    print("4. ANNUAL AGE-OUTS FY2025-2040")
    print("=" * 100)
    print(
        f"  {'FY':<6} {'Unc_mean':>10} {'Unc_lo':>10} {'Unc_hi':>10} "
        f"{'Cap_mean':>10} {'Cap_lo':>10} {'Cap_hi':>10} {'Diff':>8}"
    )
    for r in results["annual"]:
        print(
            f"  {r['year']:<6} {r['unc_mean']:>10,.1f} {r['unc_lo']:>10,.1f} "
            f"{r['unc_hi']:>10,.1f} {r['cap_mean']:>10,.1f} {r['cap_lo']:>10,.1f} "
            f"{r['cap_hi']:>10,.1f} {r['diff']:>8,.1f}"
        )
    print()

    # Section 5: Nationality age-outs
    print("=" * 100)
    print("5. NATIONALITY AGE-OUTS -- FULL (2009-2040) AND PROJECTION (2025-2040)")
    print("=" * 100)
    for nat in ["India", "China", "ROW"]:
        for pname in ["full", "proj"]:
            for sfx in ["unc", "cap", "diff"]:
                print(fmt(results[f"ageout_{nat}_{pname}_{sfx}"]))
            print()

    # Section 6: EB category age-outs
    print("=" * 100)
    print("6. EB CATEGORY AGE-OUTS -- FULL (2009-2040) AND PROJECTION (2025-2040)")
    print("=" * 100)
    for eb in ["EB1", "EB2", "EB3", "EB4", "EB5"]:
        for pname in ["full", "proj"]:
            for sfx in ["unc", "cap", "diff"]:
                print(fmt(results[f"ageout_{eb}_{pname}_{sfx}"]))
            print()

    # Save scalar CSV (no spaces in any label; year ranges explicit)
    rows = []
    for key, val in results.items():
        if key == "annual":
            continue
        rows.append(
            {
                "metric": val["label"],  # underscored, no spaces
                "mean": val["mean"],
                "lo_p025": val["lo"],  # 2.5th percentile
                "hi_p975": val["hi"],  # 97.5th percentile
            }
        )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    pd.DataFrame(rows).to_csv(outpath, index=False)

    # Save annual table separately
    annual_path = outpath.replace("ci_summary_stats", "ci_annual_table")
    pd.DataFrame(results["annual"]).to_csv(annual_path, index=False)

    print(f"\nSaved:\n  {outpath}\n  {annual_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize complete paired runs with accounting checks")
    parser.add_argument("--input", default=FILEPATH)
    parser.add_argument("--output", default=OUTPATH)
    args = parser.parse_args()
    results = compute_all_stats(args.input)
    print_and_save(results, args.output)
