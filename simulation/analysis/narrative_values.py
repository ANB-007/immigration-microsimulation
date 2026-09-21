"""Keep the manuscript's narrative numbers tied to the selected result tables."""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

from .manuscript_refresh import number


def group_name(value):
    return {"EB-1": "EbOne", "EB-2": "EbTwo", "EB-3": "EbThree",
            "EB-4": "EbFour", "EB-5": "EbFive", "Capped minus uncapped": "Diff",
            "Capped - Uncapped": "Diff", "Capped−Uncapped": "Diff"}.get(value, value)


def cohort_date_claims(cohort_runs):
    """Check the stated entry dates in every observed run, not just the mean."""
    saved = cohort_runs[cohort_runs.fate == "saved"]
    seeds = set(saved.seed)
    checks, evidence = {}, {}
    for country, category, policy, first_year in (
        ("India", "EB-2", "Capped", 2016), ("India", "EB-3", "Capped", 2016),
        ("China", "EB-2", "Capped", 2025), ("ROW", "EB-4", "Uncapped", 2025),
        ("ROW", "EB-4", "Capped", 2025),
    ):
        part = saved[(saved.nationality == country) & (saved.eb_category == category)
                     & (saved.scenario == policy) & saved.entry_year.between(first_year, 2040)]
        expected = {(seed, year) for seed in seeds for year in range(first_year, 2041)}
        observed = set(zip(part.seed, part.entry_year))
        key = f"every_cohort_near_zero_{country}_{category}_{policy}_from_{first_year}"
        checks[key] = bool(expected and observed == expected and part.value.lt(1).all())
        evidence[key] = {"maximum_save_share_pct": float(part.value.max()) if len(part) else None,
                         "missing_seed_years": sorted(expected - observed),
                         "cohorts_at_or_above_one_percent": part.loc[part.value.ge(1), ["seed", "entry_year", "value"]].to_dict("records")}
    return checks, evidence


def generate(source: Path, output: Path, manuscript: Path) -> dict:
    """Generate numbers only; retain the author's prose and verify every use."""
    source, output, manuscript = Path(source), Path(output), Path(manuscript)
    data = json.loads(source.read_text())
    values = {}

    def add(key, value, digits=0):
        if key in values:
            raise ValueError(f"Duplicate narrative value: {key}")
        values[key] = {"value": value, "formatted": number(value, digits)}

    for row in data["metric_summary"]:
        key = ".".join((row["metric"], row["period"], group_name(row["group"]), group_name(row["scenario"])))
        digits = 1 if row["metric"].endswith("_pct") else 0
        add(key, row["mean"], digits)
        add(key + ".Abs", abs(row["mean"]), digits)
        for field, suffix in (("mcse_mean", "MCSE"), ("sd_across_runs", "SD"),
                              ("mean_ci95_low", "Low"), ("mean_ci95_high", "High")):
            add(key + "." + suffix, row[field], 1 if field in {"mcse_mean", "sd_across_runs"} else digits)
    for row in data["share_summary"]:
        key = ".".join((row["metric"], row["period"], group_name(row["group"]), group_name(row["scenario"]), "Share"))
        add(key, row["mean"], 1)
    for row in data["annual_summary"]:
        key = ".".join(("annual", str(row["year"]), group_name(row["group"]), group_name(row["scenario"])))
        add(key, row["mean"])
    for row in data["historical_fit"]:
        key = "fit." + group_name(row["group"])
        for field in ("mean_observed", "mean_model", "bias", "rmse", "wape", "correlation"):
            add(key + "." + field, row[field], 3 if field == "correlation" else 1)
        if row["dimension"] == "overall":
            add("fit.totalObserved", 16 * row["mean_observed"])
            add("fit.totalModel", 16 * row["mean_model"])
            add("fit.totalBias", 16 * row["bias"])
    for row in data.get("cohort_save_rates", []):
        key = ".".join(("cohort", row["period"], row["nationality"], group_name(row["category"]), group_name(row["scenario"])))
        add(key, row["mean"], 2)
    cohort = pd.DataFrame(data.get("cohort_share_summary", []))
    if not cohort.empty:
        saved = cohort[cohort.fate == "saved"]
        for (country, category, policy), rows in saved.groupby(["nationality", "eb_category", "scenario"]):
            observed_nonzero = rows[rows["mean"] > 0]
            year = int(observed_nonzero.entry_year.max()) + 1 if len(observed_nonzero) else 2009
            key = f"cohort.zero.from.{country}.{group_name(category)}.{policy}"
            add(key, year)
            values[key]["formatted"] = str(year)

    applicants = pd.read_csv(source.parent / "annual_applicant_types.csv")
    applicants = applicants[applicants.year >= 2025]
    if sorted(applicants.seed.unique().tolist()) != data["selection"]["annual_seeds"]:
        raise ValueError("Applicant composition does not use the same primary seeds")
    totals = applicants.groupby(["seed", "scenario"])[["principals", "spouses", "children", "visas"]].sum()
    for policy in ("Uncapped", "Capped"):
        selected = totals.xs(policy, level="scenario")
        for kind in ("principals", "spouses", "children"):
            key = f"applicants.projection.{policy}.{kind}"
            add(key, float(selected[kind].mean()))
            add(key + ".Share", float((100 * selected[kind] / selected.visas).mean()), 1)

    projection_annual = [row for row in data["annual_summary"] if row["year"] >= 2025
                         and row["group"] == "All" and group_name(row["scenario"]) == "Diff"]
    peak = max(projection_annual, key=lambda row: row["mean"])
    add("annual.peak.year", peak["year"])
    values["annual.peak.year"]["formatted"] = str(int(peak["year"]))
    add("annual.peak.difference", peak["mean"])
    for period in ("full", "projection"):
        for policy in ("Uncapped", "Capped"):
            total = sum(values[f"ageouts.{period}.{group}.{policy}.Share"]["value"] for group in ("EbTwo", "EbThree"))
            add(f"ageouts.{period}.EbTwoThree.{policy}.Share", total, 1)
        unc = values[f"ageouts.{period}.India.Uncapped"]["value"]
        cap = values[f"ageouts.{period}.India.Capped"]["value"]
        add(f"ageouts.{period}.India.RelativeChange", 100 * (cap - unc) / unc, 1)

    # Numerical substitution cannot repair prose whose direction is no longer
    # supported. Require an editorial update if a new scientific run changes
    # the specific qualitative statements retained in the manuscript.
    value = lambda key: values[key]["value"]
    claims = {
        "equal_total_visas_across_policies": value("visas.full.All.Capped") == value("visas.full.All.Uncapped"),
        "capped_backlog_smaller": value("backlog.end2040.All.Diff") < 0,
        "capped_principal_conversions_fewer": value("conversions.projection.All.Diff") < 0,
        "capped_exits_more": value("exits.projection.All.Diff") > 0,
        "eb2_exit_increase": value("exits.projection.EbTwo.Diff") > 0,
        "other_categories_partly_offset_eb2_exit_increase":
            value("exits.projection.All.Diff") < value("exits.projection.EbTwo.Diff"),
        "capped_ageouts_more": value("ageouts.projection.All.Diff") > 0,
        "india_ageouts_more_than_double": value("ageouts.projection.India.RelativeChange") > 100,
        "india_majority_of_capped_ageouts": value("ageouts.projection.India.Capped.Share") > 50,
        "row_ageouts_fewer": value("ageouts.full.ROW.Diff") < 0,
        "capped_child_visa_share_higher": value("applicants.projection.Capped.children.Share")
                                            > value("applicants.projection.Uncapped.children.Share"),
        "india_and_china_extra_exits": value("exits.projection.India.Diff")
                                         > value("exits.projection.China.Diff") > 0,
        "china_extra_ageouts": value("ageouts.projection.China.Diff") > 0,
        "all_projection_years_have_extra_ageouts": all(row["mean"] > 0 for row in projection_annual),
        "eb2_and_eb3_majority_of_extra_ageouts":
            value("ageouts.projection.EbTwo.Diff") + value("ageouts.projection.EbThree.Diff")
            > 0.5 * value("ageouts.projection.All.Diff"),
        "eb2_and_eb3_majority_of_ageouts_in_both_scenarios": all(
            value(f"ageouts.full.EbTwoThree.{policy}.Share") > 50 for policy in ("Uncapped", "Capped")),
    }
    for country, category in (("India", "EbTwo"), ("India", "EbThree"), ("China", "EbTwo")):
        key = f"cohort.projection.{country}.{category}.Capped"
        if key in values:
            claims[f"near_zero_cohort_save_share_{country}_{category}"] = value(key) < 1
    for country, category, policy in (("India", "EbTwo", "Capped"), ("India", "EbThree", "Capped")):
        key = f"cohort.zero.from.{country}.{category}.{policy}"
        if key in values:
            claims[f"observed_zero_projection_cohort_saves_{country}_{category}_{policy}"] = value(key) <= 2025
    for category in ("EbOne", "EbThree", "EbFour"):
        claims[f"{category}_exits_fewer_under_cap"] = value(f"exits.projection.{category}.Diff") < 0
    claims["EbFive_exits_more_under_cap"] = value("exits.projection.EbFive.Diff") > 0
    core_categories = ("EbOne", "EbTwo", "EbThree", "EbFour")
    claims["category_one_to_four_wape_below_six_percent"] = all(value(f"fit.{category}.wape") < 6 for category in core_categories)
    correlations = [value(f"fit.{category}.correlation") for category in core_categories]
    claims["category_one_to_four_correlation_range_093_to_099"] = (
        round(min(correlations), 2) == .93 and round(max(correlations), 2) == .99)
    for policy in ("Uncapped", "Capped"):
        counts = [value(f"annual.{year}.All.{policy}") for year in range(2025, 2041)]
        claims[f"annual_ageouts_increase_each_projection_year_{policy}"] = all(b > a for a, b in zip(counts, counts[1:]))
    claims["annual_gap_peaks_early_or_mid_2030s"] = 2030 <= value("annual.peak.year") <= 2036
    claims["annual_gap_narrows_by_2040"] = value("annual.2040.All.Diff") < value("annual.peak.difference")
    cohort_checks, cohort_evidence = cohort_date_claims(pd.read_csv(source.parent / "cohort_share_runs.csv"))
    claims.update(cohort_checks)
    failed = [key for key, passed in claims.items() if not passed]
    if failed:
        output.mkdir(parents=True, exist_ok=True)
        (output / "interpretation-failures.json").write_text(json.dumps({
            "failed_claims": failed, "checks": claims, "cohort_date_evidence": cohort_evidence,
            "values": values,
        }, indent=2, sort_keys=True, allow_nan=False) + "\n")
        raise ValueError("Manuscript interpretation requires revision for these results: " + ", ".join(failed))

    # Undefined TeX control sequences otherwise silently produce empty text
    # through csname. Fail the build rather than drop a missing result number.
    required = set(re.findall(r"\\Mval\{([^}]+)\}", manuscript.read_text()))
    missing = required - set(values)
    if missing:
        raise ValueError("Unresolved manuscript narrative values: " + ", ".join(sorted(missing)))
    output.mkdir(parents=True, exist_ok=True)
    lines = [r"\newcommand{\Mval}[1]{\csname manval@#1\endcsname}"]
    for key, value in sorted(values.items()):
        lines.append(r"\expandafter\def\csname manval@" + key + r"\endcsname{" + value["formatted"] + "}")
    (output / "manuscript-narrative-values.tex").write_text("\n".join(lines) + "\n")
    report = {"source": str(source), "selection": data["selection"], "required_keys": sorted(required),
              "qualitative_claim_checks": claims,
              "cohort_date_evidence": cohort_evidence,
              "near_zero_definition_percent": 1, "values": values}
    (output / "narrative-value-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report
