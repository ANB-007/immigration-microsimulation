"""Analyze the frozen, fixed-size catchall campaign without running simulations.

Strict mode requires every declared pair. Explicit partial previews carry a
nonpublication status and never produce the manuscript-facing TeX filenames.
The input plan and original run identities remain unchanged after relocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from simulation.campaign import load_campaign
from .accounting import audit_record, projection_metric_rows, group_definitions, select_group, historical_supply
from .figure_style import BLUE, GREEN, ORANGE, FONT_SIZE, PLOT_DPI, manuscript_style

POLICIES = ("Uncapped", "Capped")
CONTRAST = "Capped minus uncapped"
CATEGORIES = tuple(f"EB-{i}" for i in range(1, 6))
NATIONALITIES = ("India", "China", "ROW")
THRESHOLDS = (60, 65, 70, 75, 80)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_precision(values, *, confidence_interval=True) -> dict:
    """Sample precision of a fixed-input mean; no population/model coverage claim."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Precision requires a nonempty finite vector")
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else None
    mcse = sd / np.sqrt(n) if sd is not None else None
    halfwidth = float(t.ppf(.975, n - 1) * mcse) if confidence_interval and n > 1 else None
    mean = float(values.mean())
    return {"n": n, "mean": mean, "sd_across_runs": sd, "mcse_mean": mcse,
            "mean_ci95_low": mean - halfwidth if halfwidth is not None else None,
            "mean_ci95_high": mean + halfwidth if halfwidth is not None else None,
            "positive_runs": int((values > 0).sum()), "zero_runs": int((values == 0).sum()),
            "negative_runs": int((values < 0).sum())}


def summarize(frame: pd.DataFrame, keys: list[str], value="value", *, intervals=True) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=keys + list(numerical_precision([0])))
    rows = []
    for group, part in frame.groupby(keys, sort=False, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        if "seed" in part and part.seed.duplicated().any():
            raise ValueError("Repeated seed in a summary cell")
        rows.append({**dict(zip(keys, group)), **numerical_precision(part[value], confidence_interval=intervals)})
    return pd.DataFrame(rows)


def audit_followup(short: dict, long: dict) -> dict:
    if short["seed"] != long["seed"]:
        raise ValueError("Follow-up must use its matching primary seed")
    for key, values in short["raw"].items():
        if key.startswith("annual_") and values != long["raw"].get(key, [])[:len(values)]:
            raise ValueError(f"Long-follow-up prefix differs: {key}")
    cohorts = pd.DataFrame(long["cohorts"])
    fields = ["total_created", "saved", "aged_out", "exited", "still_dependent"]
    counts = cohorts[fields].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Invalid follow-up child counts")
    if not cohorts[fields[1:]].sum(axis=1).eq(cohorts.total_created).all():
        raise ValueError("Follow-up child accounting failed")
    selected = cohorts[cohorts.entry_year <= 2040]
    if selected.still_dependent.sum() != 0:
        raise ValueError("Selected cohorts have unresolved outcomes at 2061")
    if cohorts.loc[cohorts.entry_year > 2040, "total_created"].sum() <= 0:
        raise ValueError("Later arrivals were not retained during follow-up")
    for policy, suffix in [("Uncapped", "unc"), ("Capped", "cap")]:
        annual_visas = np.asarray(long["raw"][f"annual_visas_consumed_{suffix}"])
        if (annual_visas[32:] > 167394).any() or (annual_visas < 0).any():
            raise ValueError("Long-follow-up annual visa budget failed")
        part = cohorts[cohorts.scenario == policy]
        if part.aged_out.sum() != sum(long["raw"][f"annual_aged_out_{suffix}"]):
            raise ValueError("Long-follow-up child age-outs disagree with annual counts")
        if part.saved.sum() != sum(long["raw"][f"annual_children_saved_{suffix}"]):
            raise ValueError("Long-follow-up saved children disagree with annual counts")
    return {"seed": short["seed"], "annual_prefix_identical": True,
            "selected_unresolved_children": 0, "post_2040_arrivals_retained": True,
            "child_cells_checked": len(cohorts)}


def age_tail_rows(value: dict) -> pd.DataFrame:
    """Use event-year ages and actual terminal ages, never inferred historic ages."""
    frame = pd.DataFrame(value["principal_age_profile"])
    key = ["scenario", "outcome", "observation_or_event_year", "nationality", "category", "age"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate age profile cell")
    counts = frame[["age", "principals"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Invalid age profile counts")
    end_year = value["raw"]["years"][-1]
    if not frame.loc[frame.outcome == "still_waiting", "observation_or_event_year"].eq(end_year).all():
        raise ValueError("Waiting ages do not use the terminal census year")
    for scenario, suffix in [("Uncapped", "unc"), ("Capped", "cap")]:
        part = frame[frame.scenario == scenario]
        if int(part.loc[part.outcome == "still_waiting", "principals"].sum()) != value["raw"][f"annual_backlog_total_{suffix}"][-1]:
            raise ValueError("Waiting age census does not match the final principal stock")
        for outcome, raw_key in [("converted", "conversions"), ("exited", "exited")]:
            events = part[part.outcome == outcome].groupby("observation_or_event_year").principals.sum()
            observed = events.reindex(value["raw"]["years"], fill_value=0).tolist()
            if observed != value["raw"][f"annual_{raw_key}_{suffix}"]:
                raise ValueError("Event age census does not match annual principal outcomes")
    rows = []
    for scenario in POLICIES:
        for year, outcome in [(2024, "converted"), (2040, "converted"), (2040, "still_waiting")]:
            if outcome == "still_waiting" and end_year != year:
                continue
            selected = frame[(frame.scenario == scenario) & (frame.observation_or_event_year == year) & (frame.outcome == outcome)]
            groups = [("overall", "All EB", selected)]
            groups += [(dimension, group, selected[selected[dimension] == group])
                       for dimension, names in [("category", CATEGORIES), ("nationality", NATIONALITIES)] for group in names]
            for dimension, group, part in groups:
                denominator = int(part.principals.sum())
                maximum = int(part.age.max()) if denominator else None
                for threshold in THRESHOLDS:
                    count = int(part.loc[part.age >= threshold, "principals"].sum())
                    rows.append({"scenario": scenario, "year": year, "outcome": outcome,
                                 "dimension": dimension, "group": group, "threshold_age": threshold,
                                 "denominator": denominator, "count_at_or_above": count,
                                 "share_pct": 100 * count / denominator if denominator else np.nan,
                                 "maximum_age": maximum})
    return pd.DataFrame(rows)


def paired_replay(primary: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "dimension", "group", "quantity", "policy"]
    subset = primary[primary.seed.isin(replay.seed.unique())]
    merged = replay[keys + ["value"]].merge(subset[keys + ["value"]], on=keys,
        how="outer", validate="one_to_one", suffixes=("_replay", "_baseline"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Replay lacks its matched primary seed and metric")
    merged["value"] = merged.value_replay - merged.value_baseline
    return merged.drop(columns="_merge")


def historical_metric_rows(record: dict) -> pd.DataFrame:
    """One copy of historical events and the end-2024 principal inventory.

    Event years, rather than entry cohorts followed beyond 2024, define this
    comparison. Both policy copies must have the same capped pre-2025 history.
    The saved records do not include an end-2024 waiting-age census.
    """
    raw = record["raw"]
    years = np.asarray(raw["years"])
    history = (years >= 2009) & (years <= 2024)
    if years[history].tolist() != list(range(2009, 2025)):
        raise ValueError("Historical comparison requires FY2009--FY2024")
    rows = []

    def add(dimension, group, quantity, value):
        rows.append({"dimension": dimension, "group": group, "quantity": quantity,
                     "policy": "Historical", "value": float(value)})

    for dimension, groups in (("overall", ["All"]), ("nationality", NATIONALITIES),
                              ("category", CATEGORIES)):
        for group in groups:
            token = "" if dimension == "overall" else group.replace("-", "") + "_"
            for field, quantity in (("aged_out", "age21_events_2009_2024"),
                                    ("exited", "principal_exits_2009_2024")):
                arrays = [np.asarray(raw[f"annual_{field}_{token}{suffix}"]) for suffix in ("cap", "unc")]
                if any(values.shape != years.shape for values in arrays):
                    raise ValueError("Incomplete historical event series")
                if not np.array_equal(arrays[0][history], arrays[1][history]):
                    raise ValueError("Historical policy copies differ")
                add(dimension, group, quantity, arrays[0][history].sum())

    cells = pd.DataFrame(record["annual_cells"])
    cells = cells[cells.year == 2024]
    keys = ["category", "nationality"]
    expected = pd.MultiIndex.from_product([CATEGORIES, NATIONALITIES], names=keys)
    inventories = []
    for policy, suffix in (("Capped", "cap"), ("Uncapped", "unc")):
        stock = cells[cells.scenario == policy].set_index(keys).principal_backlog
        if stock.index.has_duplicates or set(stock.index) != set(expected):
            raise ValueError("Incomplete end-2024 joint inventory")
        stock = stock.reindex(expected)
        if stock.sum() != raw[f"annual_backlog_total_{suffix}"][list(years).index(2024)]:
            raise ValueError("End-2024 joint inventory does not reconcile")
        inventories.append(stock)
    if not inventories[0].equals(inventories[1]):
        raise ValueError("Historical policy inventories differ")
    inventory = inventories[0].reset_index()
    for dimension, group, category, nationality in group_definitions():
        selected = select_group(inventory, category, nationality)
        add(dimension, group, "principal_backlog_2024", selected.principal_backlog.sum())
    return pd.DataFrame(rows)


def summarize_historical_replay(changes: pd.DataFrame) -> pd.DataFrame:
    keys = ["dimension", "group", "quantity", "policy"]
    precision = summarize(changes, keys)
    levels = changes.groupby(keys, as_index=False).agg(
        mean_baseline=("value_baseline", "mean"), mean_replay=("value_replay", "mean"))
    return precision.merge(levels, on=keys, validate="one_to_one")


def audit_replay_targets(value: dict, observed: pd.DataFrame) -> pd.DataFrame:
    """Verify both policy copies; return one copy of the shared reconstruction."""
    audit = pd.DataFrame(value["historical_replay_audit"])
    keys = ["scenario", "year", "category", "nationality"]
    expected = pd.MultiIndex.from_product([POLICIES, range(2009, 2025), CATEGORIES, NATIONALITIES], names=keys)
    if audit.empty or audit.duplicated(keys).any() or set(map(tuple, audit[keys].to_numpy())) != set(expected):
        raise ValueError("Replay audit requires exactly 480 historical policy cells")
    counts = audit[["target_visas", "used_visas", "unfilled_target"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Replay target counts must be nonnegative integers")
    if not audit.target_visas.eq(audit.used_visas + audit.unfilled_target).all():
        raise ValueError("Replay target does not equal used plus unfilled")
    actual = observed[observed.category.isin(CATEGORIES)]
    comparison = audit.merge(actual, on=keys[1:], how="left", validate="many_to_one")
    if comparison.visas_issued.isna().any() or not comparison.target_visas.eq(comparison.visas_issued).all():
        raise ValueError("Replay targets differ from the frozen DOS benchmark")
    annual = pd.DataFrame(value["annual_cells"])
    annual = annual[annual.year <= 2024]
    comparison = audit.merge(annual[keys + ["annual_visas"]], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not comparison._merge.eq("both").all() or not comparison.used_visas.eq(comparison.annual_visas).all():
        raise ValueError("Replayed visas disagree with annual joint cells")
    histories = [audit[audit.scenario == policy].drop(columns="scenario").set_index(keys[1:]).sort_index() for policy in POLICIES]
    if not histories[0].equals(histories[1]):
        raise ValueError("Replay historical policies differ")
    return audit[audit.scenario == "Capped"].drop(columns="scenario").reset_index(drop=True)


def summarize_resolved_cohorts(cohorts: pd.DataFrame) -> pd.DataFrame:
    """Count means include omitted zero cells; empty-cohort percentages stay undefined."""
    keys = ["seed", "scenario", "entry_year", "nationality", "eb_category"]
    if cohorts.duplicated(keys).any():
        raise ValueError("Duplicated resolved-cohort cell")
    full_index = pd.MultiIndex.from_product([sorted(cohorts.seed.unique()), POLICIES, range(2009, 2041),
                                            NATIONALITIES, CATEGORIES], names=keys)
    counts = ["total_created", "saved", "aged_out", "exited", "still_dependent"]
    complete = cohorts.set_index(keys)[counts].reindex(full_index, fill_value=0).reset_index()
    complete["observation_end_year"] = 2061
    for name, numerator in [("save_pct", "saved"), ("ageout_pct", "aged_out"), ("exit_pct", "exited")]:
        complete[name] = 100 * complete[numerator] / complete.total_created.replace(0, np.nan)
    long = complete.melt(id_vars=keys + ["observation_end_year"],
        value_vars=counts + ["save_pct", "ageout_pct", "exit_pct"], var_name="quantity", value_name="value")
    return summarize(long.dropna(subset=["value"]),
        ["scenario", "entry_year", "nationality", "eb_category", "observation_end_year", "quantity"], intervals=False)


def joint_fit(cells: pd.DataFrame, observed: pd.DataFrame):
    observed = observed[observed.category.isin(CATEGORIES)].copy()
    key = ["year", "category", "nationality"]
    if observed.duplicated(key).any() or len(observed) != 16 * 15:
        raise ValueError("DOS benchmark must contain all 240 unique joint historical cells")
    history = cells[(cells.scenario == "Capped") & (cells.year <= 2024)]
    annual = history.groupby(["case"] + key, as_index=False).agg(
        modeled_visas=("annual_visas", "mean"), n_runs=("seed", "nunique"))
    annual = annual.merge(observed, on=key, validate="many_to_one")
    annual["residual_model_minus_observed"] = annual.modeled_visas - annual.visas_issued
    fit = []
    for (case, category, nationality), part in annual.groupby(["case", "category", "nationality"], sort=False):
        if len(part) != 16:
            raise ValueError("Historical fit lacks a full 16-year trajectory")
        residual = part.residual_model_minus_observed
        fit.append({"case": case, "category": category, "nationality": nationality,
            "n_years": 16, "n_runs": int(part.n_runs.iloc[0]),
            "observed_annual_mean": float(part.visas_issued.mean()),
            "modeled_annual_mean": float(part.modeled_visas.mean()),
            "mean_residual_model_minus_dos": float(residual.mean()),
            "rmse_of_ensemble_mean": float(np.sqrt(np.mean(residual ** 2))),
            "wape_of_ensemble_mean_pct": float(100 * residual.abs().sum() / part.visas_issued.sum())})
    return annual, pd.DataFrame(fit)


def retention_curves(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        if case["role"] not in ("primary", "sensitivity"):
            continue
        override = case["overrides"]
        midpoint, tail_age, tail = (override[k] for k in
            ("catchall_median_age", "catchall_near_zero_age", "catchall_tail_probability"))
        slope = np.log((1 - tail) / tail) / (tail_age - midpoint)
        survival = lambda age: 1 / (1 + np.exp(slope * (age - midpoint)))
        for age in range(35, 86):
            rows.append({"case": case["name"], "role": case["role"], "age": age,
                "median_age": midpoint, "near_zero_age": tail_age, "age_curve_S": survival(age),
                "conditional_annual_age_exit": 1 - survival(age + 1) / survival(age),
                "age_only_retention_from_35": survival(age) / survival(35),
                "age_only_retention_from_45": survival(age) / survival(45) if age >= 45 else None})
    return pd.DataFrame(rows)


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = manuscript_style()
    plt.rcParams.update(style)
    return plt, style["font.family"]


def _save(fig, output: Path, name: str, preview: bool):
    import matplotlib
    import matplotlib.pyplot as plt
    if preview:
        fig.text(.5, .01, "PARTIAL PREVIEW — NOT PUBLICATION RESULTS", ha="center", color="#9b2323")
    fig.canvas.draw()
    text = [item for item in fig.findobj(matplotlib.text.Text) if item.get_visible() and item.get_text().strip()]
    if not text or min(item.get_fontsize() for item in text) < FONT_SIZE:
        raise ValueError("Figure text is smaller than the publication font specification")
    if not np.isclose(fig.get_figwidth(), 5.5) or fig.get_figheight() > 6.2:
        raise ValueError("Figure must preserve its native manuscript width")
    fig.savefig(output / f"{name}.png", dpi=PLOT_DPI)
    plt.close(fig)


def write_figures(output: Path, tables: dict, plan: dict, *, preview=False):
    plt, family = _style()
    cases = [c for c in plan["cases"] if c["role"] in ("primary", "sensitivity")]
    ordered = sorted(cases, key=lambda c: c["overrides"]["catchall_median_age"])
    labels = {c["name"]: f"{'Central' if c['role'] == 'primary' else 'Earlier' if i == 0 else 'Later'} ({c['overrides']['catchall_median_age']}/{c['overrides']['catchall_near_zero_age']})"
              for i, c in enumerate(ordered)}
    colors, styles = [ORANGE, BLUE, GREEN], ["--", "-", ":"]
    case_colors = {case["name"]: color for case, color in zip(ordered, colors)}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5))
    fig.subplots_adjust(left=.12, right=.98, bottom=.29, top=.88, wspace=.36)
    for case, color, line in zip(ordered, colors, styles):
        data = tables["retention_curves"]
        data = data[(data.case == case["name"]) & (data.age >= 45)]
        for ax, field in zip(axes, ["age_curve_S", "conditional_annual_age_exit"]):
            ax.plot(data.age, data[field] * 100, color=color, ls=line, lw=1.8, label=labels[case["name"]])
    axes[0].set(title="A  Retention curve S(a)", ylabel="Percent", ylim=(0, 103))
    axes[1].set(title="B  Annual age exit", ylim=(0, 50))
    for ax in axes:
        ax.set(xlabel="Current age", xlim=(45, 80), xticks=[45, 55, 65, 75])
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{c['overrides']['catchall_median_age']}/{c['overrides']['catchall_near_zero_age']}" for c in ordered],
               loc="lower center", ncol=3, bbox_to_anchor=(.55, .045), frameon=False)
    _save(fig, output, "publication-retention", preview)
    projection = tables["projection_runs"]
    main = projection[(projection.dimension == "overall") & (projection.policy == CONTRAST)]
    available = [c for c in ordered if c["name"] in set(main.case)]
    if available:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.2), sharey=True)
        fig.subplots_adjust(left=.29, right=.98, bottom=.24, top=.83, wspace=.28)
        for ax, quantity, title in zip(axes, ["age21_events_2025_2040", "principal_backlog_2040"],
                                      ["Age-out difference", "Backlog difference"]):
            for y, case in enumerate(available):
                data = main[(main.case == case["name"]) & (main.quantity == quantity)].value
                stats = numerical_precision(data)
                color = case_colors[case["name"]]
                ax.scatter(data / 1000, y + np.linspace(-.10, .10, len(data)), color=color, s=9, alpha=.35)
                mean = stats["mean"] / 1000
                error = (stats["mean_ci95_high"] - stats["mean"]) / 1000 if stats["mean_ci95_high"] is not None else 0
                ax.errorbar(mean, y, xerr=error, color=color, fmt="D", markersize=5, capsize=4, lw=1.8)
            ax.axvline(0, color="#555555", lw=.9)
            ax.set(title=title, xlabel="Capped − uncapped\n(thousands)")
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(range(len(available)), [labels[c["name"]] for c in available])
        axes[0].invert_yaxis()
        _save(fig, output, "publication-policy-contrasts", preview)
    return {"font_family": family, "minimum_text_points": FONT_SIZE, "native_width_inches": 5.5,
            "format": "png", "dpi": PLOT_DPI}


def _number(value, decimals=0):
    return f"{value:,.{decimals}f}" if value is not None and np.isfinite(value) else "--"


def single_seed_comparisons(tables: dict, plan: dict) -> dict:
    """Select the declared baseline seed for the two manuscript comparisons.

    Archived ensemble summaries remain available separately. Never substitute
    their means for a missing single-run result.
    """
    seed = plan["configuration"].get("seed", 2014)
    cases = [case["name"] for case in plan["cases"] if case["role"] in ("primary", "sensitivity")]
    results = {}
    if cases:
        runs = tables["projection_runs"]
        selected = runs[(runs.seed == seed) & runs.case.isin(cases)
                        & (runs.quantity == "age21_events_2025_2040")].copy()
        overall = selected[selected.dimension == "overall"]
        expected = {(case, policy) for case in cases for policy in (*POLICIES, CONTRAST)}
        if set(zip(overall.case, overall.policy)) != expected or len(overall) != len(expected):
            raise ValueError(f"Age-schedule comparison requires one result per case and policy for seed {seed}")
        results["age_schedule_comparison"] = selected
    historical = tables.get("replay_historical_outcomes_by_seed", pd.DataFrame())
    if not historical.empty:
        selected = historical[(historical.seed == seed) & historical.dimension.isin(["overall", "nationality"])
                              & historical.quantity.isin(["age21_events_2009_2024", "principal_backlog_2024"])].copy()
        expected = {(dimension, group, quantity) for dimension, group in
                    [("overall", "All"), *[("nationality", n) for n in NATIONALITIES]]
                    for quantity in ("age21_events_2009_2024", "principal_backlog_2024")}
        if set(zip(selected.dimension, selected.group, selected.quantity)) != expected or len(selected) != len(expected):
            raise ValueError(f"Historical replay comparison requires all population rows for seed {seed}")
        results["historical_replay_comparison"] = selected
    return results


def numeric_macros(tables: dict, plan: dict) -> dict:
    """Primary means and single-seed appendix comparisons have separate sources."""
    primary = plan["configuration"]["primary_case"]
    stats = tables["projection_summary"]
    macros = {}

    def precision_macro(name, row, decimals=0):
        macros[name] = _number(row["mean"], decimals)
        for suffix, field, digits in [("MCSE", "mcse_mean", 1), ("CILow", "mean_ci95_low", decimals),
                                      ("CIHigh", "mean_ci95_high", decimals)]:
            macros[name + suffix] = _number(row[field], digits)

    age_comparison = tables.get("age_schedule_comparison", pd.DataFrame())
    for case in plan["cases"]:
        if case["role"] not in ("primary", "sensitivity"):
            continue
        name = "Primary" if case["role"] == "primary" else (
            "Earlier" if case["overrides"]["catchall_median_age"] < 60 else "Later")
        source = stats
        if case["role"] == "sensitivity" and not age_comparison.empty:
            source = summarize(age_comparison, ["case", "dimension", "group", "quantity", "policy"])
        selected = source[(source.case == case["name"]) & (source.quantity == "age21_events_2025_2040")]
        for dimension, group, fragment in [("overall", "All", ""), ("nationality", "India", "India")]:
            for row in selected[(selected.dimension == dimension) & (selected.group == group)].to_dict("records"):
                policy = {"Capped": "Capped", "Uncapped": "Uncapped", CONTRAST: "Difference"}[row["policy"]]
                precision_macro("Pub" + name + fragment + policy + "Ageouts", row)
                macros["Pub" + name + "Runs"] = str(row["n"])
    for row in stats[(stats.case == primary) & (stats.dimension == "overall") & (stats.policy == CONTRAST)].to_dict("records"):
        aliases = {"principal_backlog_2040": "PubDeltaBacklog", "principal_exits_2025_2040": "PubDeltaExits",
                   "principal_conversions_2025_2040": "PubDeltaPrincipalConv"}
        if row["quantity"] in aliases:
            precision_macro(aliases[row["quantity"]], row)
    differences = stats[(stats.dimension == "overall") & (stats.quantity == "age21_events_2025_2040") & (stats.policy == CONTRAST)]
    primary_difference = differences[differences.case == primary]
    if len(primary_difference) == 1:
        result = primary_difference.iloc[0]
        macros["PubPrimaryDifferenceRelativeMCSE"] = _number(
            100 * result.mcse_mean / abs(result["mean"]) if result["mean"] else None, 2)
    if not age_comparison.empty:
        differences = age_comparison[(age_comparison.dimension == "overall") & (age_comparison.policy == CONTRAST)]
        macros["PubSensitivityDifferenceMin"] = _number(differences.value.min())
        macros["PubSensitivityDifferenceMax"] = _number(differences.value.max())
    for row in tables.get("india_share_summary", pd.DataFrame()).to_dict("records"):
        if row["case"] == primary:
            precision_macro("PubPrimaryIndia" + row["policy"] + "Share", row, decimals=1)
    changes = tables.get("replay_changes_by_seed", pd.DataFrame())
    if not changes.empty:
        overall_replay = changes[(changes.dimension == "overall") & (changes.group == "All")
                                & (changes.quantity == "age21_events_2025_2040") & (changes.policy == CONTRAST)]
        macros["PubReplayPositiveDifferenceRuns"] = str(int((overall_replay.value_replay > 0).sum()))
        for dimension, group, fragment in [("overall", "All", ""), ("nationality", "India", "India")]:
            selected = changes[(changes.dimension == dimension) & (changes.group == group)
                               & (changes.quantity == "age21_events_2025_2040")]
            for policy, part in selected.groupby("policy"):
                label = {"Capped": "Capped", "Uncapped": "Uncapped", CONTRAST: "Difference"}[policy]
                baseline, replay = float(part.value_baseline.mean()), float(part.value_replay.mean())
                macros["PubReplayBaseline" + fragment + label + "Ageouts"] = _number(baseline, 1)
                macros["PubReplay" + fragment + label + "Ageouts"] = _number(replay, 1)
                # Pair before calculating precision: covariance between the arms
                # is part of the Monte Carlo error of this specific comparison.
                precision_macro("PubReplayChange" + fragment + label + "Ageouts",
                                numerical_precision(part.value_replay - part.value_baseline), decimals=1)
                macros["PubReplayChange" + fragment + label + "AgeoutsPercent"] = (
                    _number(100 * (replay - baseline) / baseline, 1) if baseline else "--")
                if policy == CONTRAST:
                    macros["PubReplay" + fragment + "DifferencePercentChange"] = (
                        _number(100 * (replay - baseline) / baseline, 1) if baseline else "--")
        macros["PubReplayRuns"] = str(changes.seed.nunique())
    for row in tables.get("replay_historical_fit", pd.DataFrame()).to_dict("records"):
        if row["nationality"] == "India":
            prefix = "PubReplayBaseline" if row["arm"] == "Matched primary" else "PubReplay"
            macros[prefix + "IndiaAnnualResidual"] = _number(row["mean"], 1)
    unfilled = tables.get("replay_unfilled_targets_mean", pd.DataFrame())
    if not unfilled.empty:
        macros["PubReplayHistoricalUnfilledTotal"] = _number(float(unfilled.unfilled_target.sum()), 1)
    historical = tables.get("historical_replay_comparison", pd.DataFrame())
    if not historical.empty:
        for row in historical.to_dict("records"):
            fragment = "" if row["dimension"] == "overall" else row["group"]
            token = {"age21_events_2009_2024": "Ageouts", "principal_backlog_2024": "Backlog"}[row["quantity"]]
            stem = fragment + token
            for label, field in [("Baseline", "value_baseline"), ("Replay", "value_replay"), ("Change", "value")]:
                macros["PubHistorical" + label + stem] = _number(row[field])
            macros["PubHistoricalChange" + stem + "Percent"] = (
                _number(100 * row["value"] / row["value_baseline"], 1) if row["value_baseline"] else "--")
        ageouts = historical[historical.quantity == "age21_events_2009_2024"].set_index("group")
        macros["PubHistoricalAgeoutReductionPercent"] = _number(-100 * ageouts.loc["All", "value"] / ageouts.loc["All", "value_baseline"], 1)
        macros["PubHistoricalBaselineIndiaShare"] = _number(100 * ageouts.loc["India", "value_baseline"] / ageouts.loc["All", "value_baseline"])
        macros["PubHistoricalReplayIndiaShare"] = _number(100 * ageouts.loc["India", "value_replay"] / ageouts.loc["All", "value_replay"])
        macros["PubHistoricalRuns"] = "1"
    return macros


def write_tex(output: Path, tables: dict, plan: dict, *, preview=False):
    tables = {**tables, **single_seed_comparisons(tables, plan)}
    rows = tables.get("age_schedule_comparison", pd.DataFrame())
    if not rows.empty:
        rows = rows[rows.dimension == "overall"]
    prefix = "preview-" if preview else "publication-"
    macro_rows, macros = [], numeric_macros(tables, plan)
    for name, value in macros.items():
        macro_rows.append("\\newcommand{\\" + name + "}{" + value + "}")
    (output / f"{prefix}values.tex").write_text("\n".join(macro_rows) + "\n")
    lines = [r"% Generated from the fixed-size publication campaign; do not edit numbers."]
    if preview:
        lines.append(r"\noindent\textbf{Partial preview; not publication results.}")
    lines += [r"\newcommand{\PublicationAgeExitTable}{%",
        r"\begin{table}[!htbp]", r"\centering",
        r"\caption{Total age-outs by age-based exit schedule, FY2025--FY2040.}",
        r"\label{tab:pub-precision}\label{tab:additional-exit-checks}",
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"\textbf{Age profile} & \textbf{Uncapped age-outs} & \textbf{Capped age-outs} & \textbf{Diff.\ (Cap--Unc)} \\",
        r"\midrule"]
    cases = sorted((case for case in plan["cases"] if case["role"] in ("primary", "sensitivity")),
                   key=lambda case: case["overrides"]["catchall_median_age"])
    for case in cases:
        part = rows[rows.case == case["name"]].set_index("policy")
        if part.empty:
            continue
        difference = part.loc[CONTRAST]
        label = f"{case['overrides']['catchall_median_age']}/{case['overrides']['catchall_near_zero_age']}"
        lines.append(f"{label} & {_number(part.loc['Uncapped','value'])} & {_number(part.loc['Capped','value'])} & {_number(difference['value'])} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}\FloatBarrier", "}"]
    historical = tables.get("historical_replay_comparison", pd.DataFrame())
    if not historical.empty:
        lines += [r"\newcommand{\PublicationReplayTable}{%",
            r"\begin{table}[!htbp]", r"\centering",
            r"\caption{Historical-allocation replay, FY2009--FY2024.}\label{tab:pub-replay}",
            r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
            r"\textbf{Population} & \textbf{Baseline} & \textbf{Replay} & \textbf{Diff.\ (Replay--Base)} \\",
            r"\midrule"]
        for token, title in [("Ageouts", "Age-outs, FY2009--FY2024"),
                             ("Backlog", "Principal backlog, end-FY2024")]:
            if token == "Backlog":
                lines.append(r"\midrule")
            lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + title + r"}} \\")
            groups = [("All nationalities", ""), ("India", "India"), ("China", "China"), ("ROW", "ROW")]
            for label, fragment in groups:
                stem = fragment + token
                lines.append(label + " & " + macros["PubHistoricalBaseline" + stem] + " & "
                    + macros["PubHistoricalReplay" + stem] + " & " + macros["PubHistoricalChange" + stem]
                    + " " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}\FloatBarrier", "}"]
    # Figure environments live beside their first discussion in the authored
    # manuscript. This generated appendix contributes numeric tables only.
    (output / f"{prefix}evidence.tex").write_text("\n".join(lines) + "\n")
    return macros


def generate(campaign: Path, evidence: Path, output: Path, *, allow_partial_preview=False, campaign_data=None) -> dict:
    campaign, evidence, output = campaign.resolve(), evidence.resolve(), output.resolve()
    if output == campaign or campaign.is_relative_to(output):
        raise ValueError("Analysis output must not contain the immutable campaign")
    plan, records, missing, run_hashes = campaign_data if campaign_data is not None else load_campaign(campaign)
    if missing:
        raise ValueError("All declared paired runs are required")
    preview = bool(allow_partial_preview)
    output.mkdir(parents=True, exist_ok=True)
    if preview and (output / "publication-evidence.tex").exists():
        raise ValueError("Use a separate directory for partial previews")
    source = Path(__file__).resolve().parents[1] / "empirical_params.py"
    supply = historical_supply(source)
    observed = pd.read_csv(evidence / "dos_visa_consumption_reconciled.csv")
    all_frames = {name: [] for name in ["projection_runs", "age_tail_runs", "annual_joint_cells",
        "historical_entry_inventory", "child_outcomes", "principal_decomposition", "annual_budgets",
        "historical_outcome_runs"]}
    counts = {"audited_32year_pairs": 0, "annual_budget_checks": 0, "child_cohort_cells": 0,
              "annual_principal_stockflow_checks": 0, "replay_target_cells_both_policies": 0}
    replay_audits = []
    case_by_name = {c["name"]: c for c in plan["cases"]}
    for name, runs in records.items():
        case = case_by_name[name]
        if case["role"] == "cohorts":
            continue
        for value in runs:
            seed = value["seed"]
            audit = audit_record(value, supply)
            if case["role"] == "replay":
                replay_audits.append(audit_replay_targets(value, observed).assign(seed=seed))
                counts["replay_target_cells_both_policies"] += len(value["historical_replay_audit"])
            counts["audited_32year_pairs"] += 1
            counts["annual_budget_checks"] += len(audit["budgets"])
            counts["child_cohort_cells"] += len(audit["cohorts"])
            counts["annual_principal_stockflow_checks"] += len(audit["principal_flows"])
            values = {
                "projection_runs": pd.DataFrame(projection_metric_rows(audit)),
                "age_tail_runs": age_tail_rows(value), "annual_joint_cells": pd.DataFrame(value["annual_cells"]),
                "historical_entry_inventory": pd.DataFrame(value["historical_entry_inventory"]),
                "child_outcomes": audit["child_outcomes"], "principal_decomposition": audit["principal_decomposition"],
                "annual_budgets": audit["budgets"],
            }
            if case["role"] in ("primary", "replay"):
                values["historical_outcome_runs"] = historical_metric_rows(value)
            for key, frame in values.items():
                all_frames[key].append(frame.assign(case=name, role=case["role"], seed=seed))
    tables = {key: pd.concat(frames, ignore_index=True) for key, frames in all_frames.items() if frames}
    if replay_audits:
        tables["replay_unfilled_targets_by_seed"] = pd.concat(replay_audits, ignore_index=True)
        unfilled = tables["replay_unfilled_targets_by_seed"]
        tables["replay_unfilled_targets_mean"] = unfilled.groupby(["year", "category", "nationality"], as_index=False).agg(
            target_visas=("target_visas", "mean"), used_visas=("used_visas", "mean"),
            unfilled_target=("unfilled_target", "mean"), n_paired_seeds=("seed", "nunique"))
    if "projection_runs" not in tables:
        raise ValueError("A preview needs at least one completed 32-year case")
    metric_keys = ["case", "role", "dimension", "group", "quantity", "policy"]
    primary_runs = tables["projection_runs"][tables["projection_runs"].role.isin(["primary", "sensitivity"])]
    tables["projection_summary"] = summarize(primary_runs, metric_keys)
    ageouts = primary_runs[primary_runs.quantity == "age21_events_2025_2040"]
    denominators = ageouts[(ageouts.dimension == "overall") & ageouts.policy.isin(POLICIES)]
    numerators = ageouts[(ageouts.dimension == "nationality") & (ageouts.group == "India") & ageouts.policy.isin(POLICIES)]
    share_keys = ["case", "role", "seed", "policy"]
    shares = numerators[share_keys + ["value"]].merge(denominators[share_keys + ["value"]],
        on=share_keys, validate="one_to_one", suffixes=("_india", "_total"))
    shares["value"] = 100 * shares.value_india / shares.value_total
    if not np.isfinite(shares.value).all():
        raise ValueError("Undefined India share: zero total age-outs")
    tables["india_share_runs"] = shares
    tables["india_share_summary"] = summarize(shares, ["case", "role", "policy"])
    diagnostic_runs = tables["projection_runs"][tables["projection_runs"].role == "replay"]
    tables["replay_metric_summary"] = summarize(diagnostic_runs, metric_keys)
    # Means of the four competing child fates, including the unresolved 2040
    # group, use the same runs as each named short-horizon specification.
    child = tables["child_outcomes"]
    child = child[child.role.isin(["primary", "sensitivity"])]
    child_keys = ["case", "role", "scenario", "entry_cohort_window", "dimension", "group"]
    long_child = child.melt(id_vars=child_keys + ["seed"],
        value_vars=["total_created", "saved", "aged_out", "exited", "still_dependent"], var_name="quantity", value_name="value")
    tables["child_outcome_summary"] = summarize(long_child, child_keys + ["quantity"])
    primary = plan["configuration"]["primary_case"]
    if not diagnostic_runs.empty:
        baseline = tables["projection_runs"][tables["projection_runs"].case == primary]
        # During previews, only retain replay seeds with a completed primary mate.
        available = diagnostic_runs[diagnostic_runs.seed.isin(baseline.seed)]
        if len(available):
            tables["replay_changes_by_seed"] = paired_replay(baseline, available)
            tables["replay_changes_summary"] = summarize(tables["replay_changes_by_seed"], metric_keys[2:])
            historical = tables["historical_outcome_runs"]
            tables["replay_historical_outcomes_by_seed"] = paired_replay(
                historical[historical.case == primary], historical[historical.role == "replay"])
            tables["replay_historical_outcomes_summary"] = summarize_historical_replay(
                tables["replay_historical_outcomes_by_seed"])
    tail_keys = ["case", "role", "scenario", "year", "outcome", "dimension", "group", "threshold_age"]
    tails = tables["age_tail_runs"]
    tails = tails[tails.role.isin(["primary", "sensitivity"])]
    long = tails.melt(id_vars=tail_keys + ["seed"], value_vars=["denominator", "count_at_or_above", "share_pct"],
                      var_name="quantity", value_name="value")
    undefined = int(long.value.isna().sum())
    tables["age_tail_summary"] = summarize(long.dropna(subset=["value"]), tail_keys + ["quantity"])
    tables["historical_annual_residuals"], tables["historical_joint_fit"] = joint_fit(tables["annual_joint_cells"], observed)
    if "replay_changes_by_seed" in tables:
        matched_seeds = set(tables["replay_changes_by_seed"].seed)
        cell_data = tables["annual_joint_cells"]
        selected = cell_data[cell_data.seed.isin(matched_seeds) & cell_data.case.isin(
            [primary] + [case["name"] for case in plan["cases"] if case["role"] == "replay"])
            & (cell_data.year <= 2024) & (cell_data.scenario == "Capped")]
        actual = observed[observed.category.isin(CATEGORIES)].groupby(["year", "nationality"], as_index=False).visas_issued.sum()
        predicted = selected.groupby(["case", "seed", "year", "nationality"], as_index=False).annual_visas.sum()
        residuals = predicted.merge(actual, on=["year", "nationality"], validate="many_to_one")
        residuals["residual"] = residuals.annual_visas - residuals.visas_issued
        run_fit = residuals.groupby(["case", "seed", "nationality"], as_index=False).residual.mean().rename(columns={"residual": "value"})
        run_fit["arm"] = np.where(run_fit.case == primary, "Matched primary", "Historical replay")
        tables["replay_historical_fit_by_seed"] = run_fit
        tables["replay_historical_fit"] = summarize(run_fit, ["arm", "nationality"], intervals=False)
    followup_checks, cohort_rows = [], []
    baseline_records = {value["seed"]: value for value in records[primary]}
    for case in plan["cases"]:
        if case["role"] != "cohorts":
            continue
        for value in records[case["name"]]:
            if value["seed"] not in baseline_records:
                if preview:
                    continue
                raise ValueError("Follow-up has no primary baseline")
            followup_checks.append(audit_followup(baseline_records[value["seed"]], value))
            cohorts = pd.DataFrame(value["cohorts"])
            selected = cohorts[cohorts.entry_year <= 2040].copy()
            cohort_rows.append(selected.assign(seed=value["seed"], observation_end_year=2061))
    if cohort_rows:
        tables["resolved_cohort_outcomes"] = pd.concat(cohort_rows, ignore_index=True)
        tables["resolved_cohort_summary"] = summarize_resolved_cohorts(tables["resolved_cohort_outcomes"])
    tables.update(single_seed_comparisons(tables, plan))
    tables["retention_curves"] = retention_curves(plan["cases"])
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    typography = write_figures(output, tables, plan, preview=preview)
    macros = write_tex(output, tables, plan, preview=preview)
    headline = tables["projection_summary"]
    headline = headline[(headline.case == primary) & (headline.dimension == "overall")
                        & (headline.quantity == "age21_events_2025_2040") & (headline.policy == CONTRAST)]
    actual_mcse = float(headline.mcse_mean.iloc[0]) if len(headline) and pd.notna(headline.mcse_mean.iloc[0]) else None
    precision_checks = []
    selected = tables["projection_summary"]
    selected = selected[(selected.case == primary) & (selected.quantity == "age21_events_2025_2040")
        & ((selected.dimension == "overall") | ((selected.dimension == "nationality") & (selected.group == "India")))]
    for row in selected.to_dict("records"):
        valid_mcse = row["mcse_mean"] if pd.notna(row["mcse_mean"]) else None
        precision_checks.append({"dimension": row["dimension"], "group": row["group"], "policy": row["policy"],
            "n": row["n"], "mcse_mean": valid_mcse})
    report = {"schema_version": 1, "status": "partial_preview_not_for_publication" if preview else "complete",
        "publication_eligible": not preview, "planned_pairs": plan["planned_pairs"],
        "completed_pairs": sum(map(len, records.values())), "missing_pairs": missing,
        "per_case_completed": {name: len(values) for name, values in records.items()},
        "event_window": [2025, 2040], "historical_replay_event_window": [2009, 2024],
        "historical_replay_stock_year": 2024,
        "manuscript_comparison_seed": plan["configuration"].get("seed", 2014),
        "manuscript_comparison_runs_per_case": 1,
        "cohort_entry_end_year": 2040, "cohort_followup_end_year": 2061,
        "mean_interval_method": "Mean +/- Student-t(0.975,n-1) times sample SD/sqrt(n); approximate numerical intervals for primary means. Archived ensemble summaries retain their original precision statistics. Manuscript age-schedule and historical-replay comparisons use one seed without intervals.",
        "primary_difference_mcse": actual_mcse,
        "primary_count_precision_checks": precision_checks,
        "sample_size_rule": plan["configuration"]["sample_size_rule"],
        "sensitivity_budget_rationale": plan["configuration"].get("sensitivity_budget_rationale"),
        "audits": {**counts, "all_passed": True, "followup": followup_checks},
        "undefined_age_share_cells": undefined,
        "coverage": {"2024_waiting_age_census": "Not exported; not inferred", "2024_recipient_event_ages": "Observed export",
                     "2040_waiting_and_recipient_ages": "Observed export"},
        "replay_unfilled_target_scope": "One copy of the shared FY2009–FY2024 history; sums span all 16 years and 15 cells, not one year or both policy copies",
        "limitations": ["Student-t intervals concern fixed-input numerical means, not total model uncertainty.",
            "Replay mean intervals quantify Monte Carlo error in a historical sensitivity comparison, not systematic bias or future allocation accuracy. The two-seed resolved-cohort follow-up remains descriptive.",
            "Cross-hazard histories differ; identical seed labels do not establish individual common shocks.",
            "Historical replay uses DOS issuances as reconstruction targets, not independent validation.",
            "Age-curve retention anchors are not older-age waiting-stock shares; total retention also reflects the retained baseline and service."],
        "figure_typography": typography, "macros": macros,
        "software": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")}},
        "scientific_source": plan["source_fingerprint"],
        "recorded_external_paths": {c["name"]: c["overrides"].get("historical_allocation_path") for c in plan["cases"] if c["role"] == "replay"},
        "resolved_external_input": "data/validation/dos_visa_consumption_reconciled.csv"}
    (output / "analysis_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for relative, expected in run_hashes.items():
        if digest(campaign / relative) != expected:
            raise ValueError("A completed run changed during analysis")
    from . import accounting, figure_style
    from simulation import campaign as campaign_module
    dependencies = [Path(__file__), Path(accounting.__file__), Path(campaign_module.__file__),
                    Path(figure_style.__file__)]
    campaign_inputs = dict(run_hashes)
    manifest = {"schema_version": 1, "status": report["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "release_manifest.json"},
        "campaign_inputs": campaign_inputs,
        "analysis_sources": {p.name: digest(p) for p in dependencies},
        "saved_record_integrity_and_accounting_verified": True, "all_accounting_checks_passed": True}
    (output / "release_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path("data/runs"))
    parser.add_argument("--evidence", type=Path, default=Path("data/validation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--allow-partial-preview", action="store_true")
    args = parser.parse_args(argv)
    report = generate(args.campaign, args.evidence, args.output, allow_partial_preview=args.allow_partial_preview)
    print(json.dumps({key: report[key] for key in ["status", "completed_pairs", "planned_pairs", "primary_difference_mcse"]}, indent=2))


if __name__ == "__main__":
    main()
