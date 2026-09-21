"""Accounting checks for paired simulation records."""
from __future__ import annotations
import ast
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
POLICIES = ("Uncapped", "Capped")
CATEGORIES = tuple(f"EB-{i}" for i in range(1, 6))
NATIONALITIES = ("India", "China", "ROW")
YEARS = np.arange(2009, 2041)
CHILD_FIELDS = ("total_created", "aged_out", "saved", "exited", "still_dependent")


def targeted_record_metrics(record: Dict) -> Dict:
    """Extract one newly executed pair, preserving its explicit seed identity."""
    if record.get("complete") is not True:
        raise ValueError("Targeted result is not marked complete")
    raw = record["raw"]
    years = np.asarray(raw["years"])
    if not np.array_equal(years, np.arange(2009, 2041)):
        raise ValueError("Targeted summary requires each FY2009–2040 exactly once")
    projection = years >= 2025
    def total(field):
        values = np.asarray(raw[field], dtype=float)
        if len(values) != len(years) or not np.isfinite(values).all():
            raise ValueError(f"Invalid targeted metric {field}")
        return float(values[projection].sum())
    result = {"seed": int(record["seed"]), "elapsed_seconds": float(record["elapsed_seconds"])}
    for sc, label in [("unc", "uncapped"), ("cap", "capped")]:
        result[f"{label}_ageouts"] = total(f"annual_aged_out_{sc}")
        result[f"india_{label}_ageouts"] = total(f"annual_aged_out_India_{sc}")
        result[f"{label}_principal_exits"] = total(f"annual_exited_{sc}")
        result[f"{label}_principal_conversions"] = total(f"annual_conversions_{sc}")
        result[f"{label}_visas_consumed"] = total(f"annual_visas_consumed_{sc}")
        result[f"{label}_final_backlog"] = float(raw[f"annual_backlog_total_{sc}"][-1])
        result[f"india_{label}_share_pct"] = 100 * result[f"india_{label}_ageouts"] / result[f"{label}_ageouts"]
    result["ageout_difference"] = result["capped_ageouts"] - result["uncapped_ageouts"]
    result["india_ageout_difference"] = result["india_capped_ageouts"] - result["india_uncapped_ageouts"]
    result["ageout_relative_change_pct"] = 100 * result["ageout_difference"] / result["uncapped_ageouts"]
    result["backlog_difference"] = result["capped_final_backlog"] - result["uncapped_final_backlog"]
    result["exit_difference"] = result["capped_principal_exits"] - result["uncapped_principal_exits"]
    result["conversion_difference"] = result["capped_principal_conversions"] - result["uncapped_principal_conversions"]
    result["backlog_difference_accounting_residual"] = result["backlog_difference"] + result["exit_difference"] + result["conversion_difference"]
    if result["backlog_difference_accounting_residual"] != 0:
        raise ValueError("Targeted policy backlog difference fails principal accounting")
    for field in raw:
        if field.startswith("annual_") and field.endswith("_cap") and field[:-4] + "_unc" in raw:
            left, right = np.asarray(raw[field]), np.asarray(raw[field[:-4] + "_unc"])
            if not np.array_equal(left[~projection], right[~projection]):
                raise ValueError(f"Policies differ before intervention: {field}")
    return result

def historical_supply(source_path: Path) -> dict[int, int]:
    """Read the literal supply table without importing changing live model code."""
    for node in ast.parse(source_path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HISTORICAL_EB_VISA_POOL" for t in node.targets):
            supply = ast.literal_eval(node.value)
            if not set(range(2009, 2025)).issubset(supply):
                raise ValueError("Frozen supply table does not cover the historical window")
            return {year: int(supply.get(year, supply[2024])) for year in range(2009, 2041)}
    raise ValueError("Frozen source has no literal historical visa supply table")

def annual_arrays(record: dict) -> dict[str, np.ndarray]:
    raw = record["raw"]
    arrays = {}
    for key, value in raw.items():
        if key.startswith(("annual_", "cumul_")) and isinstance(value, list):
            array = np.asarray(value, dtype=float)
            if array.shape != (32,) or not np.isfinite(array).all():
                raise ValueError(f"Invalid annual array: {key}")
            arrays[key] = array
    return arrays

def summarize_record(record: dict, supply: dict[int, int]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics = targeted_record_metrics(record)
    arrays = annual_arrays(record)
    raw = record["raw"]
    years = np.asarray(raw["years"])
    future = years >= 2025
    cells = pd.DataFrame(record["annual_cells"])
    key = ["scenario", "year", "category", "nationality"]
    expected_index = pd.MultiIndex.from_product([["Uncapped", "Capped"], years, CATEGORIES, NATIONALITIES], names=key)
    if cells.duplicated(key).any() or set(map(tuple, cells[key].to_numpy())) != set(expected_index):
        raise ValueError("Incomplete or duplicated annual joint cells")
    history = cells[cells.year <= 2024]
    history_key = ["year", "category", "nationality"]
    historical_policy_cells = [history[history.scenario == policy].drop(columns="scenario").set_index(history_key).sort_index()
                               for policy in ("Uncapped", "Capped")]
    if not historical_policy_cells[0].equals(historical_policy_cells[1]):
        raise ValueError("Policies have different historical joint stock/flow cells")
    group_rows, budget_rows = [], []
    for sc, scenario in [("unc", "Uncapped"), ("cap", "Capped")]:
        selected = cells[cells.scenario == scenario]
        for field, column in [("annual_backlog_total", "principal_backlog"), ("annual_visas_consumed", "annual_visas"), ("annual_exited", "annual_principal_exits")]:
            joint = selected.groupby("year")[column].sum().reindex(years).to_numpy()
            if not np.array_equal(joint, arrays[f"{field}_{sc}"]):
                raise ValueError(f"Annual joint cells do not reconcile: {field}/{scenario}")
        for year, consumed in zip(years, arrays[f"annual_visas_consumed_{sc}"]):
            available = supply[int(year)]
            if consumed < 0 or consumed != int(consumed) or consumed > available:
                raise ValueError(f"Annual visa budget failed in {year}/{scenario}")
            budget_rows.append({"scenario": scenario, "year": int(year), "visas_consumed": int(consumed),
                                "visas_available": available, "unused_visas": available - int(consumed), "passed": True})
        for dimension, groups in [("nationality", NATIONALITIES), ("category", CATEGORIES)]:
            ageouts, exits = [], []
            for group in groups:
                token = group.replace("-", "") if dimension == "category" else group
                aged = arrays[f"annual_aged_out_{token}_{sc}"]
                departed = arrays[f"annual_exited_{token}_{sc}"]
                ageouts.append(aged)
                exits.append(departed)
                for counts in (aged, departed):
                    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
                        raise ValueError(f"Group event counts are not nonnegative integers: {group}/{sc}")
                stock = int(selected[(selected.year == 2040) & (selected[dimension] == group)].principal_backlog.sum())
                if dimension == "nationality" and stock != raw[f"annual_backlog_{group}_{sc}"][-1]:
                    raise ValueError("Final nationality backlog does not match joint cells")
                group_rows.append({"scenario": scenario, "dimension": dimension, "group": group,
                                   "age21_events_2025_2040": int(aged[future].sum()),
                                   "principal_exits_2025_2040": int(departed[future].sum()),
                                   "final_principal_backlog_2040": stock})
            if not np.array_equal(np.sum(ageouts, axis=0), arrays[f"annual_aged_out_{sc}"]):
                raise ValueError(f"Annual age-21 events do not add across {dimension}/{scenario}")
            if not np.array_equal(np.sum(exits, axis=0), arrays[f"annual_exited_{sc}"]):
                raise ValueError(f"Annual exits do not add across {dimension}/{scenario}")
        cumulative = arrays[f"cumul_aged_out_{sc}"]
        if not np.array_equal(np.cumsum(arrays[f"annual_aged_out_{sc}"]), cumulative):
            raise ValueError("Annual and cumulative age-21 counts do not reconcile")
    # The event label is chronological age 21, not validated legal age-out status.
    metrics = {key.replace("ageouts", "age21_events").replace("ageout_", "age21_"): value for key, value in metrics.items()}
    return metrics, pd.DataFrame(group_rows), pd.DataFrame(budget_rows)

def group_definitions():
    yield "overall", "All", None, None
    for nationality in NATIONALITIES:
        yield "nationality", nationality, None, nationality
    for category in CATEGORIES:
        yield "category", category, category, None
    for category in CATEGORIES:
        for nationality in NATIONALITIES:
            yield "joint", f"{category} / {nationality}", category, nationality

def select_group(frame: pd.DataFrame, category, nationality) -> pd.DataFrame:
    if category is not None:
        frame = frame[frame.category == category]
    if nationality is not None:
        frame = frame[frame.nationality == nationality]
    return frame

def audit_child_cohorts(record: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohorts = pd.DataFrame(record["cohorts"]).rename(columns={"eb_category": "category"})
    keys = ["scenario", "entry_year", "category", "nationality"]
    if cohorts.duplicated(keys).any():
        raise ValueError("Duplicated child entry-cohort cell")
    for key, allowed in [("scenario", POLICIES), ("entry_year", YEARS), ("category", CATEGORIES), ("nationality", NATIONALITIES)]:
        if not cohorts[key].isin(allowed).all():
            raise ValueError(f"Unknown child cohort {key}")
    counts = cohorts[list(CHILD_FIELDS)].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Child outcomes must be nonnegative integer counts")
    if not np.array_equal(cohorts.total_created, cohorts[list(CHILD_FIELDS[1:])].sum(axis=1)):
        raise ValueError("Child outcome conservation failed within an entry cohort")
    # Zero-child cohort cells are legitimately omitted by the original exporter.
    index = pd.MultiIndex.from_product([POLICIES, YEARS, CATEGORIES, NATIONALITIES], names=keys)
    cohorts = cohorts.set_index(keys)[list(CHILD_FIELDS)].reindex(index, fill_value=0).reset_index()
    for scenario, suffix in [("Uncapped", "unc"), ("Capped", "cap")]:
        aged = int(cohorts.loc[cohorts.scenario == scenario, "aged_out"].sum())
        if aged != int(record["raw"][f"final_cumul_aged_out_{suffix}"]):
            raise ValueError("Cohort age-21 counts disagree with cumulative annual events")
    history_created = cohorts[cohorts.entry_year <= 2024].pivot(index=keys[1:], columns="scenario", values="total_created")
    if not np.array_equal(history_created.Capped, history_created.Uncapped):
        raise ValueError("Paired policies do not share historical child cohorts")
    rows = []
    for scenario in POLICIES:
        for window, selected_years in [("all_entries_2009_2040", YEARS),
                                       ("entries_2009_2024", YEARS[:16]),
                                       ("entries_2025_2040", YEARS[16:])]:
            selected = cohorts[(cohorts.scenario == scenario) & cohorts.entry_year.isin(selected_years)]
            for dimension, group, category, nationality in group_definitions():
                total = select_group(selected, category, nationality)[list(CHILD_FIELDS)].sum()
                row = {"scenario": scenario, "entry_cohort_window": window, "dimension": dimension, "group": group,
                       "outcome_observation_end_year": 2040, **{name: int(total[name]) for name in CHILD_FIELDS}}
                for outcome in CHILD_FIELDS[1:]:
                    row[f"{outcome}_fraction_pct"] = 100 * total[outcome] / total.total_created if total.total_created else np.nan
                rows.append(row)
    totals = pd.DataFrame(rows)
    contrasts = []
    for (window, dimension, group), selected in totals.groupby(["entry_cohort_window", "dimension", "group"], sort=False):
        selected = selected.set_index("scenario")
        differences = {name: int(selected.loc["Capped", name] - selected.loc["Uncapped", name]) for name in CHILD_FIELDS}
        if differences["total_created"] != sum(differences[name] for name in CHILD_FIELDS[1:]):
            raise ValueError("Paired child-outcome difference does not conserve generated children")
        contrasts.append({"entry_cohort_window": window, "dimension": dimension, "group": group,
                          **{f"difference_{key}": value for key, value in differences.items()}})
    contrasts = pd.DataFrame(contrasts)
    all_group = contrasts[(contrasts.entry_cohort_window == "all_entries_2009_2040") & (contrasts.dimension == "overall")].iloc[0]
    raw = record["raw"]
    if not np.array_equal(raw["annual_aged_out_cap"][:16], raw["annual_aged_out_unc"][:16]):
        raise ValueError("Shared historical age-21 event counts differ within the policy pair")
    projected = sum(raw["annual_aged_out_cap"][16:]) - sum(raw["annual_aged_out_unc"][16:])
    if all_group.difference_aged_out != projected:
        raise ValueError("Shared historical age-21 contributions do not cancel in the cohort contrast")
    overall_parts = contrasts[(contrasts.entry_cohort_window != "all_entries_2009_2040") & (contrasts.dimension == "overall")]
    if overall_parts.difference_aged_out.sum() != projected:
        raise ValueError("Preexisting and future-entry cohort contributions fail to sum to the projection contrast")
    joint = contrasts[(contrasts.entry_cohort_window == "all_entries_2009_2040") & (contrasts.dimension == "joint")]
    if joint.difference_aged_out.sum() != projected or len(joint) != 15:
        raise ValueError("Joint child-cell contrasts do not sum to the projection contrast")
    for dimension, groups in [("nationality", NATIONALITIES), ("category", CATEGORIES)]:
        for group in groups:
            token = group.replace("-", "") if dimension == "category" else group
            expected = sum(raw[f"annual_aged_out_{token}_cap"][16:]) - sum(raw[f"annual_aged_out_{token}_unc"][16:])
            value = contrasts[(contrasts.entry_cohort_window == "all_entries_2009_2040") & (contrasts.dimension == dimension) & (contrasts.group == group)].difference_aged_out.iloc[0]
            if value != expected:
                raise ValueError("Cohort contrasts disagree with projection-window age-21 margins")
    return cohorts, totals, contrasts

def audit_principal_flows(record: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing = pd.DataFrame(record["queue_timing"]).copy()
    key = ["scenario", "nationality", "category", "entry_year", "outcome", "model_entry_tenure_years"]
    if timing.duplicated(key).any():
        raise ValueError("Duplicate terminal queue-timing cell")
    numeric = timing[["entry_year", "model_entry_tenure_years", "count"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all() or (timing["count"] < 0).any():
        raise ValueError("Invalid terminal queue counts or dates")
    timing["end_year"] = timing.entry_year + timing.model_entry_tenure_years
    if (not timing.entry_year.isin(YEARS).all() or not timing.end_year.isin(YEARS).all()
            or (timing.model_entry_tenure_years < 0).any()
            or not timing.outcome.isin(["converted", "exited", "still_waiting"]).all()
            or not timing.scenario.isin(POLICIES).all()
            or not timing.category.isin(CATEGORIES).all()
            or not timing.nationality.isin(NATIONALITIES).all()):
        raise ValueError("Terminal queue records have inconsistent categories or timing")
    if not timing.loc[timing.outcome == "still_waiting", "end_year"].eq(2040).all():
        raise ValueError("Outstanding queue records are not censored at FY2040")
    cells = pd.DataFrame(record["annual_cells"])
    rows = []
    for scenario, suffix in [("Uncapped", "unc"), ("Capped", "cap")]:
        for dimension, group, category, nationality in group_definitions():
            selected = select_group(timing[timing.scenario == scenario], category, nationality)
            annual = select_group(cells[cells.scenario == scenario], category, nationality)
            series = lambda frame, year: frame.groupby(year)["count"].sum().reindex(YEARS, fill_value=0).to_numpy(dtype=np.int64)
            entries = series(selected, "entry_year")
            conversions = series(selected[selected.outcome == "converted"], "end_year")
            exits = series(selected[selected.outcome == "exited"], "end_year")
            stock = annual.groupby("year").principal_backlog.sum().reindex(YEARS).to_numpy(dtype=np.int64)
            expected = np.cumsum(entries - conversions - exits)
            if not np.array_equal(expected, stock):
                raise ValueError(f"Principal stock/flow conservation failed: {scenario}/{dimension}/{group}")
            exported_exits = annual.groupby("year").annual_principal_exits.sum().reindex(YEARS).to_numpy(dtype=np.int64)
            if not np.array_equal(exits, exported_exits):
                raise ValueError("Terminal departure timing disagrees with annual principal exit cells")
            if int(selected.loc[selected.outcome == "still_waiting", "count"].sum()) != int(stock[-1]):
                raise ValueError("Terminal waiting records disagree with endpoint principal stock")
            if dimension == "overall":
                for calculated, raw_key in [(conversions, "annual_conversions"), (exits, "annual_exited"), (stock, "annual_backlog_total")]:
                    if not np.array_equal(calculated, record["raw"][f"{raw_key}_{suffix}"]):
                        raise ValueError(f"Terminal reconstruction disagrees with {raw_key}")
            for position, year in enumerate(YEARS):
                rows.append({"scenario": scenario, "dimension": dimension, "group": group, "year": int(year),
                             "starting_backlog": int(stock[position - 1]) if position else 0,
                             "new_principal_records": int(entries[position]), "principal_conversions": int(conversions[position]),
                             "principal_exits": int(exits[position]), "ending_backlog": int(stock[position]),
                             "accounting_residual": int(stock[position] - expected[position])})
    annual = pd.DataFrame(rows)
    decomposition = []
    for (dimension, group), frame in annual.groupby(["dimension", "group"], sort=False):
        policy_totals = {}
        for scenario in POLICIES:
            selected = frame[(frame.scenario == scenario) & (frame.year >= 2025)].sort_values("year")
            policy_totals[scenario] = {"backlog_start_2025": int(selected.starting_backlog.iloc[0]),
                                      "new_principal_records": int(selected.new_principal_records.sum()),
                                      "principal_conversions": int(selected.principal_conversions.sum()),
                                      "principal_exits": int(selected.principal_exits.sum()),
                                      "backlog_end_2040": int(selected.ending_backlog.iloc[-1])}
        diff = {key: policy_totals["Capped"][key] - policy_totals["Uncapped"][key] for key in policy_totals["Capped"]}
        residual = diff["backlog_end_2040"] - (diff["backlog_start_2025"] + diff["new_principal_records"] - diff["principal_conversions"] - diff["principal_exits"])
        if residual or diff["backlog_start_2025"]:
            raise ValueError("Paired projection backlog decomposition or shared starting stock failed")
        decomposition.append({"dimension": dimension, "group": group, **{f"difference_{key}": value for key, value in diff.items()}, "accounting_residual": residual})
    return annual, pd.DataFrame(decomposition)

def audit_record(record: dict, supply: dict) -> dict:
    annual_cells = pd.DataFrame(record["annual_cells"])
    counts = annual_cells[["principal_backlog", "annual_visas", "annual_principal_exits"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("Annual joint stock and flow cells must contain nonnegative integer counts")
    metrics, margins, budgets = summarize_record(record, supply)
    cohorts, outcomes, child_differences = audit_child_cohorts(record)
    flows, decomposition = audit_principal_flows(record)
    for scenario, suffix in [("Uncapped", "unc"), ("Capped", "cap")]:
        all_children = outcomes[(outcomes.scenario == scenario) & (outcomes.entry_cohort_window == "all_entries_2009_2040") & (outcomes.dimension == "overall")].iloc[0]
        spouse_visas = sum(record["raw"][f"annual_visas_consumed_{suffix}"]) - sum(record["raw"][f"annual_conversions_{suffix}"]) - all_children.saved
        if spouse_visas < 0 or spouse_visas != int(spouse_visas):
            raise ValueError("Whole-horizon principal and saved-child visas exceed visa consumption")
    return {"metrics": metrics, "margins": margins, "budgets": budgets, "cohorts": cohorts,
            "child_outcomes": outcomes, "child_differences": child_differences,
            "principal_flows": flows, "principal_decomposition": decomposition}

def projection_metric_rows(audit: dict) -> list[dict]:
    rows = []
    totals = audit["metrics"]
    for quantity, field in [("age21_events_2025_2040", "age21_events"),
                            ("principal_exits_2025_2040", "principal_exits"),
                            ("principal_conversions_2025_2040", "principal_conversions"),
                            ("principal_backlog_2040", "final_backlog")]:
        cap, unc = totals[f"capped_{field}"], totals[f"uncapped_{field}"]
        for policy, value in [("Capped", cap), ("Uncapped", unc), ("Capped minus uncapped", cap - unc)]:
            rows.append({"dimension": "overall", "group": "All", "quantity": quantity, "policy": policy, "value": value})
    margins = audit["margins"]
    for (dimension, group), selected in margins.groupby(["dimension", "group"], sort=False):
        selected = selected.set_index("scenario")
        for field, quantity in [("age21_events_2025_2040", "age21_events_2025_2040"),
                                ("principal_exits_2025_2040", "principal_exits_2025_2040"),
                                ("final_principal_backlog_2040", "principal_backlog_2040")]:
            cap, unc = selected.loc["Capped", field], selected.loc["Uncapped", field]
            for policy, value in [("Capped", cap), ("Uncapped", unc), ("Capped minus uncapped", cap - unc)]:
                rows.append({"dimension": dimension, "group": group, "quantity": quantity, "policy": policy, "value": value})
    joint = audit["child_differences"]
    joint = joint[(joint.dimension == "joint") & (joint.entry_cohort_window == "all_entries_2009_2040")]
    for row in joint.itertuples():
        rows.append({"dimension": "joint", "group": row.group, "quantity": "age21_events_2025_2040",
                     "policy": "Capped minus uncapped", "value": row.difference_aged_out})
    return rows
