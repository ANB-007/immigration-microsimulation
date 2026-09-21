"""Refresh numerical cells and original figure meanings from a selected ensemble.

This module never runs simulations or edits the authored manuscript. It emits a
candidate containing only table-cell replacements, a reviewable patch ledger,
numeric values for the author's prose, and figures with their plotting data.
Annual events end in 2040; eventual entry-cohort outcomes require separate
same-model records followed through 2061, including subsequent arrivals.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from .figure_style import (APPLICANT_COLORS, FONT_SIZE, NATIONALITY_COLORS,
                           OUTCOME_COLORS, PLOT_DPI, SCENARIO_COLORS, manuscript_style)


YEARS = tuple(range(2009, 2041))
NATIONALITIES = ("India", "China", "ROW")
CATEGORIES = tuple(f"EB-{i}" for i in range(1, 6))
POLICIES = ("Uncapped", "Capped", "Capped minus uncapped")
FATES = ("saved", "aged_out", "exited", "still_dependent")
RESOLVED_FATES = ("saved", "aged_out", "exited")
PERIODS = {"historical": (2009, 2024), "projection": (2025, 2040), "full": (2009, 2040)}
COHORT_ASSETS = {
    "eb2_india_capped": ("EB-2", "India", "Capped"),
    "eb3_india_capped": ("EB-3", "India", "Capped"),
    "eb2_china_capped": ("EB-2", "China", "Capped"),
    "eb4_other_uncapped": ("EB-4", "ROW", "Uncapped"),
    "eb4_other_capped": ("EB-4", "ROW", "Capped"),
}
HISTORICAL_SENSITIVITY_TABLES = {
    "tab:india-exit-aggregates", "tab:india-exit-capped-shares", "tab:india-exit-uncapped-shares",
    "tab:visa-allocation-ageouts", "tab:spouse-prob-ageouts", "tab:child-age-ageouts",
    "tab:eb-exit-ageouts", "tab:additional-exit-checks",
}
TABLE_RE = re.compile(r"\\begin\{table\}(?:\[[^]]*\])?[\s\S]*?\\end\{table\}")
ROW_END = re.compile(r"(?<!\\)\\\\(?:\[[^]]*\])?")


class MissingExportError(ValueError):
    """The requested figure cannot be recovered from the recorded experiment."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _country(value):
    return {"Other": "ROW", "Rest of World": "ROW"}.get(value, value)


def scientific_identity(identity: dict) -> dict:
    """Include every recorded identity field except the intentional run horizon/seed."""
    if not isinstance(identity.get("files"), dict) or not identity["files"]:
        raise ValueError("Run identity lacks scientific source/input hashes")
    return {key: value for key, value in identity.items()
            if key not in {"seed", "years", "start_year"}}


def load_records(directory: Path, end_year: int, *, expected_runs=None, expected_seeds=None, supplied_records=None):
    directory = Path(directory)
    folder = directory / "runs" if (directory / "runs").is_dir() else directory
    paths = ([(folder / f"seed_{r['seed']}.json", r) for r in supplied_records]
             if supplied_records is not None else [(p, None) for p in sorted(folder.glob("seed_*.json"))])
    if not paths:
        raise MissingExportError(f"No completed seed records in {folder}")
    records, hashes, common, seen = [], {}, None, set()
    for path, supplied in paths:
        if supplied is None:
            hashes[str(path.resolve())] = digest(path)
        record = supplied if supplied is not None else json.loads(path.read_text())
        identity, seed = record.get("identity", {}), record.get("seed")
        if record.get("complete") is not True or type(seed) is not int or identity.get("seed") != seed:
            raise ValueError(f"Incomplete or inconsistent seed identity: {path}")
        if path.stem != f"seed_{seed}" or seed in seen:
            raise ValueError(f"Duplicate or misnamed seed record: {path}")
        seen.add(seed)
        if identity.get("start_year") != 2009 or identity.get("years") != end_year - 2008:
            raise ValueError(f"Wrong observation horizon in {path}; expected 2009--{end_year}")
        if record.get("raw", {}).get("years") != list(range(2009, end_year + 1)):
            raise ValueError(f"Raw annual years disagree with identity in {path}")
        current = scientific_identity(identity)
        if common is not None and current != common:
            raise ValueError("Selected records mix scientific sources, inputs, or parameters")
        common = current
        # The age-profile and queue-timing exports can be much larger than the
        # tables needed here. Hash the complete record, but do not retain unused
        # individual-age aggregates for every ensemble member in memory.
        records.append({key: record[key] for key in
                        ("identity", "complete", "seed", "raw", "cohorts", "annual_cells")})
    if expected_runs is not None and len(records) != expected_runs:
        raise ValueError(f"Expected {expected_runs} completed pairs, found {len(records)}")
    if expected_seeds is not None and sorted(seen) != sorted(expected_seeds):
        raise ValueError("Completed seeds do not match the declared selection")
    return sorted(records, key=lambda record: record["seed"]), hashes, common


@lru_cache(maxsize=128)
def _mean_t_critical(n: int) -> float:
    # scipy.stats.t.ppf is the inverse CDF; use n-1 estimated-variance degrees
    # of freedom, not the empirical distribution of simulation outcomes.
    # https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.t.html
    from scipy.stats import t
    return float(t.ppf(.975, df=n - 1))


def precision(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("A summary requires finite one-dimensional observations")
    multiple = len(a) > 1
    mcse = float(a.std(ddof=1) / np.sqrt(len(a))) if multiple else None
    halfwidth = _mean_t_critical(len(a)) * mcse if multiple else None
    return {"n": len(a), "mean": float(a.mean()),
            "sd_across_runs": float(a.std(ddof=1)) if multiple else None,
            "mcse_mean": mcse,
            "mean_ci95_low": float(a.mean() - halfwidth) if multiple else None,
            "mean_ci95_high": float(a.mean() + halfwidth) if multiple else None,
            "p025": float(np.quantile(a, .025)) if multiple else None,
            "p975": float(np.quantile(a, .975)) if multiple else None}


def summarize(frame: pd.DataFrame, keys: list[str], value="value") -> pd.DataFrame:
    rows = []
    for identity, group in frame.groupby(keys, sort=False, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        rows.append({**dict(zip(keys, identity)), **precision(group[value])})
    return pd.DataFrame(rows)


def _counts(raw, key, length):
    if key not in raw:
        raise MissingExportError(f"Required export is absent: {key}")
    a = np.asarray(raw[key], dtype=float)
    if a.shape != (length,) or not np.isfinite(a).all() or (a < 0).any() or not np.equal(a, np.floor(a)).all():
        raise ValueError(f"Invalid annual nonnegative integer counts: {key}")
    return a


def summarize_projection(records: list[dict], *, allow_partial=False) -> tuple[dict, list[str]]:
    if not records or len({record["seed"] for record in records}) != len(records):
        raise ValueError("Projection records require distinct paired seeds")
    metrics, shares, annual, applicant, cells, missing = [], [], [], [], [], []
    for record in records:
        raw, seed = record["raw"], record["seed"]
        if raw["years"] != list(YEARS):
            raise ValueError("Projection records must observe annual events through 2040")
        by_policy = {}
        for policy, suffix in zip(POLICIES[:2], ("unc", "cap")):
            values = {}
            for metric, field in (("ageouts", "aged_out"), ("exits", "exited")):
                total = _counts(raw, f"annual_{field}_{suffix}", 32)
                for dimension, groups in (("overall", ("All",)), ("nationality", NATIONALITIES), ("category", CATEGORIES)):
                    arrays = []
                    for group in groups:
                        extra = "" if group == "All" else "_" + group.replace("-", "")
                        a = _counts(raw, f"annual_{field}{extra}_{suffix}", 32)
                        arrays.append(a)
                        if metric == "ageouts":
                            annual.extend({"seed": seed, "scenario": policy, "year": year,
                                           "dimension": dimension, "group": group, "value": float(value)}
                                          for year, value in zip(YEARS, a))
                        for period, (start, end) in PERIODS.items():
                            selected, denominator = a[start - 2009:end - 2008], total[start - 2009:end - 2008].sum()
                            key = (metric, period, dimension, group)
                            values[key] = float(selected.sum())
                            if denominator > 0:
                                shares.append({"seed": seed, "metric": metric, "period": period,
                                               "dimension": dimension, "group": group, "scenario": policy,
                                               "value": 100 * selected.sum() / denominator})
                    if not np.array_equal(np.sum(arrays, axis=0), total):
                        raise ValueError(f"{dimension} {metric} do not sum to the annual total")
            for name, field in (("visas", "visas_consumed"), ("conversions", "conversions")):
                a = _counts(raw, f"annual_{field}_{suffix}", 32)
                for period, (start, end) in PERIODS.items():
                    values[name, period, "overall", "All"] = float(a[start - 2009:end - 2008].sum())
            backlog = _counts(raw, f"annual_backlog_total_{suffix}", 32)
            values["backlog", "end2040", "overall", "All"] = float(backlog[-1])
            for check, value in (("final_backlog_total", backlog[-1]),
                                 ("final_cumul_aged_out", values["ageouts", "full", "overall", "All"]),
                                 ("final_cumul_exited", values["exits", "full", "overall", "All"]),
                                 ("final_cumul_visas", values["visas", "full", "overall", "All"])):
                if check + "_" + suffix in raw and raw[check + "_" + suffix] != value:
                    raise ValueError(f"Annual and final counters disagree: {check}")
            try:
                spouse_key = f"annual_spouse_conversions_{suffix}"
                if spouse_key not in raw and f"annual_converted_spouses_{suffix}" in raw:
                    spouse_key = f"annual_converted_spouses_{suffix}"
                spouses = _counts(raw, spouse_key, 32)
                children = _counts(raw, f"annual_children_saved_{suffix}", 32)
            except MissingExportError as error:
                if not allow_partial:
                    raise
                missing.append(str(error))
            else:
                principals = _counts(raw, f"annual_conversions_{suffix}", 32)
                visas = _counts(raw, f"annual_visas_consumed_{suffix}", 32)
                if not np.array_equal(principals + spouses + children, visas):
                    raise ValueError("Applicant-type conversions do not sum to annual visas")
                applicant.extend({"seed": seed, "scenario": policy, "year": year, "principals": p,
                                  "spouses": s, "children": c, "visas": v}
                                 for year, p, s, c, v in zip(YEARS, principals, spouses, children, visas))
            by_policy[policy] = values
        for key in by_policy["Uncapped"]:
            metric, period, dimension, group = key
            for policy in POLICIES:
                value = by_policy[policy][key] if policy in by_policy else by_policy["Capped"][key] - by_policy["Uncapped"][key]
                metrics.append({"seed": seed, "metric": metric, "period": period,
                                "dimension": dimension, "group": group, "scenario": policy, "value": value})
        for period in PERIODS:
            unc = by_policy["Uncapped"]["ageouts", period, "overall", "All"]
            cap = by_policy["Capped"]["ageouts", period, "overall", "All"]
            if unc:
                metrics.append({"seed": seed, "metric": "relative_ageout_change_pct", "period": period,
                                "dimension": "overall", "group": "All", "scenario": POLICIES[2],
                                "value": 100 * (cap - unc) / unc})
        for metric in ("ageouts", "exits", "visas", "conversions"):
            if by_policy["Uncapped"][metric, "historical", "overall", "All"] != by_policy["Capped"][metric, "historical", "overall", "All"]:
                raise ValueError("Policies must share the historical reconstruction")
        cells.extend({**row, "seed": seed, "nationality": _country(row["nationality"])}
                     for row in record["annual_cells"] if row["year"] <= 2024)
    frames = {"metric_runs": pd.DataFrame(metrics), "share_runs": pd.DataFrame(shares),
              "annual_events": pd.DataFrame(annual), "annual_applicant_types": pd.DataFrame(applicant),
              "historical_cells": pd.DataFrame(cells)}
    keys = ["metric", "period", "dimension", "group", "scenario"]
    frames["metric_summary"] = summarize(frames["metric_runs"], keys)
    frames["share_summary"] = summarize(frames["share_runs"], keys)
    # Pair annual contrasts before estimating their spread.
    a = frames["annual_events"].pivot(index=["seed", "year", "dimension", "group"], columns="scenario", values="value")
    if a.isna().any().any():
        raise ValueError("Annual series lack a paired policy member")
    delta = (a.Capped - a.Uncapped).rename("value").reset_index().assign(scenario=POLICIES[2])
    frames["annual_events"] = pd.concat([frames["annual_events"], delta], ignore_index=True)
    frames["annual_summary"] = summarize(frames["annual_events"], ["year", "dimension", "group", "scenario"])
    if missing:
        # Never draw an apparent ensemble mean from just the runs with exports.
        frames["annual_applicant_types"] = pd.DataFrame()
    return frames, sorted(set(missing))


def summarize_cohorts(records: list[dict]) -> dict[str, pd.DataFrame]:
    counts, shares, rates = [], [], []
    for record in records:
        raw = record["raw"]
        if raw["years"] != list(range(2009, 2062)):
            raise ValueError("Eventual cohort outcomes require follow-up through FY2061")
        frame = pd.DataFrame(record["cohorts"])
        frame["nationality"] = frame.nationality.map(_country)
        keys = ["scenario", "entry_year", "nationality", "eb_category"]
        if frame.duplicated(keys).any() or set(frame.scenario) != set(POLICIES[:2]):
            raise ValueError("Missing policies or duplicate cohort cells")
        a = frame[["total_created", *FATES]].to_numpy(dtype=float)
        if not np.isfinite(a).all() or (a < 0).any() or not np.equal(a, np.floor(a)).all():
            raise ValueError("Invalid child cohort counts")
        if not frame.total_created.eq(frame[list(FATES)].sum(axis=1)).all():
            raise ValueError("Child cohort conservation failed")
        for policy, suffix in zip(POLICIES[:2], ("unc", "cap")):
            policy_frame = frame[frame.scenario == policy]
            if policy_frame.aged_out.sum() != sum(raw[f"annual_aged_out_{suffix}"]):
                raise ValueError("Long-horizon cohort age-outs disagree with annual events")
            if f"annual_children_saved_{suffix}" in raw and policy_frame.saved.sum() != sum(raw[f"annual_children_saved_{suffix}"]):
                raise ValueError("Long-horizon cohort saves disagree with annual conversions")
            if policy_frame.loc[policy_frame.entry_year > 2040, "total_created"].sum() <= 0:
                raise ValueError("Cohort follow-up must retain post-2040 entrants")
        selected = frame[(frame.entry_year <= 2040) & (frame.total_created > 0)].copy()
        if selected.still_dependent.sum() != 0:
            raise ValueError("Entry cohorts through 2040 retain unresolved children at 2061")
        for row in selected.to_dict("records"):
            counts.append({**row, "seed": record["seed"], "observation_end_year": 2061})
            shares.extend({**{key: row[key] for key in keys}, "seed": record["seed"],
                           "fate": fate, "value": 100 * row[fate] / row["total_created"],
                           "observation_end_year": 2061} for fate in FATES)
        for (policy, nationality, category), group in selected.groupby(["scenario", "nationality", "eb_category"]):
            for period in ("full", "projection"):
                start, end = PERIODS[period]
                part = group[group.entry_year.between(start, end)]
                if len(part):
                    rates.append({"seed": record["seed"], "scenario": policy, "nationality": nationality,
                                  "category": category, "period": period, "nonempty_entry_cohorts": len(part),
                                  "value": float((100 * part.saved / part.total_created).mean())})
    s, r = pd.DataFrame(shares), pd.DataFrame(rates)
    share_summary = summarize(s, ["scenario", "entry_year", "nationality", "eb_category", "fate", "observation_end_year"])
    rate_summary = summarize(r, ["scenario", "nationality", "category", "period"])
    # The separately declared two-run follow-up is descriptive. Preserve means
    # and across-run dispersion, without exporting apparent cohort probability
    # bounds from two observations.
    for frame in (share_summary, rate_summary):
        frame.drop(columns=["mean_ci95_low", "mean_ci95_high", "p025", "p975"], inplace=True)
        frame["interval_scope"] = "No interval: descriptive long-follow-up comparison"
    return {"cohort_counts": pd.DataFrame(counts), "cohort_share_runs": s, "cohort_save_rate_runs": r,
            "cohort_share_summary": share_summary, "cohort_save_rates": rate_summary}


def verify_followup_prefix(projections: list[dict], cohorts: list[dict]) -> None:
    """Extending an overlapping seed must preserve its first 32 annual steps."""
    initial = {record["seed"]: record for record in projections}
    for record in cohorts:
        if record["seed"] not in initial:
            continue
        reference = initial[record["seed"]]["raw"]
        for key, values in reference.items():
            if isinstance(values, list) and key.startswith(("annual_", "cumul_")):
                if record["raw"].get(key, [])[:len(YEARS)] != values:
                    raise ValueError(f"Long follow-up changes the matched FY2009--2040 prefix: seed {record['seed']}, {key}")


def historical_fit(cells: pd.DataFrame, observed: pd.DataFrame, n_runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = observed.copy()
    observed["nationality"] = observed.nationality.map(_country)
    observed = observed[observed.category.isin(CATEGORIES) & observed.year.between(2009, 2024)]
    keys = ["year", "category", "nationality"]
    expected = pd.MultiIndex.from_product([range(2009, 2025), CATEGORIES, NATIONALITIES], names=keys)
    if observed.duplicated(keys).any() or set(map(tuple, observed[keys].to_numpy())) != set(expected):
        raise ValueError("Observed benchmark must contain all 240 historical joint cells")
    values = observed.visas_issued.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Historical observed visas must be nonnegative integer counts")
    if cells.duplicated(["seed", "scenario", *keys]).any():
        raise ValueError("Duplicate historical model cells")
    paired = cells.pivot(index=["seed", *keys], columns="scenario", values="annual_visas")
    if paired.isna().any().any() or not paired.Capped.eq(paired.Uncapped).all():
        raise ValueError("Historical model allocations differ between paired policies")
    if len(paired) != n_runs * 240:
        raise ValueError("Incomplete historical model cells")
    model = cells[cells.scenario == "Capped"].groupby(keys).annual_visas.mean().rename("model")
    annual = observed.set_index(keys).visas_issued.rename("observed").to_frame().join(model).reset_index()
    if annual[["observed", "model"]].isna().any().any():
        raise ValueError("Model and observed joint cells do not align")
    rows = []
    for dimension, columns in (("overall", []), ("category", ["category"]), ("nationality", ["nationality"]), ("joint", ["category", "nationality"])):
        groups = [((), annual)] if not columns else annual.groupby(columns, sort=False)
        for group, selected in groups:
            group = group if isinstance(group, tuple) else (group,)
            yearly = selected.groupby("year")[["observed", "model"]].sum()
            obs, mod = yearly.observed.to_numpy(), yearly.model.to_numpy()
            error = mod - obs
            corr = float(np.corrcoef(obs, mod)[0, 1]) if np.std(obs) > 0 and np.std(mod) > 0 else None
            rows.append({"dimension": dimension, "group": "/".join(group) if group else "All",
                         "n_runs": n_runs, "mean_observed": float(obs.mean()), "mean_model": float(mod.mean()),
                         "bias": float(error.mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
                         "wape": float(np.abs(error).sum() / obs.sum() * 100) if obs.sum() else None,
                         "correlation": corr})
    return pd.DataFrame(rows), annual


def number(value, digits=0):
    if value is None or not np.isfinite(value):
        return "---"
    if abs(value) < .5 * 10 ** -digits:
        value = 0
    return f"{value:,.{digits}f}".replace(",", "{,}").replace("-", "−")


def interval(row):
    """Conditional 95% Student-t confidence interval for the estimated mean."""
    if row["mean_ci95_low"] is None or pd.isna(row["mean_ci95_low"]):
        return "---"
    separator = " to " if row["mean_ci95_low"] < 0 else "--"
    return number(row["mean_ci95_low"]) + separator + number(row["mean_ci95_high"])


def _plain(cell):
    cell = re.sub(r"%[^\n]*", "", cell)
    cell = re.sub(r"\\(?:toprule|midrule|bottomrule|botrule|addlinespace)\b", "", cell)
    cell = re.sub(r"\\(?:cmidrule|cline)(?:\([^)]*\))?\{[^}]*\}", "", cell)
    cell = re.sub(r"\\(?:textbf|textit)\{([^}]*)\}", r"\1", cell)
    return " ".join(cell.replace("--", "-").replace("–", "-").split())


def replace_table_cells(source: str, label: str, replacements: dict[tuple, dict[int, str]]):
    """Replace identified cells; preserve every other byte, including whitespace."""
    matches = [m for m in TABLE_RE.finditer(source) if f"\\label{{{label}}}" in m.group()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one table with label {label}")
    match = matches[0]
    body, patches, seen, prior_country = match.group(), [], set(), None
    data_start = body.index(r"\begin{tabular}")
    cursor = data_start
    changes = []
    for ending in ROW_END.finditer(body, data_start):
        row = body[cursor:ending.start()]
        cells = row.split("&")
        clean = [_plain(cell) for cell in cells]
        first = _country(clean[0])
        if first in NATIONALITIES:
            prior_country = first
        if label == "tab:save_rates" and len(cells) == 4:
            identity = (first or prior_country, clean[1])
        else:
            identity = next((key for key in replacements if tuple(_country(x) for x in clean[:len(key)]) == key), None)
        if identity in replacements:
            if identity in seen:
                raise ValueError(f"Duplicate data row {label}: {identity}")
            seen.add(identity)
            offsets = [0]
            for cell in cells[:-1]:
                offsets.append(offsets[-1] + len(cell) + 1)
            for column, new in replacements[identity].items():
                if column >= len(cells):
                    raise ValueError(f"Wrong column count in {label}: {identity}")
                old = cells[column]
                left, right = len(old) - len(old.lstrip()), len(old.rstrip())
                if not old.strip():
                    raise ValueError(f"Refusing to replace an empty numeric cell in {label}")
                start = cursor + offsets[column] + left
                end = cursor + offsets[column] + right
                changes.append((start, end, new))
                patches.append({"label": label, "row": list(identity), "column": column,
                                "before": old.strip(), "after": new})
        cursor = ending.end()
    if seen != set(replacements):
        raise ValueError(f"Missing numerical rows in {label}: {set(replacements) - seen}")
    for start, end, value in sorted(changes, reverse=True):
        body = body[:start] + value + body[end:]
    return source[:match.start()] + body + source[match.end():], patches


def table_replacements(frames: dict) -> dict:
    summary = frames["metric_summary"].set_index(["metric", "period", "dimension", "group", "scenario"])
    shares = frames["share_summary"].set_index(["metric", "period", "dimension", "group", "scenario"])
    get = lambda metric, period, dimension, group, policy: summary.loc[(metric, period, dimension, group, policy)]
    result = {}
    rows = {}
    definitions = [("Final backlog (all nationalities)", "backlog", "end2040", POLICIES[:2]),
                   ("Final backlog (Cap - Unc)", "backlog", "end2040", POLICIES[2:]),
                   ("Cumulative visas consumed", "visas", "full", POLICIES[:2]),
                   ("Cumulative queue exits", "exits", "full", POLICIES[:2]),
                   ("Cumulative exits (Cap - Unc)", "exits", "full", POLICIES[2:])]
    for text, metric, period, policies in definitions:
        for policy in policies:
            row = get(metric, period, "overall", "All", policy)
            shown_policy = "Capped - Uncapped" if policy == POLICIES[2] else policy
            rows[text, shown_policy] = {2: number(row["mean"]), 3: "---" if metric == "visas" and row["sd_across_runs"] == 0 else interval(row)}
    result["tab:agg_backlog"] = rows
    for label, metric, period, dimension, groups in (
        ("tab:exits_nat", "exits", "projection", "nationality", NATIONALITIES),
        ("tab:exits_cat", "exits", "projection", "category", CATEGORIES),
        ("tab:ageouts_nat_full", "ageouts", "full", "nationality", NATIONALITIES),
        ("tab:ageouts_nat_proj", "ageouts", "projection", "nationality", NATIONALITIES),
        ("tab:ageouts_cat_full", "ageouts", "full", "category", CATEGORIES),
        ("tab:ageouts_cat_proj", "ageouts", "projection", "category", CATEGORIES),
    ):
        rows = {}
        for group in (*groups, "All"):
            dim = "overall" if group == "All" else dimension
            unc, cap, delta = (get(metric, period, dim, group, policy) for policy in POLICIES)
            share = lambda policy: number(shares.loc[(metric, period, dim, group, policy), "mean"], 1) + r"\%"
            cells = {1: number(unc["mean"]), 2: interval(unc), 3: share("Uncapped"),
                     4: number(cap["mean"]), 5: interval(cap), 6: share("Capped"), 7: number(delta["mean"])}
            if metric == "ageouts" and dimension == "category":
                cells[8] = interval(delta)
            shown = ("Net" if metric == "exits" else "Total") if group == "All" else group
            rows[shown,] = cells
        result[label] = rows
    rows = {}
    for period, display in (("historical", "Reconstruction (FY2009-FY2024)"), ("projection", "Projection (FY2025-FY2040)"), ("full", "Full simulation (FY2009-FY2040)")):
        for policy in POLICIES[:2]:
            row = get("ageouts", period, "overall", "All", policy)
            rows[display, policy] = {2: number(row["mean"]), 3: interval(row)}
    result["tab:ageouts_agg"] = rows
    annual = frames["annual_summary"].set_index(["year", "dimension", "group", "scenario"])
    rows = {}
    for year in range(2025, 2041):
        cells = {}
        for i, policy in enumerate(POLICIES):
            row = annual.loc[year, "overall", "All", policy]
            cells[1 + i * 2], cells[2 + i * 2] = number(row["mean"]), interval(row)
        rows[str(year),] = cells
    result["tab:ageouts_annual"] = rows
    result.update(validation_table_replacements(frames["historical_fit"]))
    if "cohort_save_rates" in frames:
        rates = frames["cohort_save_rates"].set_index(["scenario", "nationality", "category", "period"])
        result["tab:save_rates"] = {(country, category): {
            i: number(rates.loc["Capped", country, category, period]["mean"], 2) + r"\%"
            for i, period in ((2, "full"), (3, "projection"))}
            for country in NATIONALITIES for category in CATEGORIES}
    return result


def validation_table_replacements(historical: pd.DataFrame) -> dict:
    """Use WAPE consistently while preserving the existing validation columns."""
    fit = historical.set_index(["dimension", "group"])
    result = {}
    for dimension, label, groups in (("category", "tab:fit_category", CATEGORIES), ("nationality", "tab:fit_nationality", NATIONALITIES)):
        rows = {}
        for group in groups:
            row = fit.loc[dimension, group]
            rows[group,] = {i: number(row[name], digits) for i, name, digits in (
                (1, "mean_observed", 1), (2, "mean_model", 1), (3, "wape", 2),
                (4, "rmse", 1), (5, "bias", 1), (6, "correlation", 3))}
        result[label] = rows
    result["tab:additional-joint-fit"] = {(category, country): {
        i: number(fit.loc["joint", category + "/" + country][name], 1)
        for i, name in ((2, "mean_observed"), (3, "mean_model"), (4, "bias"), (5, "wape"))}
        for category in CATEGORIES for country in NATIONALITIES}
    return result


def draw_figures(output: Path, frames: dict) -> dict:
    """Draw at each original inclusion width, so final-size type is >=10pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.ticker import PercentFormatter, StrMethodFormatter

    figures = {}
    style = manuscript_style()

    def save(fig, stem, fraction, data):
        fig.canvas.draw()
        minimum = min(x.get_fontsize() for x in fig.findobj(Text) if x.get_visible() and x.get_text().strip())
        if minimum < FONT_SIZE or abs(fig.get_figwidth() - 5.5 * fraction) > 1e-8:
            raise ValueError("Figure font/width contract failed")
        fig.savefig(output / f"{stem}.png", dpi=PLOT_DPI, metadata={"Creator": "Recorded simulation results"})
        data.to_csv(output / f"{stem}.csv", index=False)
        figures[stem] = {"width_inches": fig.get_figwidth(), "height_inches": fig.get_figheight(),
                         "minimum_font_points": minimum, "textwidth_fraction": fraction,
                         "numeric_source": stem + ".csv", "format": "png", "dpi": PLOT_DPI,
                         "font_family": style["font.family"]}
        plt.close(fig)

    def stacked(data, columns, labels, colors, stem, fraction, xlabel, ylabel, percent=False,
                legend_columns=2):
        fig, ax = plt.subplots(figsize=(5.5 * fraction, 3.2), layout="constrained")
        x = data.iloc[:, 0].to_numpy()
        artists = ax.stackplot(x, *[data[c].to_numpy() for c in columns],
                               colors=[colors[c] for c in columns], edgecolor="white",
                               linewidth=.3, labels=labels, alpha=.9)
        for artist, hatch in zip(artists, ("", "///", "..", "xx")):
            artist.set_hatch(hatch)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(x.min(), x.max()), ylim=(0, 100) if percent else (0, None))
        ax.set_xticks([2009, 2020, 2030, 2040] if x.min() == 2009 else [2025, 2030, 2035, 2040])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0) if percent else StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.26), ncol=legend_columns,
                  frameon=False, handlelength=1.3, columnspacing=.8)
        save(fig, stem, fraction, data)

    with plt.rc_context(style):
        annual = frames["annual_summary"]
        selected = annual[(annual.dimension == "overall") & (annual.year >= 2025) & annual.scenario.isin(POLICIES[:2])]
        fig, ax = plt.subplots(figsize=(5.5 * .9, 3.2), layout="constrained")
        for policy, linestyle in (("Uncapped", "--"), ("Capped", "-")):
            rows = selected[selected.scenario == policy].sort_values("year")
            ax.plot(rows.year, rows["mean"], linestyle=linestyle, color=SCENARIO_COLORS[policy],
                    marker="o" if policy == "Uncapped" else "s", label=policy)
        ax.set(xlabel="Fiscal year", ylabel="Children aged out", xlim=(2025, 2040), ylim=(0, None), xticks=[2025, 2030, 2035, 2040])
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.legend(frameon=False)
        save(fig, "annual_age_outs_by_scenario", .9, selected)
        fit = frames["historical_fit_annual"].groupby(["year", "nationality"])[["observed", "model"]].sum().reset_index()
        for country, stem in (("India", "india_validation"), ("China", "china_validation"), ("ROW", "row_validation")):
            data = fit[fit.nationality == country].sort_values("year")
            fig, ax = plt.subplots(figsize=(5.5 * .7, 2.8), layout="constrained")
            ax.plot(data.year, data.observed, "--s", color=NATIONALITY_COLORS[country], alpha=.7, label="Recorded")
            ax.plot(data.year, data.model, "-o", color=NATIONALITY_COLORS[country], label="Model mean")
            ax.set(xlabel="Fiscal year", ylabel="Visas", xlim=(2009, 2024), ylim=(0, None), xticks=[2009, 2014, 2019, 2024])
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
            ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.2), handlelength=1.3, columnspacing=.8)
            save(fig, stem, .7, data)
        for policy in POLICIES[:2]:
            part = annual[(annual.dimension == "nationality") & (annual.year >= 2025) & (annual.scenario == policy)]
            data = part.pivot(index="year", columns="group", values="mean").reset_index()
            stacked(data, NATIONALITIES, NATIONALITIES, NATIONALITY_COLORS, policy.lower() + "_by_nationality", .65, "Fiscal year", "Children aged out")
        applicant = frames["annual_applicant_types"]
        if not applicant.empty:
            for policy in POLICIES[:2]:
                data = applicant[applicant.scenario == policy].groupby("year")[["principals", "spouses", "children"]].mean().reset_index()
                stacked(data, ("principals", "spouses", "children"), ("Principals", "Spouses", "Children"), APPLICANT_COLORS, "applicant_type_" + policy.lower(), .65, "Fiscal year", "Visas")
        if "cohort_share_summary" in frames:
            frame = frames["cohort_share_summary"]
            for stem, (category, country, policy) in COHORT_ASSETS.items():
                part = frame[(frame.eb_category == category) & (frame.nationality == country) & (frame.scenario == policy)]
                data = part.pivot(index="entry_year", columns="fate", values="mean").reset_index()
                if data.empty or data[list(FATES)].isna().any().any():
                    raise ValueError(f"Incomplete cohort figure data: {stem}")
                if not data["still_dependent"].eq(0).all():
                    raise ValueError(f"Unresolved children in cohort figure data: {stem}")
                data = data[["entry_year", *RESOLVED_FATES]]
                stacked(data, RESOLVED_FATES, ("Saved", "Aged out", "Exited queue"),
                        OUTCOME_COLORS, stem, .8, "Queue-entry year", "Share of children",
                        percent=True, legend_columns=3)
    return figures


def coverage(source: str, refreshed: set[str], figures: dict) -> dict:
    tables = re.findall(r"\\label\{(tab:[^}]+)\}", source)
    pending = {label for match in TABLE_RE.finditer(source)
               if r"\caption{Pending rerun:" in match.group()
               for label in re.findall(r"\\label\{(tab:[^}]+)\}", match.group())}
    assets = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    return {"tables": [{"label": label, "status": "pending_rerun" if label in pending else
                        "refreshed" if label in refreshed else
                        "retained_historical_sensitivity" if label in HISTORICAL_SENSITIVITY_TABLES else
                        "requires_long_followup" if label == "tab:save_rates" else "retained_input_or_demography"}
                       for label in tables],
            "figure_assets": [{"original_path": asset, "generated_stem": Path(asset).stem.removeprefix("current-"),
                               "status": "refreshed" if Path(asset).stem.removeprefix("current-") in figures else
                               "missing_required_export"}
                              for asset in assets]}


def _clear_prior_generated(output: Path) -> None:
    """Avoid stale complete assets surviving a subsequent explicit partial preview."""
    manifest = output / "refresh-manifest.json"
    if not manifest.exists():
        if any(output.iterdir()):
            raise ValueError("Refresh output is nonempty without a valid manifest; choose a new directory")
        return
    previous = json.loads(manifest.read_text())
    owned = previous.get("generated_artifacts", {})
    paths = []
    for relative, sha in owned.items():
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()) or not path.is_file() or digest(path) != sha:
            raise ValueError(f"Previous generated output was changed; preserving it: {relative}")
        paths.append(path)
    unexpected = {path.resolve() for path in output.rglob("*") if path.is_file()} - {path.resolve() for path in paths} - {manifest.resolve()}
    if unexpected:
        raise ValueError("Refresh output contains unrecorded files; choose a new directory")
    for path in paths:
        path.unlink()
    manifest.unlink()


def build_refresh(manuscript: Path, projection_dir: Path, cohort_dir: Path | None,
                  observed_csv: Path, output: Path, *, expected_runs=None, expected_seeds=None,
                  selection_label: str, allow_partial=False, projection_records=None, cohort_records=None) -> dict:
    manuscript, output, observed_csv = Path(manuscript), Path(output), Path(observed_csv)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(manuscript.resolve()): digest(manuscript), str(observed_csv.resolve()): digest(observed_csv),
              str(Path(__file__).resolve()): digest(Path(__file__))}
    style_source = Path(__file__).with_name("figure_style.py").resolve()
    inputs[str(style_source)] = digest(style_source)
    records, hashes, identity = load_records(Path(projection_dir), 2040, expected_runs=expected_runs, expected_seeds=expected_seeds, supplied_records=projection_records)
    inputs.update(hashes)
    if len(records) < 2 and not allow_partial:
        raise ValueError("Final numerical uncertainty summaries require at least two paired records")
    frames, missing = summarize_projection(records, allow_partial=allow_partial)
    cohort_seeds = []
    if cohort_dir is not None:
        cohorts, hashes, cohort_identity = load_records(Path(cohort_dir), 2061, supplied_records=cohort_records)
        inputs.update(hashes)
        if cohort_identity != identity:
            raise ValueError("Cohort and annual-event records use different scientific sources or assumptions")
        verify_followup_prefix(records, cohorts)
        frames.update(summarize_cohorts(cohorts))
        cohort_seeds = [r["seed"] for r in cohorts]
    elif not allow_partial:
        raise MissingExportError("Current-engine FY2061 cohort records are required; FY2040 outcomes cannot substitute")
    else:
        missing.append("Current-engine FY2061 cohort records")
    frames["historical_fit"], frames["historical_fit_annual"] = historical_fit(frames["historical_cells"], pd.read_csv(observed_csv), len(records))
    source, ledger = manuscript.read_text(), []
    replacements = table_replacements(frames)
    # The added joint-fit table is optional in other source-preserving layouts.
    if r"\label{tab:additional-joint-fit}" not in source:
        replacements.pop("tab:additional-joint-fit")
    revised = source
    for label, rows in replacements.items():
        revised, changes = replace_table_cells(revised, label, rows)
        ledger.extend(changes)
    candidate = output / "refreshed-tables.tex"
    if candidate.resolve() == manuscript.resolve():
        raise ValueError("Refresher must never overwrite the authored manuscript")
    _clear_prior_generated(output)
    candidate.write_text(revised)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures = draw_figures(figure_dir, frames)
    _json(output / "table-patches.json", {"input_sha256": digest(manuscript), "cells": ledger,
                                          "changed_table_labels": list(replacements), "prose_changed": False})
    # JSON null represents undefined one-run MCSEs; never write nonstandard NaN.
    values = {name: json.loads(frame.to_json(orient="records")) for name, frame in frames.items()
              if name.endswith("summary") or name in {"historical_fit", "cohort_save_rates"}}
    values["selection"] = {"label": selection_label, "annual_n": len(records), "annual_seeds": [r["seed"] for r in records],
                           "cohort_n": len(cohort_seeds), "cohort_seeds": cohort_seeds,
                           "event_window": [2025, 2040], "cohort_entry_window": [2009, 2040], "cohort_observation_end": 2061,
                           "table_interval": "conditional 95% Student-t confidence interval for the mean",
                           "interval_formula": "mean +/- t.ppf(0.975, n-1) * sample_sd / sqrt(n)",
                           "empirical_p025_p975": "retained separately; not the printed mean confidence interval"}
    _json(output / "narrative-values.json", values)
    _json(output / "coverage.json", coverage(source, set(replacements), figures))
    for path, sha in inputs.items():
        if digest(Path(path)) != sha:
            raise ValueError(f"Input changed during manuscript refresh: {path}")
    generated = {str(path.relative_to(output)): digest(path) for path in output.rglob("*")
                 if path.is_file() and path.name != "refresh-manifest.json"}
    report = {"schema_version": 1, "status": "partial_preview" if allow_partial or missing or len(records) < 2 else "complete",
              "selection": values["selection"], "scientific_identity": identity, "input_sha256": inputs,
              "generated_artifacts": generated, "figures": figures, "missing_exports": missing,
              "prose_changed": False, "requires_author_integration": [
                  "Use narrative-values.json for Abstract, Results, Discussion and captions; prose was not rewritten.",
                  "State actual annual and long-follow-up run counts and the selected exit specification.",
                  "Label refreshed interval columns as 95% confidence intervals for conditional means, not simulation percentiles.",
                  "Label retained original sensitivity tables as historical or replace them from matching new experiments.",
                  "Refresh the added federal stock/timing diagnostic separately from current historical inventories.",
                  "Copy the generated color PNG figures at the recorded original inclusion widths."]}
    import scipy
    import matplotlib
    report["analysis_software"] = {"python": platform.python_version(), "numpy": np.__version__,
                                   "pandas": pd.__version__, "scipy": scipy.__version__,
                                   "matplotlib": matplotlib.__version__}
    _json(output / "refresh-manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--cohort-records", type=Path)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = build_refresh(args.manuscript, args.records, args.cohort_records, args.observed,
                           args.output, expected_runs=args.expected_runs, selection_label=args.label,
                           allow_partial=args.allow_partial)
    print(json.dumps({"status": report["status"], "selection": report["selection"],
                      "missing_exports": report["missing_exports"]}, indent=2))


if __name__ == "__main__":
    main()
