"""Rebuild the manuscript from saved paired runs, or execute fresh replications.

    python reproduce.py                 # figures, tables and publication PDFs
    python reproduce.py --analysis-only # figures and tables, no TeX tools
    python reproduce.py --run --workers 3 # new runs, then the same analysis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
SCRATCH = ROOT / ".local/build"
RESULTS_POINTER = ROOT / "data/current-results.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_records(pointer=RESULTS_POINTER):
    """Select the published campaign explicitly, without falling back to old results."""
    pointer = Path(pointer)
    if not pointer.exists():
        return ROOT / "data/runs"
    selection = json.loads(pointer.read_text())
    records = (ROOT / selection["records"]).resolve()
    if not records.is_relative_to(ROOT):
        raise ValueError("Current result records must stay within the repository")
    if digest(records / "manifest.json") != selection["manifest_sha256"]:
        raise ValueError("Current result selection is missing or changed; specify verified --records")
    return records


def response_values(text, values):
    """Resolve annotated response numbers from the same saved-run summaries."""
    pattern = re.compile(r"(<!-- ijm-value:([^\s]+) -->).*?(<!-- /ijm-value -->)", re.S)

    def replace(match):
        before, key, after = match.groups()
        if key not in values:
            raise ValueError(f"Response refers to missing result: {key}")
        return before + str(values[key]) + after

    return pattern.sub(replace, text)


def response_locations(text, auxiliary):
    """Resolve response references by semantic manuscript labels after typesetting."""
    labels = {}
    pattern = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{(\d+)\}")
    for key, number, page in pattern.findall(Path(auxiliary).read_text()):
        if key.startswith("tab:"):
            kind = "Table"
        elif key.startswith("fig:"):
            kind = "Figure"
        elif key.startswith(("sec", "subsec", "subsubsec", "app:")):
            kind = "Section"
        else:
            continue
        labels[key] = (number, page, kind)
    reference = re.compile(r"\b(Table|Figure|Section|Appendix) ([A-Z]?\.?\d+(?:\.\d+)*), p\. (\d+)(\s*)<!-- ijm-ref:([^\s]+) -->")
    def replace(match):
        kind, _, _, spacing, key = match.groups()
        if key not in labels:
            raise ValueError(f"Response refers to missing manuscript label: {key}")
        number, page, actual = labels[key]
        if kind != actual and not (kind == "Appendix" and actual == "Section"):
            raise ValueError(f"Response reference has wrong type: {key}")
        return f"{kind} {number}, p. {page}{spacing}<!-- ijm-ref:{key} -->"
    result = reference.sub(replace, text)
    # Existing references should all carry semantic anchors; do not silently
    # retain a stale page number if an author adds an unanchored reference.
    stripped = reference.sub("", result)
    if re.search(r"\b(?:Table|Figure|Section|Appendix) [A-Z]?\.?\d+(?:\.\d+)*, p\. \d+", stripped):
        raise ValueError("A response reference needs an ijm-ref semantic label")
    return result


def compile_document(directory, name):
    command = ["tectonic", "-X", "compile", "--only-cached", "--keep-logs",
               "--keep-intermediates", name + ".tex", "--outdir", "."]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
    (SCRATCH / f"{name}-compile.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Typesetting failed; see {SCRATCH / (name + '-compile.log')}")
    log = (directory / f"{name}.log").read_text()
    problems = re.findall(r"^.*(?:Overfull|Missing character|undefined references|Citation .*undefined).*$", log, re.M)
    if problems:
        raise RuntimeError(f"Unresolved typesetting problems: {problems}")
    shutil.copyfile(directory / f"{name}.pdf", PAPER / f"{name}.pdf")


def publish_tables(source, destination, *, figure_prefix=""):
    """Keep public results to tabular data and the figures printed in the paper."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.csv"):
        shutil.copyfile(path, destination / path.name)
    figure_source = source / "figures" if (source / "figures").is_dir() else source
    for path in figure_source.glob("*.png"):
        shutil.copyfile(path, destination / (figure_prefix + path.name))
    # These are dedicated result-asset folders, never manuscript or input
    # document folders. Remove the superseded vector figure exports on rebuild.
    for path in destination.glob("*.pdf"):
        path.unlink()


def validate_figure_layout(source):
    """Prevent regeneration from restoring floating figures or top captions."""
    source = re.sub(r"(?<!\\)%[^\n]*", "", source)
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
    if any(Path(asset).suffix.lower() != ".png" for asset in graphics):
        raise ValueError("Manuscript figures must use explicit PNG assets")
    for figure in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", source, re.S):
        body = figure.group(1)
        if not body.lstrip().startswith("[H]"):
            raise ValueError("Manuscript figures must stay at their references with [H] placement")
        images = list(re.finditer(r"\\includegraphics\b", body))
        captions = list(re.finditer(r"\\caption\b", body))
        if not images or not captions or captions[0].start() < images[-1].start():
            raise ValueError("Every figure caption must appear below all its images")


def integrate_table_cells(manuscript, candidate, ledger_path):
    """Publish only the audited result cells, preserving all authored prose."""
    from simulation.analysis.manuscript_refresh import replace_table_cells

    manuscript, candidate = Path(manuscript), Path(candidate)
    ledger = json.loads(Path(ledger_path).read_text())
    if digest(manuscript) != ledger["input_sha256"]:
        raise ValueError("Authored manuscript changed after the table refresh")
    replacements = {}
    for cell in ledger["cells"]:
        replacements.setdefault(cell["label"], {}).setdefault(tuple(cell["row"]), {})[cell["column"]] = cell["after"]
    revised = manuscript.read_text()
    for label, rows in replacements.items():
        revised, _ = replace_table_cells(revised, label, rows)
    if revised != candidate.read_text():
        raise ValueError("Refreshed manuscript contains changes outside audited table cells")
    manuscript.write_text(revised)
    return len(ledger["cells"])


def build(records_path, *, compile_pdfs=True, update_manuscript_tables=False):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH / "matplotlib"))
    from simulation.campaign import load_campaign
    from simulation.analysis.manuscript_refresh import build_refresh
    from simulation.analysis.narrative_values import generate as narrative_values
    from simulation.analysis.publication_evidence import generate as publication_evidence

    SCRATCH.mkdir(parents=True, exist_ok=True)
    if compile_pdfs:
        missing = [name for name in ("tectonic", "pandoc") if shutil.which(name) is None]
        if missing:
            raise RuntimeError("Missing PDF tools: " + ", ".join(missing))
    before = {p.name: digest(p) for p in PAPER.glob("*.tex")}
    validate_figure_layout((PAPER / "revised-paper.tex").read_text())
    data = load_campaign(records_path)
    plan, records, missing, hashes = data
    if missing:
        raise ValueError("Complete all declared pairs before rebuilding the manuscript")
    primary = next(case for case in plan["cases"] if case["role"] == "primary")
    cohorts = next(case for case in plan["cases"] if case["role"] == "cohorts")
    main, sensitivity = SCRATCH / "main", SCRATCH / "sensitivity"
    observed = ROOT / "data/validation/dos_visa_consumption_reconciled.csv"
    if digest(observed) != plan["observed_sha256"]:
        raise ValueError("Historical comparison data differ from the declared run inputs")
    for directory in (main, sensitivity):
        if records_path == directory or records_path.is_relative_to(directory):
            raise ValueError("Saved records must be outside the temporary analysis folders")
        if directory.exists():
            shutil.rmtree(directory)
    print(f"Analyzing {plan['planned_pairs']} saved capped/uncapped pairs...", flush=True)
    settings = primary["overrides"]
    selection_label = (
        "Catchall retention in " + ", ".join(settings["catchall_categories"])
        + f", midpoint {settings['catchall_median_age']} and "
        + f"{100 * settings['catchall_tail_probability']:g}% age-only retention at {settings['catchall_near_zero_age']}")
    refresh = build_refresh(
        PAPER / "revised-paper.tex", records_path, records_path, observed, main,
        expected_runs=primary["pairs"], expected_seeds=primary["seeds"],
        selection_label=selection_label,
        projection_records=records[primary["name"]], cohort_records=records[cohorts["name"]],
    )
    narrative_values(main / "narrative-values.json", main, PAPER / "revised-paper.tex")
    evidence = publication_evidence(records_path, ROOT / "data/validation", sensitivity, campaign_data=data)
    if refresh["status"] != "complete" or evidence["status"] != "complete":
        raise ValueError("Incomplete records cannot supply manuscript results")
    publish_tables(main, ROOT / "outputs/main", figure_prefix="current-")
    publish_tables(sensitivity, ROOT / "outputs/sensitivity")
    generated = PAPER / "generated"
    generated.mkdir(exist_ok=True)
    shutil.copyfile(main / "manuscript-narrative-values.tex", generated / "manuscript-narrative-values.tex")
    for name in ("publication-values.tex", "publication-evidence.tex"):
        text = (sensitivity / name).read_text().replace("{figures/", "{")
        validate_figure_layout(text)
        (generated / name).write_text(text)

    typesetting = SCRATCH / "typesetting"
    typesetting.mkdir(exist_ok=True)
    for name in ("thebib.tex", "response-to-reviewer.tex"):
        shutil.copyfile(PAPER / name, typesetting / name)
    shutil.copytree(generated, typesetting / "generated", dirs_exist_ok=True)
    # Typesetting uses the refreshed cells. Optional source integration below
    # verifies that the candidate contains only these audited numeric changes.
    source = (main / "refreshed-tables.tex").read_text()
    graphics = ("\\graphicspath{{" + (ROOT / "outputs/main").as_posix() + "/}{"
                + (ROOT / "outputs/sensitivity").as_posix() + "/}}")
    source = re.sub(r"\\graphicspath\{(?:\{[^}]*\})+\}", lambda _: graphics, source)
    validate_figure_layout(source)
    (typesetting / "revised-paper.tex").write_text(source)
    if compile_pdfs:
        print("Compiling manuscript and response...", flush=True)
        compile_document(typesetting, "revised-paper")
        response = response_values((PAPER / "point-by-point-reviewer-response.md").read_text(), evidence["macros"])
        response = response_locations(response, typesetting / "revised-paper.aux")
        (PAPER / "point-by-point-reviewer-response.md").write_text(response)
        (typesetting / "point-by-point-reviewer-response.md").write_text(response)
        subprocess.run(["pandoc", "--from", "markdown", "--to", "latex", "--top-level-division=section",
                        "point-by-point-reviewer-response.md", "-o", "response-body.tex"], cwd=typesetting, check=True)
        compile_document(typesetting, "response-to-reviewer")
    for name, expected in before.items():
        if digest(PAPER / name) != expected:
            raise ValueError(f"Authored manuscript source changed during build: {name}")
    integrated_cells = 0
    if update_manuscript_tables:
        integrated_cells = integrate_table_cells(PAPER / "revised-paper.tex", main / "refreshed-tables.tex",
                                                main / "table-patches.json")
    report = {"status": "complete" if compile_pdfs else "analysis_complete", "paired_runs": plan["planned_pairs"],
              "publication_ready": r"\textbf{Working draft.}" not in source and "Pending rerun:" not in source,
              "selected_records": str(records_path),
              "pending_sensitivity_tables": re.findall(
                  r"\\caption\{Pending rerun:[\s\S]*?\\label\{([^}]+)\}", source),
              "case_pairs": evidence["per_case_completed"], "all_accounting_checks_passed": evidence["audits"]["all_passed"],
              "primary_difference_mcse": evidence["primary_difference_mcse"], "inputs": hashes,
              "authored_tex_unchanged": not update_manuscript_tables,
              "authored_table_cells_integrated": integrated_cells}
    (SCRATCH / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Done. Tables and figures: outputs/. " + ("Manuscript and response: paper/." if compile_pdfs else "PDFs were not rebuilt."), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-only", action="store_true", help="Regenerate tables and figures without compiling PDFs")
    parser.add_argument("--update-manuscript-tables", action="store_true",
                        help="Integrate audited result table cells into the authored manuscript after a successful build")
    parser.add_argument("--run", action="store_true", help="Execute new replications before building; saved runs are the default")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--records", type=Path, help="Directory containing saved paired records (default: current published campaign)")
    parser.add_argument("--run-output", type=Path, default=ROOT / ".local/runs", help="Separate destination for newly simulated records")
    args = parser.parse_args()
    records = args.records.resolve() if args.records else (args.run_output.resolve() if args.run else default_records())
    if args.run:
        from simulation.campaign import run_campaign
        records = args.run_output.resolve()
        run_campaign(records, workers=args.workers)
    build(records, compile_pdfs=not args.analysis_only, update_manuscript_tables=args.update_manuscript_tables)


if __name__ == "__main__":
    main()
