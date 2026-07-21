"""
Command-line entry point and orchestration for the immigration microsimulation.

Configures and runs paired capped versus uncapped scenarios from the CLI, then
assembles comparative outputs: backlog snapshots, age-out and exit statistics,
visa allocation and spillover patterns, and cohort-level child outcomes. Also
coordinates export of all time-series and summary datasets to CSV and triggers
generation of standard visualizations for downstream analysis.
"""


import argparse
import csv
import logging
import sys
import time
import textwrap
from pathlib import Path


from .backlog_analyzer import (
    create_backlog_analysis,
    save_backlog_analysis,
    save_states_to_csv,
)
from .empirical_params import COUNTRIES
from .models import EBCategory
from .sim import Simulation
from .states import SimulationConfig
from .ci_runner import run_ci_analysis, print_ci_summary, save_ci_results_to_csv
from .visa_consumption_exporter import VisaConsumptionExporter
from .visualization import SimulationVisualizer


def run_comparative_analysis(config: SimulationConfig) -> None:
    """
    Run capped vs uncapped comparison with full exports.

    Both scenarios use historical reconstruction (2009-2024 always capped),
    then diverge in projection period.

    Workflow:
        1. Run uncapped simulation (no per-country caps in projection)
        2. Run capped simulation (7% per-country caps in projection)
        3. Generate backlog snapshots
        4. Print comprehensive comparison summary
        5. Generate all visualizations
        6. Export all CSV datasets

    Args:
        config: Base simulation configuration (used to derive both scenarios)
    """
    print("\n" + "#" * 100)
    print("Comparative Analysis: Capped vs Uncapped")
    print("#" * 100)

    start_time = time.perf_counter()

    ################################
    # STEP 1: Run uncapped scenario
    ################################
    print("\n[1/6] Running UNCAPPED scenario...")
    config_uncapped = SimulationConfig(
        years=config.years,
        seed=config.seed,
        output_path=config.output_path,
        country_cap_enabled=False,
        debug=config.debug,
        start_year=config.start_year,
    )
    sim_uncapped = Simulation(config_uncapped)
    states_uncapped = sim_uncapped.run()
    print(f"Uncapped: {states_uncapped[0].year}-{states_uncapped[-1].year}")

    ##############################
    # STEP 2: Run capped scenario
    ##############################
    print("\n[2/6] Running CAPPED scenario...")
    config_capped = SimulationConfig(
        years=config.years,
        seed=config.seed,
        output_path=config.output_path,
        country_cap_enabled=True,
        debug=config.debug,
        start_year=config.start_year,
    )
    sim_capped = Simulation(config_capped)
    states_capped = sim_capped.run()
    print(f"Capped: {states_capped[0].year}-{states_capped[-1].year}")

    # Create output directory for all exports
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    ####################################
    # STEP 3: Generate backlog analysis
    ####################################
    print("\n[3/6] Generating backlog analysis...")
    backlog_uncapped = create_backlog_analysis(sim_uncapped, "Uncapped")
    backlog_capped = create_backlog_analysis(sim_capped, "Capped")

    # Export final backlog snapshots (for cross-scenario comparison)
    save_backlog_analysis([backlog_uncapped], str(output_dir / "backlog_uncapped.csv"))
    save_backlog_analysis([backlog_capped], str(output_dir / "backlog_capped.csv"))
    print(f"Saved to {output_dir}/backlog_*.csv")

    ############################################
    # STEP 4: Print detailed comparison summary
    ############################################
    print("\n[4/6] Generating comparison summary...")
    print_comprehensive_comparison_summary(
        states_uncapped,
        states_capped,
        backlog_uncapped,
        backlog_capped,
        sim_uncapped,
        sim_capped,
    )

    ##################################
    # STEP 5: Generate visualizations
    ##################################
    print("\n[5/6] Generating visualizations...")
    try:
        visualizer = SimulationVisualizer(output_dir=str(output_dir))

        # EB category conversion charts
        try:
            visualizer.generate_eb_conversion_charts(states_uncapped, states_capped)
            print("EB conversion charts done")
        except Exception as e:
            print(f"EB conversion charts skipped ({e})")

        # Children aged-out analysis
        try:
            visualizer.generate_age_out_charts(states_uncapped, states_capped)
            print("Age-out charts done")
        except Exception as e:
            print(f"Age-out charts skipped ({e})")

        # Applicant type conversion charts
        try:
            visualizer.generate_applicant_type_charts(states_uncapped, states_capped)
            print("Applicant type charts done")
        except Exception as e:
            print(f"Applicant type charts skipped ({e})")

    except Exception as e:
        print(f"Visualization error: {e}")

    ##################################
    # STEP 6: Export all CSV datasets
    ##################################
    print("\n[6/6] Exporting CSV files...")

    # Export yearly time-series states (comprehensive year-by-year data)
    try:
        save_states_to_csv(states_uncapped, str(output_dir / "states_uncapped.csv"))
        save_states_to_csv(states_capped, str(output_dir / "states_capped.csv"))
        print("Wrote states_uncapped.csv")
        print("Wrote states_capped.csv")
    except Exception as e:
        print(f"Yearly states skipped ({e})")

    # Export visa consumption by nationality
    try:
        exporter = VisaConsumptionExporter(output_dir=output_dir)
        exporter.export_visa_consumption(states_uncapped, states_capped)
        print("Wrote visa_consumption.csv")
    except Exception as e:
        print(f"Visa consumption skipped ({e})")

    # Export children aged-out segmentation
    try:
        export_children_aged_out_segmentation_csv(states_uncapped, states_capped, output_dir)
        print("Wrote children_aged_out_segmentation.csv")
    except Exception as e:
        print(f"Children aged-out segmentation skipped ({e})")

    # Export exited data (queue exits)
    try:
        export_exited_data_csv(states_uncapped, states_capped, output_dir)
        print("Wrote exited_data.csv")
    except Exception as e:
        print(f"Exited data skipped ({e})")

    # Export pass 2 visa allocation (spillover)
    try:
        export_pass2_visa_allocation_csv(states_uncapped, states_capped, output_dir)
        print("Wrote pass2_visa_allocations.csv")
    except Exception as e:
        print(f"Pass 2 allocation skipped ({e})")

    # Export age-out percentages
    try:
        export_ageout_percentage_csv(states_uncapped, states_capped, output_dir)
        print("Wrote ageout_percentage_by_nationality.csv")
    except Exception as e:
        print(f"Age-out percentage skipped ({e})")

    # Export child outcomes by entry-year cohort
    try:
        export_outcomes_by_entry_year_csv(sim_uncapped, sim_capped, output_dir)
        print("Wrote outcomes_by_entry_year.csv")
    except Exception as e:
        print(f"Outcomes by entry year skipped ({e})")

    elapsed = time.perf_counter() - start_time
    print(f"\n{'#'*100}")
    print(f"Total runtime: {elapsed:.1f}s")
    print(f"Output directory: {output_dir}/")
    print(f"{'#'*100}\n")


def export_children_aged_out_segmentation_csv(states_uncapped, states_capped, output_dir: Path) -> str:
    """
    Export children aged-out by year, nationality, and EB category.

    Creates a CSV with yearly breakdown of children who aged out, segmented by:
        - Nationality (India, China, etc.)
        - EB category (EB-1, EB-2, EB-3)

    Args:
        states_uncapped: List of SimulationState for uncapped scenario
        states_capped: List of SimulationState for capped scenario
        output_dir: Directory to write CSV file

    Returns:
        Path to exported CSV file
    """
    csv_path = output_dir / "children_aged_out_segmentation.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Scenario", "Nationality", "EB_Category", "Count"])

        # Process both scenarios
        for scenario_name, states in [
            ("Uncapped", states_uncapped),
            ("Capped", states_capped),
        ]:
            for state in states:
                # Count by nationality
                nat_counts = {}
                for child in state.children_aged_out_this_year_list:
                    nat_counts[child.nationality] = nat_counts.get(child.nationality, 0) + 1

                for nat, count in nat_counts.items():
                    writer.writerow([state.year, scenario_name, nat, "Overall", count])

                # Count by EB category
                eb_counts = {}
                for child in state.children_aged_out_this_year_list:
                    if child.parent_eb_category:
                        cat = child.parent_eb_category.value
                        eb_counts[cat] = eb_counts.get(cat, 0) + 1

                for cat, count in eb_counts.items():
                    writer.writerow([state.year, scenario_name, "Overall", cat, count])

    return str(csv_path)


def export_exited_data_csv(states_uncapped, states_capped, output_dir: Path) -> str:
    """
    Export queue exits by nationality and EB category.

    Tracks people who exit the green card queue each year.

    Args:
        states_uncapped: List of SimulationState for uncapped scenario
        states_capped: List of SimulationState for capped scenario
        output_dir: Directory to write CSV file

    Returns:
        Path to exported CSV file
    """
    csv_path = output_dir / "exited_data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Scenario", "Nationality", "EB_Category", "Count"])

        for scenario_name, states in [
            ("Uncapped", states_uncapped),
            ("Capped", states_capped),
        ]:
            for state in states:
                # Aggregate by nationality
                for nat, count in state.exited_by_nationality.items():
                    writer.writerow([state.year, scenario_name, nat, "Overall", count])

                # Aggregate by EB category
                for cat, count in state.exited_by_eb_category.items():
                    writer.writerow([state.year, scenario_name, "Overall", cat.value, count])

    return str(csv_path)


def export_pass2_visa_allocation_csv(states_uncapped, states_capped, output_dir: Path) -> str:
    """
    Export pass 2 visa allocations (spillover to oversubscribed countries) by EB category.

    Args:
        states_uncapped: List of SimulationState for uncapped scenario
        states_capped: List of SimulationState for capped scenario
        output_dir: Directory to write CSV file

    Returns:
        Path to exported CSV file
    """
    csv_path = output_dir / "pass2_visa_allocations.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Year", "Scenario", "EB_Category", "Pass2_Allocations"])

        for scenario_name, states in [
            ("Uncapped", states_uncapped),
            ("Capped", states_capped),
        ]:
            for state in states:
                for (
                    eb_cat,
                    allocations,
                ) in state.visa_allocations_pass2_annual_by_eb_category.items():
                    writer.writerow([state.year, scenario_name, eb_cat.value, allocations])

    return str(csv_path)


def export_ageout_percentage_csv(states_uncapped, states_capped, output_dir: Path) -> str:
    """
    Export final age-out percentages by nationality.

    Calculates what percentage of children of a particular nationality created throughout the simulation
    eventually aged out before their parent received a green card.

    Args:
        states_uncapped: List of SimulationState for uncapped scenario
        states_capped: List of SimulationState for capped scenario
        output_dir: Directory to write CSV file

    Returns:
        Path to exported CSV file
    """
    csv_path = output_dir / "ageout_percentage_by_nationality.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(
            f,
            lineterminator="\n",
        )
        writer.writerow(
            [
                "Scenario",
                "Nationality",
                "Total_Children_Created",
                "Total_Children_Aged_Out",
                "Ageout_Percentage",
            ]
        )

        # Get final states (cumulative totals)
        final_uncapped = states_uncapped[-1]
        final_capped = states_capped[-1]

        # Process both scenarios
        for scenario_name, final_state in [
            ("Uncapped", final_uncapped),
            ("Capped", final_capped),
        ]:
            for nationality in COUNTRIES:
                total_created = final_state.children_created_total_by_nationality.get(nationality, 0)
                ageout_pct = final_state.children_ageout_percentage_by_nationality.get(nationality, 0.0)

                # Calculate total aged out from percentage
                total_aged_out = int(total_created * ageout_pct / 100) if total_created > 0 else 0

                writer.writerow(
                    [
                        scenario_name,
                        nationality,
                        total_created,
                        total_aged_out,
                        f"{ageout_pct:.2f}",
                    ]
                )

    return str(csv_path)


def export_outcomes_by_entry_year_csv(
    sim_uncapped: Simulation,
    sim_capped: Simulation,
    output_dir: Path,
) -> str:
    """
    Export child outcomes (aged_out, saved, exited) by entry-year cohort,
    nationality, and EB category.

    Each row represents one (scenario, entry_year, nationality, eb_category) cohort.
    Percentages are relative to total children created in that cohort, so
    AgeOut_Pct + Save_Pct + Exit_Pct will be < 100% for recent cohorts
    where some children are still dependent at simulation end.

    Args:
        sim_uncapped: Completed uncapped Simulation object
        sim_capped:   Completed capped Simulation object
        output_dir:   Directory to write CSV file

    Returns:
        Path to exported CSV file
    """
    csv_path = output_dir / "outcomes_by_entry_year.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(
            f,
            lineterminator="\n",
        )
        writer.writerow(
            [
                "Scenario",
                "Entry_Year",
                "Nationality",
                "EB_Category",
                "Total_Created",
                "Aged_Out",
                "Saved",
                "Exited",
                "Still_Dependent",
                "AgeOut_Pct",
                "Save_Pct",
                "Exit_Pct",
            ]
        )

        for scenario_name, sim in [
            ("Uncapped", sim_uncapped),
            ("Capped", sim_capped),
        ]:
            outcomes = sim.child_processor.get_outcomes_by_entry_year_nationality_eb()

            for (entry_year, nationality, eb_category), counts in sorted(outcomes.items()):
                total = counts["total_created"]
                if total == 0:
                    continue

                aged_out = counts["aged_out"]
                saved = counts["saved"]
                exited = counts["exited"]
                still_dependent = total - aged_out - saved - exited

                writer.writerow(
                    [
                        scenario_name,
                        entry_year,
                        nationality,
                        eb_category,
                        total,
                        aged_out,
                        saved,
                        exited,
                        still_dependent,
                        f"{aged_out / total * 100:.2f}",
                        f"{saved    / total * 100:.2f}",
                        f"{exited   / total * 100:.2f}",
                    ]
                )

    return str(csv_path)


def print_comprehensive_comparison_summary(
    states_uncapped,
    states_capped,
    backlog_uncapped,
    backlog_capped,
    sim_uncapped,
    sim_capped,
):
    """
    Print detailed comparison of key metrics between scenarios.

    Displays side-by-side comparison of:
        1. Queue backlog (principals waiting)
        2. Children aged out (lost derivative status)
        3. Backlog by EB category
        4. Family-adjusted backlog (principals + dependents)

    Args:
        states_uncapped: List of SimulationState for uncapped scenario
        states_capped: List of SimulationState for capped scenario
        backlog_uncapped: BacklogAnalysis for uncapped scenario
        backlog_capped: BacklogAnalysis for capped scenario
        sim_uncapped: Simulation object for uncapped scenario
        sim_capped: Simulation object for capped scenario
    """
    print("\n" + "#" * 100)
    print("Comprehensive Comparison:")
    print("#" * 100)

    # Extract final year states (cumulative totals)
    final_unc = states_uncapped[-1]
    final_cap = states_capped[-1]

    ########################################################
    # 1. Queue Backlog (principals waiting for green cards)
    ########################################################
    backlog_unc = sum(final_unc.queue_backlog_by_country.values())
    backlog_cap = sum(final_cap.queue_backlog_by_country.values())

    print("\n1. Queue Backlog (principals waiting for green cards)")
    print("-" * 100)
    print(f"Uncapped: {backlog_unc:,}")
    print(f"Capped:   {backlog_cap:,}")
    print(f"Increase: {backlog_cap - backlog_unc:+,} ({(backlog_cap/max(backlog_unc, 1) - 1)*100:.1f}%)")

    # Breakdown by nationality
    print("\nBy nationality:")
    for nat in COUNTRIES:
        unc_nat = final_unc.queue_backlog_by_country.get(nat, 0)
        cap_nat = final_cap.queue_backlog_by_country.get(nat, 0)
        print(f"{nat}: Uncapped={unc_nat:,}, Capped={cap_nat:,}, Diff={cap_nat - unc_nat:+,}")

    #######################
    # 2. Children Aged Out
    #######################
    print("\n2. Children Aged Out")
    print("-" * 100)
    print(f"Uncapped: {final_unc.cumulative_children_aged_out:,}")
    print(f"Capped:   {final_cap.cumulative_children_aged_out:,}")
    additional = final_cap.cumulative_children_aged_out - final_unc.cumulative_children_aged_out
    print(f"Additional due to caps: {additional:+,}")

    if final_unc.cumulative_children_aged_out > 0:
        pct = (additional / final_unc.cumulative_children_aged_out) * 100
        print(f"Percentage increase: {pct:.1f}%")

    # Breakdown by nationality
    print("\nBy nationality:")
    for nat in COUNTRIES:
        unc_aged = final_unc.aged_out_by_nationality.get(nat, 0)
        cap_aged = final_cap.aged_out_by_nationality.get(nat, 0)
        print(f"{nat}: Uncapped={unc_aged:,}, Capped={cap_aged:,}, Diff={cap_aged - unc_aged:+,}")

    #########################################
    # 3. Backlog by EB Category (principals)
    #########################################
    print("\n3. Backlog by EB Category (principals)")
    print("-" * 100)
    for cat in [EBCategory.EB1, EBCategory.EB2, EBCategory.EB3]:
        unc_cat = backlog_uncapped.backlog_by_eb_category.get(cat, 0)
        cap_cat = backlog_capped.backlog_by_eb_category.get(cat, 0)
        print(f"{cat.value}: Uncapped={unc_cat:,}, Capped={cap_cat:,}, Diff={cap_cat - unc_cat:+,}")

    ###############################################################
    # 4. Family Adjusted Backlog (principals + spouses + children)
    ###############################################################
    print("\n4. Family Adjusted Backlog (principals + spouses + children)")
    print("-" * 100)
    print(f"Uncapped:")
    print(f"  Principals only: {backlog_uncapped.total_backlog:,}")
    print(f"  With families:   {backlog_uncapped.total_family_adjusted_backlog:,.0f}")
    print(f"Capped:")
    print(f"  Principals only: {backlog_capped.total_backlog:,}")
    print(f"  With families:   {backlog_capped.total_family_adjusted_backlog:,.0f}")
    print(f"Additional people due to caps:")
    print(f"  Principals:  {backlog_capped.total_backlog - backlog_uncapped.total_backlog:+,}")
    print(
        f"  With families: {backlog_capped.total_family_adjusted_backlog - backlog_uncapped.total_family_adjusted_backlog:+,.0f}"
    )

    print("#" * 100)


def run_ci_pipeline(config: SimulationConfig, n_runs: int) -> None:
    """
    Run Monte Carlo CI analysis and export all CI artifacts.

    Executes N paired (uncapped, capped) simulation runs, aggregates 95%
    empirical confidence intervals (2.5th-97.5th percentile across runs),
    prints a console summary, and persists the raw per-run data
    (ci_raw_scalars.csv, ci_raw_timeseries.csv, ci_raw_cohorts.csv) plus the
    aggregated summaries (ci_scalars.csv, ci_timeseries.csv) under outputs/ci/.

    Kept separate from run_comparative_analysis so the deterministic run and
    the Monte Carlo run can be invoked independently or together.

    Args:
        config: Base SimulationConfig. country_cap_enabled is ignored
                (overridden per scenario inside run_ci_analysis).
        n_runs: Number of paired Monte Carlo iterations.

    CI artifacts always land in outputs/ci/ (anchored to the project root),
    kept separate from the standard run's --output for clean archival.
    """
    print("\n" + "=" * 80)
    print("CI PIPELINE: Monte Carlo Confidence Intervals")
    print("=" * 80)

    start_time = time.perf_counter()

    ci_output_dir = Path(__file__).resolve().parent.parent / "outputs" / "ci"
    ci_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run N paired simulations and aggregate empirical CIs
    ci = run_ci_analysis(config, n_runs=n_runs)

    # Step 2: Print human-readable console summary
    print_ci_summary(ci)

    # Step 3: Persist aggregated CI CSVs for reproducible downstream analysis
    print("\n[CI] Exporting CI CSVs...")
    try:
        ci_csv_files = save_ci_results_to_csv(ci, ci_output_dir)
        for label, path in ci_csv_files.items():
            print(f"  {label}  ->  {path}")
    except Exception as e:
        print(f"  CI CSV export failed: {e}")

    elapsed = time.perf_counter() - start_time
    print(f"\n{'=' * 80}")
    print(f"CI pipeline complete - {n_runs} runs in {elapsed:.1f}s")
    print(f"CI outputs: {ci_output_dir}/")
    print(f"{'=' * 80}\n")


def main():
    """
    Main CLI entry point for immigration simulation.

    Parses command-line arguments and runs comparative analysis between
    capped and uncapped scenarios. Exports comprehensive datasets and
    visualizations to output directory.
    """
    parser = argparse.ArgumentParser(
        description="Immigration simulation: child age-out and backlog analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              # Run historical reconstruction only (2009-2024)
              python -m simulation --reconstruction

              # Run 2009 + 30 years (2009-2039)
              python -m simulation --years 30

              # With reproducible seed
              python -m simulation --years 30 --seed 12345

              # Enable debug logging
              python -m simulation --years 30 --debug
        """
        ).strip(),
    )

    parser.add_argument(
        "--reconstruction",
        action="store_true",
        help="Run historical reconstruction only (2009-2024)",
    )
    parser.add_argument(
        "--years",
        type=int,
        help="Number of years to simulate from 2009 (e.g., 2 years = 2009-2010)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-year INFO and WARNING logging (faster; recommended for full runs)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="After the standard run, run N paired Monte Carlo iterations and "
        "export 95 percent empirical confidence intervals to outputs/ci/",
    )
    parser.add_argument(
        "--ci-runs",
        type=int,
        default=50,
        help="Number of paired Monte Carlo iterations for --ci "
        "(default: 50, matching the published intervals; must be >= 2)",
    )

    args = parser.parse_args()

    if args.quiet and not args.debug:
        logging.disable(logging.WARNING)

    ##############################
    # Determine simulation period
    ##############################
    start_year = 2009

    if args.reconstruction:
        years = 16  # 2009-2024 inclusive
    elif args.years:
        years = args.years
    else:
        print("No simulation period given. Use --reconstruction or --years N")
        sys.exit(1)

    end_year = start_year + years - 1

    ##################################
    # Create simulation configuration
    ##################################
    seed = args.seed if args.seed is not None else 2014
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        years=years,
        seed=seed,
        output_path=str(output_dir),
        country_cap_enabled=False,
        debug=args.debug,
        start_year=start_year,
    )

    ##############################
    # Print simulation parameters
    ##############################
    print("\n" + "#" * 100)
    print("Immigration Simulation")
    print("#" * 100)
    print(f"Period:  {start_year}-{end_year} ({years} years)")
    print(f"Seed:    {config.seed}")
    print(f"Output:  {output_dir}/")
    print(f"Countries: {', '.join(COUNTRIES)}")
    if args.debug:
        print("debug logging enabled")
    print("#" * 100)

    ############################################
    # Standard single-seed comparative analysis
    ############################################
    try:
        run_comparative_analysis(config)
    except Exception as e:
        print(f"\nComparative analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Optional Stage 2: Monte Carlo CI analysis (runs only when --ci is passed)
    if args.ci:
        if args.ci_runs < 2:
            print(f"--ci-runs must be >= 2 to compute CI bounds; got {args.ci_runs}.")
            sys.exit(1)
        try:
            run_ci_pipeline(config, n_runs=args.ci_runs)
        except Exception as e:
            print(f"\nCI pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
