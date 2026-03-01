"""
Backlog analysis utilities for the immigration microsimulation.

Takes completed simulation runs and constructs snapshots of the green
card queue, including both principal-only backlogs and family-adjusted totals
that count spouses and dependent children. Provides helpers to aggregate these
backlogs by nationality and EB category, compare capped versus uncapped
scenarios, and export year-by-year states and final backlog summaries to CSV
for external analysis and visualization.
"""


import logging
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from .models import EBCategory
from .states import SimulationState

logger = logging.getLogger(__name__)


@dataclass
class BacklogAnalysis:
    """
    Comprehensive snapshot of green card queue backlogs at end of simulation.

    Provides two perspectives on the backlog:
    1. Principal counts - just the primary applicants (standard government metric)
    2. Family-adjusted counts - includes spouses and children (actual people waiting)

    Example: If 100,000 Indian principals are in queue, and each has on average
    a spouse and 1.5 children, the family-adjusted backlog is ~350,000 people.

    All counts are broken down by:
    - Nationality (India, China, ROW)
    - EB category (EB-1 through EB-5)
    - Category-nationality pairs (EB-2 India, EB-2 China, etc.)
    """

    scenario_name: str  # "Capped" or "Uncapped"
    total_backlog: int  # Total principal applicants waiting
    backlog_by_country: Dict[str, int]  # Principals by nationality
    backlog_by_eb_category: Dict[EBCategory, int] = field(default_factory=dict)
    backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(default_factory=dict)

    # Family-adjusted counts (principals + spouses + dependent children)
    family_adjusted_backlog: Dict[str, int] = field(default_factory=dict)
    total_family_adjusted_backlog: int = 0
    family_adjusted_backlog_by_eb_category: Dict[EBCategory, int] = field(default_factory=dict)
    family_adjusted_backlog_by_category_nationality: Dict[Tuple[EBCategory, str], int] = field(
        default_factory=dict
    )

    def to_dataframe(self):
        """
        Export backlog data to pandas DataFrame for CSV writing or further analysis.

        Creates a long-format table with one row per (scenario, category, nationality)
        combination.

        Returns:
            pandas DataFrame with columns: scenario, category, nationality,
            backlog_size, family_adjusted_backlog
        """
        rows = []

        # Grand total across all categories and countries
        rows.append(
            {
                "scenario": self.scenario_name,
                "category": "TOTAL",
                "nationality": "ALL",
                "backlog_size": self.total_backlog,
                "family_adjusted_backlog": self.total_family_adjusted_backlog,
            }
        )

        # Country subtotals (sum across all EB categories)
        for country, backlog in self.backlog_by_country.items():
            family_backlog = self.family_adjusted_backlog[country]
            rows.append(
                {
                    "scenario": self.scenario_name,
                    "category": "ALL_CATEGORIES",
                    "nationality": country,
                    "backlog_size": backlog,
                    "family_adjusted_backlog": family_backlog,
                }
            )

        # Category subtotals (sum across all countries)
        for category, backlog in self.backlog_by_eb_category.items():
            family_backlog = self.family_adjusted_backlog_by_eb_category[category]
            rows.append(
                {
                    "scenario": self.scenario_name,
                    "category": category.value,
                    "nationality": "ALL",
                    "backlog_size": backlog,
                    "family_adjusted_backlog": family_backlog,
                }
            )

        # Granular breakdown by both category AND nationality
        # Shows e.g., EB-2 India vs EB-2 China vs EB-2 ROW separately
        for (
            category,
            nationality,
        ), backlog in self.backlog_by_category_nationality.items():
            family_backlog = self.family_adjusted_backlog_by_category_nationality[(category, nationality)]
            rows.append(
                {
                    "scenario": self.scenario_name,
                    "category": category.value,
                    "nationality": nationality,
                    "backlog_size": backlog,
                    "family_adjusted_backlog": family_backlog,
                }
            )

        return pd.DataFrame(rows)


def create_backlog_analysis(sim: "Simulation", scenario_name: str) -> BacklogAnalysis:
    """
    Analyze final backlog state from a completed simulation.

    Calculates both principal counts (just the applicants) and family-adjusted counts
    (applicants + their spouses + dependent children under 21).

    Args:
        sim: Completed Simulation object with all state history
        scenario_name: "Capped" or "Uncapped"

    Returns:
        BacklogAnalysis with principal counts and family-adjusted totals
    """
    final_state = sim.states[-1]
    total_backlog = sum(final_state.queue_backlog_by_country.values())

    # Count how many children each parent has
    # Only includes children who are (1) under 21 and (2) have parents still in queue
    # Child processor already filters out aged-out kids and those whose parents converted
    children_by_parent = defaultdict(int)
    for child in sim.child_processor.dependent_children:
        children_by_parent[child.parent_worker_id] += 1

    # Calculate family-adjusted backlogs in a single pass through the queues
    # We accumulate counts across three dimensions simultaneously: nationality,
    # EB category, and (nationality, category) pairs
    family_adjusted = defaultdict(int)
    family_adjusted_by_eb = {}
    family_adjusted_by_cat_nat = {}

    # Initialize all EB categories to 0 to prevent KeyError when categories have no workers
    for category in EBCategory:
        family_adjusted_by_eb[category] = 0

    # Iterate through all 15 queues (5 EB categories x 3 nationalities)
    for (
        category,
        nationality,
    ), queue in sim.category_nationality_queues.items():
        cat_nat_total = 0

        for worker_id in queue:
            worker = sim.worker_lookup[worker_id]

            # Count this worker's actual family size
            # 1 (principal) + actual spouse count (0 or 1) + actual children count
            family_size = 1 + worker.spouse_count + children_by_parent.get(worker_id, 0)

            # Accumulate across all three dimensions in one pass
            family_adjusted[nationality] += family_size
            family_adjusted_by_eb[category] += family_size
            cat_nat_total += family_size

        # Store the (category, nationality) pair total
        family_adjusted_by_cat_nat[(category, nationality)] = cat_nat_total

    return BacklogAnalysis(
        scenario_name=scenario_name,
        total_backlog=total_backlog,
        backlog_by_country=final_state.queue_backlog_by_country.copy(),
        backlog_by_eb_category=final_state.queue_backlog_by_eb_category.copy(),
        backlog_by_category_nationality=final_state.queue_backlog_by_eb_category_nationality.copy(),
        family_adjusted_backlog=dict(family_adjusted),
        total_family_adjusted_backlog=sum(family_adjusted.values()),
        family_adjusted_backlog_by_eb_category=dict(family_adjusted_by_eb),
        family_adjusted_backlog_by_category_nationality=family_adjusted_by_cat_nat,
    )


#########################################
# Export and Comparison Helper Functions
#########################################


def save_states_to_csv(states: List[SimulationState], output_path: str) -> None:
    """
    Export year-by-year simulation results to CSV for external analysis.

    Creates a time-series dataset with one row per simulation year. Each row includes
    workforce demographics, conversion statistics, child age-out counts, and per-nationality
    backlog sizes.

    Args:
        states: List of SimulationState objects (each representing a year)
        output_path: Where to write the CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as csvfile:
        # Core columns present in every simulation
        fieldnames = [
            "year",
            "total_workers",
            "permanent_workers",
            "temporary_workers",
            "new_permanent",
            "new_temporary",
            "converted_temps",
            "cumulative_conversions",
            "temporary_share",
            "children_aged_out_this_year",
            "cumulative_children_aged_out",
            "children_at_risk",
            "queue_backlog_total",
            "country_cap_enabled",
            "annual_conversion_cap",
        ]

        # Dynamically discover nationalities from the data
        nationalities = set()
        for state in states:
            nationalities.update(state.queue_backlog_by_country.keys())

        # Add one column per nationality for backlog tracking
        for nat in sorted(nationalities):
            fieldnames.append(f"backlog_{nat}")

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write one row per simulation year
        for state in states:
            row = {
                "year": state.year,
                "total_workers": state.total_workers,
                "permanent_workers": state.permanent_workers,
                "temporary_workers": state.temporary_workers,
                "new_permanent": state.new_permanent,
                "new_temporary": state.new_temporary,
                "converted_temps": state.converted_temps,
                "cumulative_conversions": state.cumulative_conversions,
                "temporary_share": round(state.temporary_share, 4),
                "children_aged_out_this_year": state.children_aged_out_this_year,
                "cumulative_children_aged_out": state.cumulative_children_aged_out,
                "children_at_risk": state.children_at_risk,
                "queue_backlog_total": sum(state.queue_backlog_by_country.values()),
                "country_cap_enabled": state.country_cap_enabled,
                "annual_conversion_cap": state.annual_conversion_cap,
            }

            # Add per-nationality backlog columns (principal counts only)
            for nat in sorted(nationalities):
                principal_backlog = state.queue_backlog_by_country.get(nat, 0)
                row[f"backlog_{nat}"] = principal_backlog

            writer.writerow(row)

    logger.info(f"Saved simulation results to {output_file}")


def save_backlog_analysis(analyses: List[BacklogAnalysis], output_path: str) -> None:
    """
    Export final backlog analysis to CSV for scenario comparison.

    Unlike save_states_to_csv (which exports time-series), this exports a final
    snapshot comparing different scenarios side-by-side.

    Includes both principal counts and family-adjusted totals broken down by
    EB category and nationality.

    Args:
        analyses: List of BacklogAnalysis objects for comparison
        output_path: Where to write the CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as csvfile:
        fieldnames = [
            "scenario",
            "category",
            "nationality",
            "backlog_size",
            "family_adjusted_backlog",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Each BacklogAnalysis converts itself to DataFrame, then we write rows
        for analysis in analyses:
            df = analysis.to_dataframe()
            for _, row in df.iterrows():
                writer.writerow(row.to_dict())

    logger.info(f"Saved backlog analysis to {output_file}")
