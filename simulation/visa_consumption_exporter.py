"""
CSV export utilities for visa consumption in the immigration microsimulation.

Provides a helper class that aggregates visas used by principals, spouses,
and children by year, scenario (capped vs uncapped), EB category, and
nationality, and writes the resulting counts to a CSV file for
downstream analysis and visualization.
"""

import logging
from pathlib import Path
import csv

from .models import EBCategory
from .empirical_params import COUNTRIES

logger = logging.getLogger(__name__)


class VisaConsumptionExporter:
    """
    Exports visa consumption statistics to CSV files.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the exporter with an output directory.

        Args:
            output_dir: Directory path where CSV files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_visa_consumption(
        self,
        states_uncapped,
        states_capped,
    ) -> str:
        """
        Export visa consumption data to CSV for both scenarios.

        CSV structure:
        - year: Simulation year
        - scenario: "Uncapped" or "Capped"
        - category: "Overall" or specific EB category (EB-1, EB-2, etc.)
        - nationality: India, China, or ROW
        - visas_consumed: Total visas (principals + spouses + children)

        Args:
            states_uncapped: Simulation states from uncapped scenario
            states_capped: Simulation states from capped scenario

        Returns:
            Path to saved CSV file
        """
        eb_categories = [
            EBCategory.EB1,
            EBCategory.EB2,
            EBCategory.EB3,
            EBCategory.EB4,
            EBCategory.EB5,
        ]
        csv_rows = []

        # Process both scenarios in single loop
        for scenario_name, states in [
            ("Uncapped", states_uncapped),
            ("Capped", states_capped),
        ]:
            for state in states:
                # Overall consumption (aggregated across all EB categories)
                for nationality in COUNTRIES:
                    total_visas = sum(
                        state.visas_consumed_by_category_nationality.get((cat, nationality), 0)
                        for cat in eb_categories
                    )
                    csv_rows.append(
                        {
                            "year": state.year,
                            "scenario": scenario_name,
                            "category": "Overall",
                            "nationality": nationality,
                            "visas_consumed": total_visas,
                        }
                    )

                # Category-specific consumption
                for eb_cat in eb_categories:
                    for nationality in COUNTRIES:
                        visas = state.visas_consumed_by_category_nationality.get((eb_cat, nationality), 0)
                        csv_rows.append(
                            {
                                "year": state.year,
                                "scenario": scenario_name,
                                "category": eb_cat.value,
                                "nationality": nationality,
                                "visas_consumed": visas,
                            }
                        )

        # Write CSV
        csv_file = self.output_dir / "visa_consumption.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "year",
                    "scenario",
                    "category",
                    "nationality",
                    "visas_consumed",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(f"Visa consumption CSV saved: {csv_file} ({len(csv_rows)} rows)")
        return str(csv_file)
