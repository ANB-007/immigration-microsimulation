"""
Configuration and annual state containers for the immigration microsimulation.

Defines the simulation configuration object and the year-by-year SimulationState
snapshots that record worker stocks and flows, child age-outs, visa usage,
backlogs, exits, and Pass 2 allocations by EB category and nationality. These
structures provide a stable interface between the core engine and all
downstream analysis and exports.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import EBCategory, AgedOutChild


@dataclass
class SimulationConfig:
    """
    Configuration for simulation runs.

    HYBRID APPROACH:
    The simulation models two distinct periods with different policy regimes:

    - Historical period (2009-2024): Always enforces per-country caps to match
      real-world outcomes. This ensures the simulation accurately reflects the
      backlog buildup that actually occurred.

    - Projection period (2025+): Policy enforcement depends on scenario:
      -- Uncapped scenario: country_cap_enabled=False (removes per-country limit)
      -- Capped scenario: country_cap_enabled=True (keeps current policy)

    This approach lets us validate against historical data while exploring
    counterfactual policy reforms.
    """

    years: int
    seed: int
    output_path: str
    country_cap_enabled: bool
    debug: bool
    start_year: int


@dataclass
class SimulationState:
    """
    Complete snapshot of the simulation at a single point in time.

    Each year produces one SimulationState capturing everything that happened:
    how many workers converted, how many children aged out,
    how many visas were consumed, who exited, etc.
    """

    # Basic worker counts
    year: int
    total_workers: int
    permanent_workers: int
    temporary_workers: int
    exited_workers: int
    new_permanent: int
    new_temporary: int
    converted_temps: int
    cumulative_conversions: int
    temporary_share: float

    # Child age-out data
    children_aged_out_this_year: int
    cumulative_children_aged_out: int
    children_at_risk: int
    aged_out_by_nationality: Dict[str, int]
    children_aged_out_this_year_list: List[AgedOutChild]

    # EB category tracking
    converted_by_eb_category: Dict[EBCategory, int]
    queue_backlog_by_eb_category: Dict[EBCategory, int]
    queue_backlog_by_eb_category_nationality: Dict[Tuple[EBCategory, str], int]
    aged_out_by_eb_category: Dict[EBCategory, int]
    converted_by_eb_category_total: Dict[EBCategory, int]

    # Nationality tracking
    converted_by_country: Dict[str, int]
    queue_backlog_by_country: Dict[str, int]
    country_cap_enabled: bool
    annual_conversion_cap: int

    # Visa consumption tracking
    visas_consumed_this_year: int
    cumulative_visas_consumed: int
    visas_available_this_year: int
    visas_consumed_by_category_nationality: Dict[Tuple[EBCategory, str], int]

    # Applicant type tracking
    converted_spouses: int
    children_saved_this_year: int
    children_aged_out_by_nationality: Dict[str, int]

    # Exited tracking
    exited_this_year: int
    cumulative_exited: int
    exited_by_nationality: Dict[str, int]
    exited_by_eb_category: Dict[EBCategory, int]
    exited_by_category_nationality: Dict[Tuple[EBCategory, str], int]

    # Pass 2 visa allocations tracking
    visa_allocations_pass2_annual_by_eb_category: Dict[EBCategory, int]

    # Child age-out percentage tracking
    children_created_total_by_nationality: Dict[str, int]
    children_ageout_percentage_by_nationality: Dict[str, float]
