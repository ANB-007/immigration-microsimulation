"""
Empirical parameterization for the immigration microsimulation.

Centralizes statutory caps, spillover rules, nationality distributions, and
queue-exit processes, and provides helper functions to compute visa pools,
per-country limits, and family composition inputs used throughout the model.
"""


from typing import Dict
import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Import EBCategory for type safety
from .models import EBCategory

# Import petition data
from .petition_data import (
    I140_ANNUAL_APPROVALS,
    COUNTRY_APPROVALS,
    EB4_ANNUAL_TOTALS_BY_YEAR,
    SIJS_NATIONALITY_DISTRIBUTION,
    OTHER_EB4_NATIONALITY_DISTRIBUTION_BY_YEAR,
    EB5_HISTORICAL,
    EB5_NATIONALITY_DISTRIBUTION_BY_YEAR,
)

###############################
# Simulation period parameters
###############################

# Last year of historical data we have from USCIS
# Years <= 2024: Use actual historical data
# Years > 2024: Use 2024 values for projection
RECONSTRUCTION_END_YEAR = 2024

#############################
# Core visa pool parameters
#############################

# Base employment-based visa cap set by Congress
# This is the statutory limit for EB green cards per year
GREEN_CARD_CAP_ABS = 140_000

# Family-sponsored spillover visas (unused from family-based pool)
# When family-based categories don't use all their visas, the remainder
# "spills over" to employment-based categories
FAMILY_SPILLOVER_ASSUMPTION = 27_394

# Total annual visa pool = base + spillover
# This is what's actually available for EB applicants each year
TOTAL_EB_VISA_POOL = GREEN_CARD_CAP_ABS + FAMILY_SPILLOVER_ASSUMPTION

# Historical EB visa pool by fiscal year
HISTORICAL_EB_VISA_POOL = {
    2009: 140987,
    2010: 150262,
    2011: 139302,
    2012: 144647,
    2013: 161269,
    2014: 151359,
    2015: 143952,
    2016: 140350,
    2017: 139604,
    2018: 139483,
    2019: 140586,
    2020: 147153,
    2021: 195507,  # COVID-era spike due to recapture provisions
    2022: 275250,  # Peak due to unused family-based visas
    2023: 193928,
    2024: 167394,  # Use this value for projections beyond 2024
    -1: 200000,
}

#########################
# Per-country cap
#########################

# 7% per-country cap within each EB category
PER_COUNTRY_CAP_SHARE = 0.07

###############################
# EB category statutory shares
###############################

# Base visa allocation by EB category (before cascade spillover)
EB_CATEGORY_STATUTORY_SHARES = {
    EBCategory.EB1: 0.286,  # 28.6%
    EBCategory.EB2: 0.286,  # 28.6%
    EBCategory.EB3: 0.286,  # 28.6%
    EBCategory.EB4: 0.071,  # 7.1%
    EBCategory.EB5: 0.071,  # 7.1%
}

########################
# Cascade configuration
########################

# How unused visas flow between EB categories
# Example: If EB-4 only uses 5,000 visas but has 11,865 allocated,
# the remaining ~6,865 visas flow to EB-1
CASCADE_FLOWS = {
    # EB-4 and EB-5 are "faucet" categories - they send unused visas upward
    EBCategory.EB4: {
        "receives_from": [],
        "sends_to": [EBCategory.EB1],
    },
    EBCategory.EB5: {
        "receives_from": [],
        "sends_to": [EBCategory.EB1],
    },
    # EB-1 receives from EB-4/5, then sends its own unused visas to EB-2
    EBCategory.EB1: {
        "receives_from": [EBCategory.EB4, EBCategory.EB5],
        "sends_to": [EBCategory.EB2],
    },
    # EB-2 receives from EB-1, then sends unused visas to EB-3
    EBCategory.EB2: {
        "receives_from": [EBCategory.EB1],
        "sends_to": [EBCategory.EB3],
    },
    # EB-3 is the "sink" - receives from EB-2 but unused visas are lost
    EBCategory.EB3: {
        "receives_from": [EBCategory.EB2],
        "sends_to": [],
    },
}

# Processing order ensures all possible spillover is captured
# Process EB-4/5 first (sources), then EB-1 (receives and passes on), etc.
CASCADE_PROCESSING_ORDER = [
    EBCategory.EB4,
    EBCategory.EB5,
    EBCategory.EB1,
    EBCategory.EB2,
    EBCategory.EB3,
]

#######################
# Nationality tracking
#######################

# Three nationality groups tracked in the simulation
COUNTRIES = [
    "India",
    "China",
    "ROW",
]

##########################
# Child age-out threshold
##########################

# Child ages out at 21 years old (loses dependent status)
# Children who turn 21 while parent is still in queue must either:
# 1. Return to home country
# 2. Obtain their own visa (H-1B, F-1, etc.)
# 3. Become undocumented if they stay
CHILD_AGEOUT_AGE = 21

################################
# Spouse presence distributions
################################

# Load empirical spouse presence distributions from JSON file
_SPOUSE_DIST_PATH = Path(__file__).parent / "distributions" / "spouse_presence_empirical_distributions.json"
with open(_SPOUSE_DIST_PATH, "r") as f:
    SPOUSE_PRESENCE_EMPIRICAL = json.load(f)


def get_spouse_probability(nationality: str, pathway: str = None) -> float:
    """
    Get probability that a worker has a spouse based on nationality and pathway.

    Special case: SIJS pathway (unaccompanied minors) always returns 0.0
    """
    # SIJS override
    if pathway == "SIJS":
        return 0.0

    base_p = SPOUSE_PRESENCE_EMPIRICAL[pathway][nationality]["probabilities"][1]

    return base_p


###############################
# Children count distributions
###############################

# Load empirical children count distribution from JSON file
_CHILDREN_COUNT_MARRIED_DIST_PATH = (
    Path(__file__).parent / "distributions" / "children_count_empirical_distributions_married.json"
)
with open(_CHILDREN_COUNT_MARRIED_DIST_PATH, "r") as f:
    CHILDREN_COUNT_EMPIRICAL_MARRIED = json.load(f)


def sample_children_count(
    nationality: str,
    pathway: str = None,
    spouse_count: int = 0,
    rng: np.random.Generator = None,
) -> int:
    """
    Sample number of children from empirical distributions conditional on marital status.

    Args:
        nationality: Country of birth (e.g., "China", "India", "ROW")
        pathway: Immigration pathway (e.g., "EB-1", "EB-2", "SIJS")
        spouse_count: Binary indicator (0=no spouse, 1=has spouse)
        rng: Random number generator

    Returns:
        Integer count of children (0-5)
    """

    # SIJS override
    if pathway == "SIJS":
        return 0

    # Deterministic gate: unmarried principals have no foreign-born dependent children
    if spouse_count == 0:
        return 0

    dist = CHILDREN_COUNT_EMPIRICAL_MARRIED[pathway][nationality]

    counts = np.array(dist["counts"])
    probs = np.array(dist["probabilities"])

    return int(rng.choice(counts, p=probs))


################################
# Child entry-age distributions
################################

# Load empirical child entry-age distributions from JSON file
_CHILD_ENTRY_AGE_DIST_PATH = (
    Path(__file__).parent / "distributions" / "child_entry_age_empirical_distributions.json"
)
with open(_CHILD_ENTRY_AGE_DIST_PATH, "r") as f:
    CHILD_ENTRY_AGE_EMPIRICAL = json.load(f)


def sample_child_entry_age(nationality: str, pathway: str = None, rng: np.random.Generator = None) -> int:
    """
    Sample child entry age from empirical distributions.

    Args:
        nationality: Country of birth (e.g., "China", "India", "ROW")
        pathway: Immigration pathway (e.g., "EB-1", "EB-2") -- required for lookup
        rng: Random number generator (required)

    Returns:
        Integer entry age (0-20)
    """

    dist = CHILD_ENTRY_AGE_EMPIRICAL[pathway][nationality]
    ages = np.array(dist["ages"])
    probs = np.array(dist["probabilities"])

    return int(rng.choice(ages, p=probs))


#########################
# Queue exit parameters
#########################

# Base emigration rates by tenure
QUEUE_EXIT_RATE_PROJECTION_BY_TENURE = {
    "recently_arrived": 0.050,  # 5.0% annual rate for 0-4 years in US
    "intermediate": 0.036,  # 3.6% annual rate for 5-9 years in US
    "settled": 0.020,  # 2.0% annual rate for 10+ years in US
}

# Default rate for fallback
DEFAULT_QUEUE_EXIT_RATE_PROJECTION = 0.029

# Nationality-specific emigration multipliers
NATIONALITY_EMIGRATION_MULTIPLIERS = {
    "India": 0.7,  # 30% lower than baseline
    "China": 1.0,  # Baseline rate
    "ROW": 1.0,  # Baseline rate
}

# EB category-specific emigration multipliers
CATEGORY_EMIGRATION_MULTIPLIERS = {
    EBCategory.EB1: 0.25,  # 75% lower
    EBCategory.EB2: 0.5,  # 50% lower
    EBCategory.EB3: 0.75,  # 25% lower
    EBCategory.EB4: 1.0,  # Baseline
    EBCategory.EB5: 0.125,  # 87.5% lower
}


###################
# Helper functions
###################


def get_total_eb_visa_pool(year: int) -> int:
    """
    Get total EB visa pool for a given year.

    For 2009-2024: uses actual historical USCIS data
    For 2025+: uses 2024 value (167,394) for projections

    Args:
        year: Fiscal year

    Returns:
        Total EB visas available for that year
    """
    return HISTORICAL_EB_VISA_POOL.get(year, HISTORICAL_EB_VISA_POOL[2024])


def get_pathway_totals(year: int) -> Dict[str, int]:
    """
    Get petition approval totals for all pathways in a specific year.

    Historical period (2009-2024): Heterogeneous petition approval counts
    Projection period (2025+): Uses 2024 values as steady-state baseline

    Args:
        year: Fiscal year

    Returns:
        Dictionary mapping pathway name to total petition approvals

    Raises:
        ValueError: If year is before 2009 (no data available)
    """
    if year < 2009:
        raise ValueError(f"No data available for year {year} (earliest: 2009)")

    # Use 2024 values for projection years
    lookup_year = min(year, RECONSTRUCTION_END_YEAR)

    return {
        "EB-1": I140_ANNUAL_APPROVALS["EB-1"][lookup_year],
        "EB-2": I140_ANNUAL_APPROVALS["EB-2"][lookup_year],
        "EB-3": I140_ANNUAL_APPROVALS["EB-3"][lookup_year],
        "SIJS": EB4_ANNUAL_TOTALS_BY_YEAR[lookup_year]["SIJS"],
        "Other_EB4": EB4_ANNUAL_TOTALS_BY_YEAR[lookup_year]["Other_EB4"],
        "EB-5": EB5_HISTORICAL[lookup_year],
    }


def get_nationality_distribution_for_pathway(pathway: str, year: int) -> Dict[str, float]:
    """
    Get nationality distribution for a pathway in a specific year.

    Args:
        pathway: Pathway name (EB-1, EB-2, EB-3, SIJS, Other_EB4, EB-5)
        year: Fiscal year

    Returns:
        Dictionary of {nationality: proportion} summing to 1.0
    """
    # Use 2024 values for projection years
    lookup_year = min(year, RECONSTRUCTION_END_YEAR)

    # SIJS uses pooled distribution (same for all years)
    if pathway == "SIJS":
        return SIJS_NATIONALITY_DISTRIBUTION

    # Other_EB4 uses year-specific R-1 data
    if pathway == "Other_EB4":
        return OTHER_EB4_NATIONALITY_DISTRIBUTION_BY_YEAR[lookup_year]

    # EB-5 uses year-specific visa issuance data
    if pathway == "EB-5":
        return EB5_NATIONALITY_DISTRIBUTION_BY_YEAR[lookup_year]

    # EB-1/2/3: Calculate from I-140 approval counts
    if pathway in ["EB-1", "EB-2", "EB-3"]:
        total_approvals = I140_ANNUAL_APPROVALS[pathway][lookup_year]
        india_count = COUNTRY_APPROVALS["India"][pathway][lookup_year]
        china_count = COUNTRY_APPROVALS["China"][pathway][lookup_year]
        row_count = total_approvals - india_count - china_count

        return {
            "India": round(india_count / total_approvals, 4),
            "China": round(china_count / total_approvals, 4),
            "ROW": round(row_count / total_approvals, 4),
        }

    raise ValueError(f"Unknown pathway: {pathway}")


def get_queue_exit_rate(
    nationality: str,
    eb_category: EBCategory,
    current_year: int,
    years_in_us: int,
    age: int,
) -> float:
    """
    Calculate annual queue exit probability for an applicant.

    Formula:
        tenure_base_rate x nationality_multiplier x category_multiplier
        x age_adjustments (logistic curve for ages 45+)

    Args:
        nationality: Country of birth
        eb_category: EB category (EB1-EB-5)
        current_year: Unused (retained for future updates; changes based on political situation, etc.)
        years_in_us: Years since priority date
        age: Current age of principal applicant

    Returns:
        Annual queue exit probability (0.0-1.0)
    """
    # Tenure-based baseline
    if years_in_us < 5:
        base_rate = QUEUE_EXIT_RATE_PROJECTION_BY_TENURE["recently_arrived"]
    elif years_in_us < 10:
        base_rate = QUEUE_EXIT_RATE_PROJECTION_BY_TENURE["intermediate"]
    else:
        base_rate = QUEUE_EXIT_RATE_PROJECTION_BY_TENURE["settled"]

    # Apply nationality-specific multiplier
    nationality_multiplier = NATIONALITY_EMIGRATION_MULTIPLIERS.get(nationality, 1.0)

    # Apply category-specific multiplier
    category_multiplier = CATEGORY_EMIGRATION_MULTIPLIERS.get(eb_category, 1.0)

    # Calculate base exit rate
    exit_rate = base_rate * nationality_multiplier * category_multiplier

    # Age-based adjustment: logistic curve
    if age >= 45:
        L = 1.0  # Certain exit assumption
        k = 0.5  # Steepness
        midpoint = 55  # Inflection at age

        # Calculate age-based exit probability
        age_exit_prob = L / (1 + np.exp(-k * (age - midpoint)))

        # Take max of base rate and age-based probability
        exit_rate = max(exit_rate, age_exit_prob)

    return exit_rate


def calculate_annual_eb_caps(
    annual_cap: int = TOTAL_EB_VISA_POOL,
) -> Dict[EBCategory, int]:
    """
    Calculate annual EB category visa allocation based on statutory shares.

    These are BASE allocations before cascade spillover.

    Args:
        annual_cap: Total annual employment-based visa allocation. Defaults to TOTAL_EB_VISA_POOL (167,394 visas/year).

    Returns:
        Dictionary mapping EBCategory enum to base annual visa allocation.
    """
    eb_caps = {}
    for category, share in EB_CATEGORY_STATUTORY_SHARES.items():
        eb_caps[category] = max(1, round(annual_cap * share))

    return eb_caps


def calculate_per_country_caps_by_category(
    annual_eb_caps: Dict[EBCategory, int],
) -> Dict[EBCategory, Dict[str, int]]:
    """
    Calculate per-country visa caps within each EB category.

    Args:
        annual_eb_caps: Base EB category allocations (before cascade)

    Returns:
        Nested dictionary: {EBCategory: {nationality: visa_cap}}
    """
    per_country_caps = {}

    for category, total_cap in annual_eb_caps.items():
        # Each country gets 7% of category total
        country_cap = max(1, round(total_cap * PER_COUNTRY_CAP_SHARE))

        # ROW gets remaining allocation
        row_cap = max(1, total_cap - (2 * country_cap))

        per_country_caps[category] = {
            "India": country_cap,
            "China": country_cap,
            "ROW": row_cap,
        }

    return per_country_caps


#######################
# Validation functions
#######################


def validate_nationality_distributions():
    """
    Ensure all nationality distributions sum to 1.0.
    """
    # Validate SIJS
    total = sum(SIJS_NATIONALITY_DISTRIBUTION.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"SIJS nationality distribution sums to {total}, not 1.0")

    # Validate Other_EB4 for all years
    for year, nat_dist in OTHER_EB4_NATIONALITY_DISTRIBUTION_BY_YEAR.items():
        total = sum(nat_dist.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Other_EB4 year {year}: nationality distribution sums to {total}, not 1.0")

    # Validate EB-5 for all years
    for year, nat_dist in EB5_NATIONALITY_DISTRIBUTION_BY_YEAR.items():
        total = sum(nat_dist.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"EB-5 year {year}: nationality distribution sums to {total}, not 1.0")


def validate_spouse_probabilities():
    """
    Ensure spouse presence probabilities are within valid range (0.0-1.0).
    """
    for pathway, nat_dists in SPOUSE_PRESENCE_EMPIRICAL.items():
        for nationality, dist in nat_dists.items():
            # Validate probabilities sum to 1.0
            total = sum(dist["probabilities"])
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Pathway {pathway}, Nationality {nationality}: spouse probabilities sum to {total}, not 1.0"
                )

            # Validate each probability is in valid range
            for prob in dist["probabilities"]:
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(
                        f"Pathway {pathway}, Nationality {nationality}: probability {prob} must be between 0.0 and 1.0"
                    )


def validate_queue_exit_parameters():
    """
    Ensure queue exit parameters are valid.
    """
    # Validate tenure rates
    for tenure, rate in QUEUE_EXIT_RATE_PROJECTION_BY_TENURE.items():
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"Invalid tenure rate for {tenure}: {rate} (must be 0.0 between 1.0)")

    # Validate category multipliers
    for category, mult in CATEGORY_EMIGRATION_MULTIPLIERS.items():
        if mult <= 0:
            raise ValueError(f"Category emigration multiplier for {category} must be > 0")


def validate_pathway_totals():
    """
    Ensure pathway totals are positive.
    """
    for year in range(2009, RECONSTRUCTION_END_YEAR + 1):
        totals = get_pathway_totals(year)
        for pathway, total in totals.items():
            if total <= 0:
                raise ValueError(f"Year {year}, pathway {pathway}: total={total} must be > 0")


def validate_eb_shares():
    """
    Ensure EB category shares sum to 1.0.
    """
    total_share = sum(EB_CATEGORY_STATUTORY_SHARES.values())
    if abs(total_share - 1.0) > 0.01:
        raise ValueError(f"EB category shares sum to {total_share}, not 1.0")


#######################################
# Module initialization and validation
#######################################

# Run all validations at module load
validate_eb_shares()
validate_nationality_distributions()
validate_spouse_probabilities()
validate_queue_exit_parameters()
validate_pathway_totals()
