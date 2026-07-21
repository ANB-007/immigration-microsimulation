"""
Visa conversion engine for the immigration microsimulation.

Implements the full green card allocation logic, including historical demand
constraints, category-level caps, per-country limits, and cascade spillovers
across EB-1 through EB-5. Provides capped and uncapped processing modes, a
two-pass allocation routine for enforcing 7% per-country ceilings and then
reassigning overflow, and utilities to compute family-level visa costs and
shuffle applicants within priority-date cohorts to avoid artificial
nationality clustering.
"""

import logging
from typing import List, Dict, Tuple, Set, Deque, TYPE_CHECKING
from collections import defaultdict, deque
import numpy as np

from .models import Worker, EBCategory
from .empirical_params import (
    CASCADE_PROCESSING_ORDER,
    CASCADE_FLOWS,
    RECONSTRUCTION_END_YEAR,
)

if TYPE_CHECKING:
    from .child_processor import ChildProcessor

logger = logging.getLogger(__name__)

# Historical demand constraints reflect years when certain EB categories or
# nationalities had an abnormal number of allocated visas.

# EB-1 hit its demand ceiling in FY2022 despite 83,211 available visas (including
# spillover), using only 54,137. The remaining 29,074 visas automatically "fell
# down" to EB-2, giving EB-2 an adjusted limit of 109,585 visas.
# Source: (https://wolfsdorf.com/employment-based-immigrant-visa-trends-and-immigrant-visa-usage-for-2022-fiscal-year/)
EB1_HISTORICAL_DEMAND = {
    2022: 54137,
}

# EB-4 experienced suppressed demand during 2011-2014 due to non-minister
# religious worker program sunset provisions. The program expired in 2009 and
# again in 2012, creating multi-year uncertainty that discouraged applications.
# Source: (https://www.usccb.org/resources/non-minister-special-immigrant-religious-worker-visa-program)
EB4_HISTORICAL_DEMAND = {
    2011: 6381,
    2012: 7478,
    2013: 6446,
    2014: 8289,
}

# EB-5 investor visas saw sharp demand drops during the COVID-19 pandemic and
# the Regional Center Program suspension. In total, 25,880 EB-5 visas went unused
# between fiscal years 2018 and 2022.
# Source: (https://greencardbyinvestment.com/article/eb5-2024-dos-issues-all-unreserved-visas)
EB5_HISTORICAL_DEMAND = {
    2020: 3596,
    2021: 2949,
    2022: 10885,
}

# India EB-1 historical demand constraints for COVID-era volatility
EB1_INDIA_HISTORICAL_DEMAND = {
    2020: 17014,
    2021: 30825,
    2022: 21437,
}

# China EB-1 historical demand constraints for COVID-era volatility
EB1_CHINA_HISTORICAL_DEMAND = {
    2020: 4766,
    2021: 9118,
    2022: 11425,
}

# EB-1 ROW historical demand constraints
EB1_ROW_HISTORICAL_DEMAND = {
    2016: 26027,
    2017: 22409,
    2018: 21972,
    2019: 25800,
}

# DOS data shows Indian-born applicants received 54.2% of EB-2 numbers available in FY2022,
# far exceeding the normal 7% per-country limit. This occurred because ROW exhausted its
# demand early in the fiscal year while India maintained deep backlogs. Under current law,
# once ROW demand is satisfied, remaining "otherwise unused" visa numbers are
# allocated to oversubscribed countries without the per-country cap.
# Source: (https://wolfsdorf.com/employment-based-immigrant-visa-trends-and-immigrant-visa-usage-for-2022-fiscal-year/)
EB2_ROW_HISTORICAL_DEMAND = {
    2021: 25404,
    2022: 41355,
}

# India EB-2 historical demand constraints for COVID-era volatility
EB2_INDIA_HISTORICAL_DEMAND = {
    2020: 2599,
    2021: 28246,
    2022: 59432,
}

# China EB-2 historical demand constraints for COVID-era volatility
EB2_CHINA_HISTORICAL_DEMAND = {
    2020: 3118,
    2021: 6091,
    2022: 8523,
}

# The 2009 backlog uses backward extrapolation from NFAP 2011 estimates, which likely
# overstated historical queue depth. This constraint adjusts early allocations to match
# actual USCIS visa issuance records rather than the inflated backlog-derived estimates.
EB3_INDIA_HISTORICAL_DEMAND = {
    2009: 2304,
    # 2021: 14954,
    # 2022: 12600,
}

# Same reason as India EB-3 constraint:
# early backlog estimates likely overstate demand.
EB3_CHINA_HISTORICAL_DEMAND = {
    2009: 1077,
    # 2021: 5113,
    # 2022: 6452,
}

# Rest of World (ROW) EB-3 demand constraints in 2021-2022. Similar to EB-2,
# ROW EB-3 exhausted its eligible applicants early in the fiscal year while
# India maintained deep backlogs.
EB3_ROW_HISTORICAL_DEMAND = {
    2021: 35173,
    2022: 61946,
}


class VisaProcessor:
    """
    Unified visa processor for EB green card conversions.

    This determines which workers get green cards each year based on
    complex statutory rules including:
    - Category-based quotas (28.6% for EB-1/2/3, 7.1% for EB-4/5)
    - Per-country caps (7% of category allocation per country)
    - Cascade spillover (unused visas flow through EB chain)
    - Two-pass allocation (cap enforcement, then overflow processing)

    The hybrid approach handles two distinct periods:
    - Reconstruction (2009-2024): Always enforces caps to match reality
    - Projection (2025+): Caps enforcement varies depending on scenario
    """

    def __init__(
        self,
        country_cap_enabled: bool = False,
        seed_seq: np.random.SeedSequence = None,
    ):
        """
        Initialize visa processor with scenario-specific configuration.

        Args:
            country_cap_enabled: Whether to enforce 7% per-country caps in projection period
            seed_seq: Random seed for reproducible within-cohort shuffling
        """
        self.country_cap_enabled = country_cap_enabled
        if seed_seq is None:
            seed_seq = np.random.SeedSequence(2014)
        self.rng = np.random.default_rng(seed_seq)
        logger.info(
            f"VisaProcessor initialized: Projection period = {'CAPPED' if country_cap_enabled else 'UNCAPPED'}"
        )

    def _build_visa_cost_cache(
        self,
        worker_lookup: Dict[int, Worker],
        child_processor: "ChildProcessor",
    ) -> Dict[int, int]:
        """
        Pre-compute visa costs for all workers to avoid repeated lookups.

        Each green card approval consumes visas for the entire family unit:
        - 1 visa for the principal applicant
        - 1 visa per spouse
        - 1 visa per dependent child under 21

        Args:
            worker_lookup: All workers in the system
            child_processor: Processor tracking dependent children

        Returns:
            Dictionary mapping worker_id -> total visa cost for family
        """
        # Build a lookup table counting children per parent
        children_by_parent = defaultdict(int)
        for child in child_processor.dependent_children:
            children_by_parent[child.parent_worker_id] += 1

        # Pre-compute visa costs for every temporary worker
        visa_cost_cache = {}
        for worker_id, worker in worker_lookup.items():
            if worker.is_temporary:
                visa_cost = 1 + worker.spouse_count + children_by_parent.get(worker_id, 0)
                visa_cost_cache[worker_id] = visa_cost

        return visa_cost_cache

    def process_conversions(
        self,
        annual_limit: int,
        annual_eb_caps: Dict[EBCategory, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[int]],
        per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
        worker_lookup: Dict[int, Worker],
        child_processor: "ChildProcessor",
        current_year: int,
    ) -> Tuple[
        int,
        Dict[str, int],
        Dict[EBCategory, int],
        Set[int],
        int,
        int,
        Dict[Tuple[EBCategory, str], int],
        Dict[EBCategory, int],
    ]:
        """
        Process EB conversions for current year.

        Routes to capped or uncapped processing based on hybrid approach:
        - Reconstruction period (<=2024): ALWAYS capped (matches reality)
        - Projection period (2025+): Uses scenario-specific processing framework.

        Args:
            annual_limit: Total visa budget for the year
            annual_eb_caps: Base allocation by category before cascade
            category_nationality_queues: Queues organized by (category, nationality)
            per_country_caps_by_category: 7% caps per country per category
            worker_lookup: All workers in the system
            child_processor: Tracks dependent children
            current_year: Current simulation year

        Returns:
            Tuple of (total_conversions, by_country, by_category, worker_ids,
                     total_visas, spouse_count, detailed_consumption, pass2_visas_by_category)
        """
        # Pre-compute all visa costs once for the year
        visa_cost_cache = self._build_visa_cost_cache(worker_lookup, child_processor)

        # Determine capping policy based on period
        is_reconstruction = current_year <= RECONSTRUCTION_END_YEAR
        use_caps = is_reconstruction or self.country_cap_enabled

        if use_caps:
            # Use realistic two-pass allocation with 7% per-country cap
            return self._process_capped_cascade(
                annual_limit,
                annual_eb_caps,
                category_nationality_queues,
                per_country_caps_by_category,
                worker_lookup,
                visa_cost_cache,
                current_year,
            )
        else:
            # Pure FIFO allocation without per-country cap
            result = self._process_uncapped(
                annual_limit,
                annual_eb_caps,
                category_nationality_queues,
                worker_lookup,
                visa_cost_cache,
                current_year,
            )
            # Uncapped has no Pass 2, return zeros
            pass2_zeros = {cat: 0 for cat in EBCategory}
            return (*result, pass2_zeros)

    def _shuffle_applicants_within_cohorts(
        self, applicants: List[int], worker_lookup: Dict[int, Worker]
    ) -> List[int]:
        """
        Shuffle applicants within priority date cohorts to prevent nationality clustering.

        Without this, all workers from one country who entered in the same year
        would process together as a block. This creates artificial "waves" where
        one nationality dominates conversions for several years, then another takes over.

        Within-cohort shuffling preserves FIFO fairness (earlier cohorts still process
        first).

        Example:
        Before: [India2020, India2020, India2020, China2020, China2020, India2021]
        After:  [India2020, China2020, India2020, China2020, India2020, India2021]

        Args:
            applicants: Worker IDs sorted by entry_year (priority date)
            worker_lookup: Dictionary to look up worker details

        Returns:
            Same workers, shuffled within each priority date cohort
        """
        # Group workers by the year they entered the queue (priority date)
        cohorts = defaultdict(list)
        for worker_id in applicants:
            entry_year = worker_lookup[worker_id].entry_year
            cohorts[entry_year].append(worker_id)

        # Shuffle within each year's cohort, but keep years in order
        # This preserves FIFO fairness while preventing nationality clustering
        shuffled_applicants = []
        for entry_year in sorted(cohorts.keys()):
            cohort = cohorts[entry_year]
            self.rng.shuffle(cohort)  # In-place shuffle
            shuffled_applicants.extend(cohort)

        return shuffled_applicants

    def _process_uncapped(
        self,
        annual_limit: int,
        annual_eb_caps: Dict[EBCategory, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[int]],
        worker_lookup: Dict[int, Worker],
        visa_cost_cache: Dict[int, int],
        current_year: int,
    ) -> Tuple[
        int,
        Dict[str, int],
        Dict[EBCategory, int],
        Set[int],
        int,
        int,
        Dict[Tuple[EBCategory, str], int],
    ]:
        """
        Process conversions WITHOUT per-country caps.

        Applicants process strictly by priority date regardless of nationality.

        Cascade still applies (unused visas flow up the chain), but there's no
        artificial throttling of any particular country.

        Processing order:
        1. EB-4 (spillover -> EB-1)
        2. EB-5 (spillover -> EB-1)
        3. EB-1 (gets EB-4/EB-5 spillover, spillover -> EB-2)
        4. EB-2 (gets EB-1 spillover, spillover -> EB-3)
        5. EB-3 (gets EB-2 spillover, no further cascade)

        Args:
            annual_limit: Total visa budget for the year
            annual_eb_caps: Base allocation per category
            category_nationality_queues: Queues by (category, country)
            worker_lookup: All workers
            visa_cost_cache: Pre-computed visa costs
            current_year: Current simulation year

        Returns:
            Conversion results tuple
        """
        # Initialize tracking structures
        converted_worker_ids = set()
        conversions_by_country = defaultdict(int)
        conversions_by_category = {cat: 0 for cat in EBCategory}
        total_visas_consumed = 0
        converted_spouses = 0
        visas_consumed_by_category_nationality = defaultdict(int)

        # Track available visas per category
        # This increases as unused visas cascade down from higher categories
        available_visas = {cat: annual_eb_caps.get(cat, 0) for cat in EBCategory}

        # Process categories in cascade order (EB-4/EB-5 -> EB-1 -> EB-2 -> EB-3)
        for eb_category in CASCADE_PROCESSING_ORDER:
            # Stop if we've exhausted the entire annual visa pool
            if total_visas_consumed >= annual_limit:
                break

            category_limit = available_visas.get(eb_category, 0)

            # Get all applicants for this category across all nationalities
            applicants = self._get_applicants_for_category(
                eb_category, category_nationality_queues, worker_lookup
            )
            # Sort by priority date (entry year), then by worker ID
            # The worker ID tiebreaker ensures the shuffle gets identical input every run
            applicants.sort(key=lambda wid: (worker_lookup[wid].entry_year, wid))

            # Shuffle within each year's cohort to prevent nationality clustering
            applicants = self._shuffle_applicants_within_cohorts(applicants, worker_lookup)

            visas_used_in_category = 0
            converted_ids_set = set()

            # Process applicants in strict priority order (no per-country cap)
            for worker_id in applicants:
                worker = worker_lookup[worker_id]
                visa_cost = visa_cost_cache[worker_id]

                # Stop if category budget is completely exhausted
                if visas_used_in_category >= category_limit:
                    break  # Even single workers won't fit

                # Stop if annual pool is completely exhausted
                if total_visas_consumed >= annual_limit:
                    break  # No visas left in entire annual allocation

                # Check if this worker's family is too large for remaining category budget
                if visas_used_in_category + visa_cost > category_limit:
                    continue  # This family is too big, but smaller families might fit

                # Check if this worker's family would exceed annual pool
                if total_visas_consumed + visa_cost > annual_limit:
                    continue  # This family is too big, but smaller families might fit

                # Convert worker and family
                worker.convert_to_permanent(current_year)
                converted_worker_ids.add(worker.id)
                converted_ids_set.add(worker_id)
                conversions_by_country[worker.nationality] += 1
                conversions_by_category[eb_category] += 1
                visas_used_in_category += visa_cost
                total_visas_consumed += visa_cost
                converted_spouses += worker.spouse_count

                # Track detailed consumption
                visas_consumed_by_category_nationality[(eb_category, worker.nationality)] += visa_cost

            # Remove converted workers from their nationality-specific queues
            for key, queue in list(category_nationality_queues.items()):
                cat, nat = key
                if cat == eb_category:
                    # Filter out converted workers from this queue
                    category_nationality_queues[key] = deque(
                        wid for wid in queue if wid not in converted_ids_set
                    )

            logger.debug(
                f"  {eb_category.value}: {conversions_by_category[eb_category]} principals, {visas_used_in_category} visas / {category_limit}"
            )

            # Calculate unused visas and cascade to next category
            original_allocation = available_visas.get(eb_category, 0)
            unused_visas = original_allocation - visas_used_in_category
            if unused_visas > 0:
                logger.debug(f"  {unused_visas:,} unused visas from {eb_category.value}")

                # Apply cascade flow rules
                # Example: EB-4 unused -> EB-1, EB-5 unused -> EB-1, etc.
                flows = CASCADE_FLOWS.get(eb_category, {})
                for receiving_category in flows.get("sends_to", []):
                    available_visas[receiving_category] += unused_visas
                    logger.debug(f"    Cascading {unused_visas:,} to {receiving_category.value}")

        total_conversions = len(converted_worker_ids)
        logger.info(
            f"Year {current_year} (Uncapped): {total_conversions} principals using {total_visas_consumed:,} visas"
        )

        return (
            total_conversions,
            dict(conversions_by_country),
            conversions_by_category,
            converted_worker_ids,
            total_visas_consumed,
            converted_spouses,
            dict(visas_consumed_by_category_nationality),
        )

    def _process_capped_cascade(
        self,
        annual_limit: int,
        annual_eb_caps: Dict[EBCategory, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[int]],
        per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
        worker_lookup: Dict[int, Worker],
        visa_cost_cache: Dict[int, int],
        current_year: int,
    ) -> Tuple[
        int,
        Dict[str, int],
        Dict[EBCategory, int],
        Set[int],
        int,
        int,
        Dict[Tuple[EBCategory, str], int],
        Dict[EBCategory, int],
    ]:
        """
        Process conversions WITH per-country caps and cascade.

        Two-pass allocation per category:
        - Pass 1: Process with 7% per-country cap, identify oversubscribed countries
        - Pass 2: Process overflow from oversubscribed countries without cap

        Processing order:
        1. EB-4 (spillover -> EB-1)
        2. EB-5 (spillover -> EB-1)
        3. EB-1 (gets EB-4/EB-5 spillover, spillover -> EB-2)
        4. EB-2 (gets EB-1 spillover, spillover -> EB-3)
        5. EB-3 (gets EB-2 spillover, no further cascade)

        Args:
            annual_limit: Total visa budget
            annual_eb_caps: Base allocation per category
            category_nationality_queues: Queues by (category, country)
            per_country_caps_by_category: 7% caps
            worker_lookup: All workers
            visa_cost_cache: Pre-computed visa costs
            current_year: Current year

        Returns:
            Conversion results tuple including Pass 2 visas by category
        """
        # Initialize tracking structures
        converted_worker_ids = set()
        conversions_by_country = defaultdict(int)
        conversions_by_category = {cat: 0 for cat in EBCategory}
        total_visas_consumed = 0
        converted_spouses = 0
        visas_consumed_by_category_nationality = defaultdict(int)
        pass2_visas_by_category = {cat: 0 for cat in EBCategory}

        # Track available visas per category (increases with cascade)
        available_visas = {cat: annual_eb_caps.get(cat, 0) for cat in EBCategory}

        # Track cumulative visa usage per country per category for 7% cap enforcement
        # This persists across Pass 1 and Pass 2 to maintain accurate country caps
        cumulative_visas_by_country_category = defaultdict(lambda: defaultdict(int))

        # Process categories in cascade order (EB-4/EB-5 -> EB-1 -> EB-2 -> EB-3)
        for eb_category in CASCADE_PROCESSING_ORDER:
            # Stop if we've exhausted the entire annual visa pool
            if total_visas_consumed >= annual_limit:
                break

            category_limit = available_visas.get(eb_category, 0)

            # Apply historical demand constraints
            if eb_category == EBCategory.EB4 and current_year in EB4_HISTORICAL_DEMAND:
                historical_demand = EB4_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-4 historical demand constraint for {current_year}: {historical_demand} visas (allocation: {category_limit})"
                )
                category_limit = min(category_limit, historical_demand)

            if eb_category == EBCategory.EB5 and current_year in EB5_HISTORICAL_DEMAND:
                historical_demand = EB5_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-5 historical demand constraint for {current_year}: {historical_demand} visas (allocation: {category_limit})"
                )
                category_limit = min(category_limit, historical_demand)

            if eb_category == EBCategory.EB1 and current_year in EB1_HISTORICAL_DEMAND:
                historical_demand = EB1_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-1 historical demand constraint for {current_year}: {historical_demand} visas (allocation: {category_limit})"
                )
                category_limit = min(category_limit, historical_demand)

            logger.info(f"Processing {eb_category.value} with {category_limit:,} visas (Year {current_year})")

            # Check for nationality-specific demand constraints
            # These reflect years when ROW exhausted demand early in the fiscal year
            india_eb1_cap = None
            if eb_category == EBCategory.EB1 and current_year in EB1_INDIA_HISTORICAL_DEMAND:
                india_eb1_cap = EB1_INDIA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-1 India historical demand constraint for {current_year}: {india_eb1_cap} visas for India"
                )

            china_eb1_cap = None
            if eb_category == EBCategory.EB1 and current_year in EB1_CHINA_HISTORICAL_DEMAND:
                china_eb1_cap = EB1_CHINA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-1 China historical demand constraint for {current_year}: {china_eb1_cap} visas for China"
                )

            row_eb1_cap = None
            if eb_category == EBCategory.EB1 and current_year in EB1_ROW_HISTORICAL_DEMAND:
                row_eb1_cap = EB1_ROW_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-1 ROW historical demand constraint for {current_year}: {row_eb1_cap} visas for ROW"
                )

            row_eb2_cap = None
            if eb_category == EBCategory.EB2 and current_year in EB2_ROW_HISTORICAL_DEMAND:
                row_eb2_cap = EB2_ROW_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-2 ROW historical demand constraint for {current_year}: {row_eb2_cap} visas for ROW"
                )

            india_eb2_cap = None
            if eb_category == EBCategory.EB2 and current_year in EB2_INDIA_HISTORICAL_DEMAND:
                india_eb2_cap = EB2_INDIA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-2 India historical demand constraint for {current_year}: {india_eb2_cap} visas for India"
                )

            china_eb2_cap = None
            if eb_category == EBCategory.EB2 and current_year in EB2_CHINA_HISTORICAL_DEMAND:
                china_eb2_cap = EB2_CHINA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-2 China historical demand constraint for {current_year}: {china_eb2_cap} visas for China"
                )

            row_eb3_cap = None
            if eb_category == EBCategory.EB3 and current_year in EB3_ROW_HISTORICAL_DEMAND:
                row_eb3_cap = EB3_ROW_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-3 ROW historical demand constraint for {current_year}: {row_eb3_cap} visas for ROW"
                )

            india_eb3_cap = None
            if eb_category == EBCategory.EB3 and current_year in EB3_INDIA_HISTORICAL_DEMAND:
                india_eb3_cap = EB3_INDIA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-3 India historical demand constraint for {current_year}: {india_eb3_cap} visas for India"
                )

            china_eb3_cap = None
            if eb_category == EBCategory.EB3 and current_year in EB3_CHINA_HISTORICAL_DEMAND:
                china_eb3_cap = EB3_CHINA_HISTORICAL_DEMAND[current_year]
                logger.info(
                    f"EB-3 China historical demand constraint for {current_year}: {china_eb3_cap} visas for China"
                )

            # Calculate remaining annual budget for this category
            remaining_annual_budget = annual_limit - total_visas_consumed

            # Run two-pass allocation for this category
            # Pass 1: Enforce 7% per-country cap
            # Pass 2: Process overflow without country cap
            (
                visas_allocated,
                newly_converted,
                spouses_converted,
                category_nat_visas,
                pass2_visas,
            ) = self._allocate_category_two_pass(
                eb_category,
                category_limit,
                per_country_caps_by_category.get(eb_category, {}),
                category_nationality_queues,
                worker_lookup,
                visa_cost_cache,
                cumulative_visas_by_country_category,
                remaining_annual_budget,
                current_year,
                india_eb1_cap,
                china_eb1_cap,
                row_eb1_cap,
                row_eb2_cap,
                india_eb2_cap,
                china_eb2_cap,
                row_eb3_cap,
                india_eb3_cap,
                china_eb3_cap,
            )

            # Accumulate results from this category
            converted_worker_ids.update(newly_converted)
            converted_spouses += spouses_converted

            # Store Pass 2 visas for this category
            pass2_visas_by_category[eb_category] = pass2_visas

            # Accumulate detailed consumption by (category, nationality) pairs
            for key, count in category_nat_visas.items():
                visas_consumed_by_category_nationality[key] += count

            # Update country and category conversion counts
            for worker_id in newly_converted:
                worker = worker_lookup[worker_id]
                conversions_by_country[worker.nationality] += 1
                conversions_by_category[eb_category] += 1

            # Calculate actual visas used (principal + spouse + children)
            visas_actually_used = sum(visa_cost_cache.get(wid, 0) for wid in newly_converted)
            total_visas_consumed += visas_actually_used

            # Calculate unused visas and cascade to next category
            original_allocation = available_visas.get(eb_category, 0)
            unused_visas = original_allocation - visas_allocated
            if unused_visas > 0:
                logger.debug(f"  {unused_visas:,} unused visas from {eb_category.value}")

                # Apply cascade flow rules
                # Example: EB-4 unused -> EB-1, EB-5 unused -> EB-1, etc.
                flows = CASCADE_FLOWS.get(eb_category, {})
                for receiving_category in flows.get("sends_to", []):
                    available_visas[receiving_category] += unused_visas
                    logger.debug(f"    Cascading {unused_visas:,} to {receiving_category.value}")

            # Remove converted workers from their nationality-specific queues
            converted_ids_set = set(newly_converted)
            for key, queue in list(category_nationality_queues.items()):
                cat, nat = key
                if cat == eb_category:
                    # Filter out converted workers from this queue
                    category_nationality_queues[key] = deque(
                        wid for wid in queue if wid not in converted_ids_set
                    )

        total_conversions = len(converted_worker_ids)
        logger.info(
            f"Year {current_year} (Capped): {total_conversions} principals using {total_visas_consumed:,} visas"
        )

        return (
            total_conversions,
            dict(conversions_by_country),
            conversions_by_category,
            converted_worker_ids,
            total_visas_consumed,
            converted_spouses,
            dict(visas_consumed_by_category_nationality),
            pass2_visas_by_category,
        )

    def _allocate_category_two_pass(
        self,
        eb_category: EBCategory,
        category_limit: int,
        per_country_cap: Dict[str, int],
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[int]],
        worker_lookup: Dict[int, Worker],
        visa_cost_cache: Dict[int, int],
        cumulative_visas_by_country_category,
        remaining_annual_budget: int,
        current_year: int,
        india_eb1_cap: int = None,
        china_eb1_cap: int = None,
        row_eb1_cap: int = None,
        row_eb2_cap: int = None,
        india_eb2_cap: int = None,
        china_eb2_cap: int = None,
        row_eb3_cap: int = None,
        india_eb3_cap: int = None,
        china_eb3_cap: int = None,
    ) -> Tuple[int, Set[int], int, Dict[Tuple[EBCategory, str], int], int]:
        """
        Two-pass allocation for a single EB category.

        Pass 1: Enforce 7% per-country cap
        - Process applicants by priority date
        - Stop each country at its 7% limit
        - Collect rejected applicants from oversubscribed countries

        Pass 2: Process oversubscribed overflow without country cap
        - Take applicants rejected in Pass 1
        - Process by priority date without country limit
        - Only constrained by category total and global pool

        Args:
            eb_category: Category being processed
            category_limit: Total visas for this category (after cascade)
            per_country_cap: 7% caps per country
            category_nationality_queues: Queues by (category, country)
            worker_lookup: All workers
            visa_cost_cache: Pre-computed visa costs
            cumulative_visas_by_country_category: Running count for cap enforcement
            remaining_annual_budget: Remaining visas in annual pool
            current_year: Current year
            india_eb1_cap: Optional India demand ceiling for EB-1
            china_eb1_cap: Optional China demand ceiling for EB-1
            row_eb1_cap: Optional ROW demand ceiling for EB-1
            row_eb2_cap: Optional ROW demand ceiling for EB-2
            india_eb2_cap: Optional India demand ceiling for EB-2
            china_eb2_cap: Optional China demand ceiling for EB-2
            row_eb3_cap: Optional ROW demand ceiling for EB-3
            india_eb3_cap: Optional India demand ceiling for EB-3
            china_eb3_cap: Optional China demand ceiling for EB-3

        Returns:
            (visas_allocated, converted_worker_ids, spouse_count, detailed_consumption, pass2_visas)
        """
        # Initialize tracking structures for this category
        converted_ids = set()
        visas_allocated = 0
        converted_spouses = 0
        visas_by_category_nationality = defaultdict(int)

        # Get all applicants for this category across all nationalities
        applicants = self._get_applicants_for_category(
            eb_category, category_nationality_queues, worker_lookup
        )
        # Sort by priority date (entry year), then by worker ID
        # The worker ID tiebreaker ensures the shuffle gets identical input every run
        applicants.sort(key=lambda wid: (worker_lookup[wid].entry_year, wid))

        # Shuffle within each year's cohort to prevent nationality clustering
        applicants = self._shuffle_applicants_within_cohorts(applicants, worker_lookup)

        # Pass 1: With 7% per-country cap
        # Enforce the statutory 7% per-country limit within this category
        pass1_rejected = []  # Workers who hit their country's 7% cap
        pass1_converted_by_country = defaultdict(int)

        for worker_id in applicants:
            # If category budget is exhausted, stop at Pass 1
            # No point continuing since we can't convert anyone else
            if visas_allocated >= category_limit:
                break

            worker = worker_lookup[worker_id]
            visa_cost = visa_cost_cache[worker_id]

            # Get this worker's country cap
            country_cap = per_country_cap.get(worker.nationality, 0)
            # Get cumulative visas already used by this country in this category
            cumulative = cumulative_visas_by_country_category[eb_category][worker.nationality]

            # Check nationality-specific demand constraints
            # These reflect years when ROW exhausted its demand early
            if india_eb1_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb1_cap:
                    # India has exhausted its demand for this year - skip this worker entirely
                    # No point sending to Pass 2 since the constraint still applies there
                    continue

            if china_eb1_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb1_cap:
                    continue

            if row_eb1_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb1_cap:
                    continue

            if row_eb2_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb2_cap:
                    continue

            if india_eb2_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb2_cap:
                    continue

            if china_eb2_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb2_cap:
                    continue

            if row_eb3_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb3_cap:
                    continue

            if india_eb3_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb3_cap:
                    continue

            if china_eb3_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb3_cap:
                    continue

            # Check if this worker would exceed their country's 7% cap
            if cumulative + visa_cost > country_cap:
                # This country has hit its limit - send worker to Pass 2
                # Pass 2 will process them without the country cap
                pass1_rejected.append(worker_id)
                continue

            # Check if this worker's family would exceed the category limit
            if visas_allocated + visa_cost > category_limit:
                # This family is too big for remaining category budget
                # No point in sending to Pass 2 since the category limit still applies there
                # Smaller families might still fit, so continue
                continue

            # All checks passed - convert this worker and their family
            worker.convert_to_permanent(current_year)
            converted_ids.add(worker_id)
            visas_allocated += visa_cost
            converted_spouses += worker.spouse_count

            # Update cumulative country usage (persists across Pass 1 and Pass 2)
            cumulative_visas_by_country_category[eb_category][worker.nationality] += visa_cost
            pass1_converted_by_country[worker.nationality] += visa_cost

            # Track detailed consumption by (category, nationality)
            visas_by_category_nationality[(eb_category, worker.nationality)] += visa_cost

        logger.debug(f"    Pass 1 Results:")
        logger.debug(f"      Converted: {len(converted_ids)} principals, {visas_allocated:,} visas")
        logger.debug(f"      Rejected by country cap: {len(pass1_rejected)} workers")

        # Pass 2: Without 7% per-country cap
        # Process overflow from oversubscribed countries
        # No country cap applies here - only category limit and annual pool matter

        # Track Pass 2 visa usage separately
        pass2_visas_start = visas_allocated
        pass2_converted_by_country = defaultdict(int)
        pass2_start_size = len(converted_ids)

        for worker_id in pass1_rejected:
            # If category budget is completely exhausted, stop Pass 2
            # No visas left at all in this category
            if visas_allocated >= category_limit:
                break

            worker = worker_lookup[worker_id]
            visa_cost = visa_cost_cache[worker_id]

            # These constraints represent demand ceilings and must apply in both passes
            cumulative = cumulative_visas_by_country_category[eb_category][worker.nationality]

            if india_eb1_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb1_cap:
                    continue

            if china_eb1_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb1_cap:
                    continue

            if row_eb1_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb1_cap:
                    continue

            if row_eb2_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb2_cap:
                    continue

            if india_eb2_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb2_cap:
                    continue

            if china_eb2_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb2_cap:
                    continue

            if row_eb3_cap is not None and worker.nationality == "ROW":
                if cumulative + visa_cost > row_eb3_cap:
                    continue

            if india_eb3_cap is not None and worker.nationality == "India":
                if cumulative + visa_cost > india_eb3_cap:
                    continue

            if china_eb3_cap is not None and worker.nationality == "China":
                if cumulative + visa_cost > china_eb3_cap:
                    continue

            # If this worker's family is too large for remaining category budget,
            # skip them - smaller families might still fit
            if visas_allocated + visa_cost > category_limit:
                continue

            # If this worker's family would exceed the remaining annual pool,
            # skip them - smaller families might still fit
            if visas_allocated + visa_cost > remaining_annual_budget:
                continue

            # All checks passed - convert this worker and their family
            worker.convert_to_permanent(current_year)
            converted_ids.add(worker_id)
            visas_allocated += visa_cost
            converted_spouses += worker.spouse_count

            # Update cumulative country usage (persists from Pass 1)
            cumulative_visas_by_country_category[eb_category][worker.nationality] += visa_cost
            pass2_converted_by_country[worker.nationality] += visa_cost

            # Track detailed consumption by (category, nationality)
            visas_by_category_nationality[(eb_category, worker.nationality)] += visa_cost

        # Calculate Pass 2 visas used
        pass2_visas_used = visas_allocated - pass2_visas_start

        # Log Pass 2 results
        pass2_conversions = len(converted_ids) - pass2_start_size
        logger.debug(f"    Pass 2 Results:")
        logger.debug(
            f"      Converted: {pass2_conversions} principals, {sum(pass2_converted_by_country.values()):,} visas"
        )
        logger.debug(f"      Total for category: {len(converted_ids)} principals, {visas_allocated:,} visas")

        return (
            visas_allocated,
            converted_ids,
            converted_spouses,
            dict(visas_by_category_nationality),
            pass2_visas_used,
        )

    def _get_applicants_for_category(
        self,
        eb_category: EBCategory,
        category_nationality_queues: Dict[Tuple[EBCategory, str], Deque[int]],
        worker_lookup: Dict[int, Worker],
    ) -> List[int]:
        """
        Get all applicants waiting in a given EB category across all nationalities.

        Args:
            eb_category: EB category to retrieve
            category_nationality_queues: Queues organized by (category, country)
            worker_lookup: All workers (unused here, kept for API consistency)

        Returns:
            List of worker IDs in this category's queues
        """
        applicants = []

        # Iterate through all (category, nationality) queue pairs
        # Collect worker IDs from queues matching this category
        for (cat, nat), queue in category_nationality_queues.items():
            if cat == eb_category:
                applicants.extend(list(queue))

        return applicants
