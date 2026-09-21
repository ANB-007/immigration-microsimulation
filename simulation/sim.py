"""
Simulation engine for the immigration microsimulation.

Implements the multi-year agent-based model of the EB green card queue,
including worker inflows, visa allocation with and without per-country caps,
queue exits, and the full lifecycle of dependent children. Coordinates
category- and nationality-specific queues, calls out to visa and child
processors, and records rich annual state snapshots for downstream analysis
of backlogs, age-outs, and visa consumption.
"""


import logging
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque
import numpy as np

import json
from .input_data import DATA_DIR

from .models import Worker, WorkerStatus, EBCategory
from .states import SimulationConfig, SimulationState
from .child_processor import ChildAgeoutStatistics

from .empirical_params import (
    TOTAL_EB_VISA_POOL,
    COUNTRIES,
    calculate_annual_eb_caps,
    calculate_per_country_caps_by_category,
    get_spouse_probability,
    get_pathway_totals,
    get_total_eb_visa_pool,
    get_nationality_distribution_for_pathway,
    get_queue_exit_rate,
)
from .visa_processor import VisaProcessor
from .child_processor import ChildProcessor

logger = logging.getLogger(__name__)

PATHWAY_TO_EB_CATEGORY = {
    "EB-1": EBCategory.EB1,
    "EB-2": EBCategory.EB2,
    "EB-3": EBCategory.EB3,
    "SIJS": EBCategory.EB4,
    "Other_EB4": EBCategory.EB4,
    "EB-5": EBCategory.EB5,
}


class Simulation:
    """
    Agent-based microsimulation of the employment-based green card queue.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation engine using numpy's SeedSequence.spawn() to create
        independent RNG streams from a single master seed.

        Args:
            config: Simulation configuration specifying years, seed, policy settings, etc.
        """
        self.config = config
        self.current_year = config.start_year
        self.states: List[SimulationState] = []  # Year-by-year simulation history

        # Create the master seed that parents all other random streams
        self.master_seed_seq = np.random.SeedSequence(config.seed)

        # Spawn 4 cryptographically independent child sequences for major components
        seeds = self.master_seed_seq.spawn(4)

        self.worker_creation_seed_seq = seeds[0]  # For sampling nationalities, ages, family structures
        self.queue_exit_seed_seq = seeds[1]  # For deciding who leaves the queue before conversion
        child_processor_seed_seq = seeds[2]  # For child demographics (number of kids, their ages)
        visa_processor_seed_seq = seeds[3]  # For green card allocation decisions

        # Each petition pathway gets its own RNG stream to prevent order-dependence
        pathway_seeds = self.worker_creation_seed_seq.spawn(len(PATHWAY_TO_EB_CATEGORY))
        self.worker_creation_rngs: Dict[str, np.random.Generator] = {}
        for idx, pathway in enumerate(sorted(PATHWAY_TO_EB_CATEGORY.keys())):
            self.worker_creation_rngs[pathway] = np.random.default_rng(pathway_seeds[idx])

        # Queue exits use a single shared RNG since they happen across all categories
        self.queue_exit_rng = np.random.default_rng(self.queue_exit_seed_seq)

        # Visa processor handles the complex logic of allocating visas each year,
        # including spillover and cascade effects
        self.visa_processor = VisaProcessor(
            country_cap_enabled=config.country_cap_enabled,
            seed_seq=visa_processor_seed_seq,
        )

        # worker_lookup: Dictionary for O(1) access to any worker by their unique ID
        # temp_worker_ids: Set of workers currently in queue (not yet permanent residents)
        # perm_worker_ids: Set of workers who successfully received green cards
        self.worker_lookup: Dict[int, Worker] = {}
        self.temp_worker_ids: Set[int] = set()
        self.perm_worker_ids: Set[int] = set()

        self.next_worker_id = 1  # Increments for each new worker we create

        # Calculate base visa allocations before spillover/cascade effects
        self.base_annual_eb_caps = calculate_annual_eb_caps(TOTAL_EB_VISA_POOL)
        self.base_per_country_caps_by_category = calculate_per_country_caps_by_category(
            self.base_annual_eb_caps
        )

        # Load empirical principal age distributions from JSON file
        # Structure: pathway -> nationality -> {ages, probabilities}
        parent_age_dist_path = (
            DATA_DIR / "demographics" / "principal_age_empirical_distributions.json"
        )  
        with open(parent_age_dist_path, "r") as f:
            self.parent_age_distributions = json.load(f)

        logger.info(f"Simulation initialized: {config.years} years")

        eb_caps_str = {cat.value: cap for cat, cap in self.base_annual_eb_caps.items()}
        logger.info(f"Base EB category proportions: {eb_caps_str}")

        self.country_cap_enabled = config.country_cap_enabled
        if self.country_cap_enabled:
            logger.info("Per-country caps ENABLED (7% per country limit within each EB category)")
        else:
            logger.info("Per-country caps DISABLED (pure first-in-first-out within each category)")

        # Create separate FIFO queues for each (EB category, nationality) pair
        # For example: (EB-2, India), (EB-2, China), (EB-2, ROW) are three separate queues
        self.category_nationality_queues: Dict[Tuple[EBCategory, str], deque] = {}
        for category in EBCategory:
            for nationality in COUNTRIES:
                key = (category, nationality)
                self.category_nationality_queues[key] = deque()

        # Cumulative counters track totals across the entire simulation run
        self.cumulative_conversions = 0  # Total workers who received green cards
        self.cumulative_children_aged = 0  # Total children who aged out at 21
        self.cumulative_visas_consumed = 0  # Total green cards issued (including dependents)
        self.cumulative_queue_exits = 0  # Total who left queue without converting

        # The child processor tracks every child from
        # creation through one of four possible outcomes:
        # 1. Dependent (still under 21, parent still waiting)
        # 2. Saved (parent got green card before child turned 21)
        # 3. Aged out (turned 21 while parent still waiting)
        # 4. Exited (parent left queue, child exits with them)
        self.child_processor = ChildProcessor(child_processor_seed_seq, debug=config.debug)

    def _assign_eb_category(self, pathway: str) -> EBCategory:
        """
        Map petition pathway to EB category.

        We could inline this lookup, but extracting it to a method makes the code
        more readable and easier to modify.

        Args:
            pathway: Petition pathway (EB-1, EB-2, EB-3, SIJS, Other_EB4, EB-5)

        Returns:
            The EB category this pathway belongs to
        """
        return PATHWAY_TO_EB_CATEGORY[pathway]

    def _assign_spouse_probability(self, worker: Worker, pathway: str, rng: np.random.Generator) -> None:
        """
        Determine if this worker has a spouse. 

        Special Immigrant Juveniles are unaccompanied minors entering the system alone, 
        so they never have a spouse.

        Args:
            worker: Worker object to modify (will set worker.spouse_count)
            pathway: Petition pathway--used to detect SIJS override
            rng: Random number generator for this pathway (ensures reproducibility)
        """
        # Fetch empirical marriage probability for this nationality/pathway combination
        spouse_probability = get_spouse_probability(worker.nationality, pathway)

        # If random number <=probability, they have a spouse
        worker.spouse_count = 1 if rng.random() <= spouse_probability else 0

    def _add_new_workers_by_pathway(self, current_year: int) -> None:
        """
        Create and enqueue all new workers entering the green card queue this year.

        Args:
            current_year: The simulation year we're processing (e.g., 2020)
        """
        # Get synthetic entry totals by pathway; EB-1/2/3 use receipt-cohort inputs.
        # Returns something like: {"EB-1": 40070, "EB-2": 80345, "EB-3": 35000, ...}
        pathway_totals = get_pathway_totals(current_year)

        for pathway, total_entries in pathway_totals.items():

            # Get nationality distribution for this specific pathway and year
            # Returns something like: {"India": 0.59, "China": 0.19, "ROW": 0.22}
            nationality_distribution = get_nationality_distribution_for_pathway(pathway, current_year)

            nationalities = list(nationality_distribution.keys())
            probabilities = list(nationality_distribution.values())

            # Normalize probabilities to handle floating-point rounding errors
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()

            # Use this pathway's dedicated RNG stream for reproducibility
            # This isolation means that EB-1 worker creation won't affect EB-2 randomness
            rng = self.worker_creation_rngs[pathway]

            # Step 1: Sample nationality for each worker
            # Example: if total_entries=5, might get nat_list = ["India", "China", "India", "ROW", "India"]
            nat_indices = rng.choice(len(nationalities), size=total_entries, p=probabilities)
            nat_list = [nationalities[i] for i in nat_indices]

            # Step 2: Sample ages matched to each worker's nationality using empirical distributions
            # Initialize array with zeros as placeholder (will be filled with actual ages)
            ages = np.zeros(total_entries, dtype=int)

            # Process each nationality group separately to match ages to nationalities
            for nationality_index, nationality in enumerate(nationalities):
                # Create boolean mask: True at positions where worker is this nationality
                # Example: if nat_list = ["India", "China", "India"], and nationality = "India"
                #          then mask = [True, False, True]
                mask = nat_indices == nationality_index

                # Count how many workers belong to this nationality
                # Example: mask.sum() = 2 (two Indian workers in this example)
                n_workers = mask.sum()

                if n_workers > 0:
                    # Sample n_workers ages from this pathway+nationality's empirical distribution
                    # Structure is pathway -> nationality -> {ages, probabilities}
                    # SIJS has fall back to uniform draw over [14, 20]
                    if pathway in self.parent_age_distributions:
                        age_dist = self.parent_age_distributions[pathway][nationality]
                        sampled_ages = rng.choice(
                            a=age_dist["ages"],
                            size=n_workers,
                            p=age_dist["probabilities"],
                        )
                    else:
                        sampled_ages = rng.integers(14, 21, size=n_workers)

                    # Assign sampled ages ONLY to positions where mask is True
                    # Example: ages[mask] = [32, 35] assigns ages to positions 0 and 2
                    #          Result: ages = [32, 0, 35] (position 1 still zero, will be filled by China loop)
                    ages[mask] = sampled_ages

            # Step 3: Create individual worker objects
            # At this point, nat_list and ages are perfectly aligned by index
            # Example: nat_list[0]="India" and ages[0]=32, nat_list[1]="China" and ages[1]=38, etc.
            for i in range(total_entries):
                eb_category = self._assign_eb_category(pathway)

                # Create the worker agent with unique demographics
                worker = Worker(
                    id=self.next_worker_id,
                    status=WorkerStatus.TEMPORARY,
                    nationality=nat_list[i],
                    age=int(ages[i]),
                    entry_year=current_year,
                    eb_category=eb_category,
                    pathway=pathway,
                )

                # Determine spouse existence using nationality-specific probabilities
                # Must happen before create_children_for_worker
                self._assign_spouse_probability(worker, pathway, rng)

                # Add to central tracking data structures
                self.worker_lookup[worker.id] = worker
                self.temp_worker_ids.add(worker.id)

                # Enqueue in the appropriate (category, nationality) FIFO queue
                # This is THE queue they'll wait in until conversion or exit
                key = (worker.eb_category, worker.nationality)
                if key in self.category_nationality_queues:
                    self.category_nationality_queues[key].append(worker.id)

                # Create children for this worker using empirical distributions
                # sample_children_count gates on worker.spouse_count; returns 0 if no spouse
                self.child_processor.create_children_for_worker(worker, current_year, pathway, rng)

                # Increment worker ID counter for next worker
                self.next_worker_id += 1

    def _track_exited(
        self, exited_ids: List[int]
    ) -> Tuple[Dict[str, int], Dict[EBCategory, int], Dict[Tuple[EBCategory, str], int]]:
        """
        Count exited workers by nationality and EB category BEFORE marking them as exited.

        When workers leave the queue without getting green cards (emigration, death,
        petition withdrawal, etc.), we need to track where they came from for statistical 
        reporting. We aggregate exits across three dimensions:

        1. By nationality: How many Indians vs Chinese vs ROW left this year?

        2. By EB category: How many EB-1 vs EB-2 vs EB-3 workers exited?

        3. By (category, nationality): How many EB-2 Indians left? EB-3 Chinese?

        Args:
            exited_ids: List of worker IDs who are leaving the system this year

        Returns:
            Tuple of three dictionaries with exit counts broken down by:
            (by_nationality, by_eb_category, by_category_nationality)
        """
        exited_by_nationality = defaultdict(int)
        exited_by_eb_category = defaultdict(int)
        exited_by_category_nationality = defaultdict(int)

        # Loop through exited workers and tally them across all three dimensions
        for worker_id in exited_ids:
            worker = self.worker_lookup[worker_id]

            # Count by nationality (India, China, ROW)
            exited_by_nationality[worker.nationality] += 1
            # Count by EB category and (category, nationality) combo
            exited_by_eb_category[worker.eb_category] += 1
            # Also count by the fine-grained (category, nationality) pair
            key = (worker.eb_category, worker.nationality)
            exited_by_category_nationality[key] += 1

        # Convert defaultdicts to regular dicts for cleaner output
        return (
            dict(exited_by_nationality),
            dict(exited_by_eb_category),
            dict(exited_by_category_nationality),
        )

    def _remove_queue_exits(self, queue_exit_ids: List[int], current_year: int) -> None:
        """
        Mark workers as exited and remove them from all active queues.

        When a parent exits, their children must also exit.

        We don't remove workers from worker_lookup because we want to preserve the complete
        historical record of everyone who entered the system.

        Args:
            queue_exit_ids: List of worker IDs to remove from active queues
            current_year: Year this exit is happening (stored on worker for analysis)
        """
        queue_exit_id_set = set(queue_exit_ids)

        # Remove from the set of temporary (in-queue) workers
        self.temp_worker_ids.difference_update(queue_exit_id_set)

        # Remove from all category-nationality queues
        self._clean_queues(queue_exit_id_set)

        # Remove all children whose parents are exiting the queue
        self.child_processor.remove_children_of_queue_exits_batch(
            current_year, queue_exit_ids, self.worker_lookup
        )

        # Mark each worker as exited (but keep them in worker_lookup for tracking)
        for worker_id in queue_exit_ids:
            worker = self.worker_lookup[worker_id]
            worker.mark_as_exited(current_year)

    def _process_conversions(
        self,
        current_year: int,
        total_slots: int,
        annual_eb_caps: Dict[EBCategory, int],
        per_country_caps_by_category: Dict[EBCategory, Dict[str, int]],
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
        Convert temporary workers to permanent residents.

        The VisaProcessor orchestrates the complex two-pass allocation algorithm. 

        Args:
            current_year: Year being processed
            total_slots: Total visas available this year (varies by year)
            annual_eb_caps: Base allocation by EB category before cascade
            per_country_caps_by_category: 7% caps within each category

        Returns:
            Tuple of (total_conversions, conversions_by_country, conversions_by_category,
                     converted_worker_ids, total_visas_consumed, converted_spouses,
                     visas_consumed_by_category_nationality, pass2_visas_by_category)
        """
        # Delegate the complex allocation logic to VisaProcessor
        (
            total_conversions,
            conversions_by_country,
            conversions_by_category,
            converted_worker_ids,
            total_visas_consumed,
            converted_spouses,
            visas_consumed_by_category_nationality,
            pass2_visas_by_category,
        ) = self.visa_processor.process_conversions(
            annual_limit=total_slots,
            annual_eb_caps=annual_eb_caps,
            category_nationality_queues=self.category_nationality_queues,
            per_country_caps_by_category=per_country_caps_by_category,
            worker_lookup=self.worker_lookup,
            child_processor=self.child_processor,
            current_year=current_year,
            active_worker_ids=self.temp_worker_ids,
        )

        # Move workers from temporary status to permanent status
        self.temp_worker_ids.difference_update(converted_worker_ids)
        self.perm_worker_ids.update(converted_worker_ids)

        # Remove from all queues since they've received their green cards
        self._clean_queues(converted_worker_ids)

        return (
            total_conversions,
            conversions_by_country,
            conversions_by_category,
            converted_worker_ids,
            total_visas_consumed,
            converted_spouses,
            visas_consumed_by_category_nationality,
            pass2_visas_by_category,
        )

    def _clean_queues(self, removed_worker_ids: Set[int]) -> None:
        """
        Remove workers from all queue data structures.

        When workers either convert to permanent residents or exit the queue, they
        must be removed from the category-nationality queues.

        Args:
            removed_worker_ids: Set of worker IDs to remove from all queues
        """
        # Filter each category-nationality queue, keeping only workers NOT in removed_worker_ids
        for key in self.category_nationality_queues:
            self.category_nationality_queues[key] = deque(
                wid for wid in self.category_nationality_queues[key] if wid not in removed_worker_ids
            )

    def _calculate_queue_backlogs(self) -> Dict[str, int]:
        """
        Count principal applicants waiting in queue by nationality.

        Returns:
            Dictionary mapping each nationality (India, China, ROW) to principal worker count
        """
        backlogs = defaultdict(int)

        # Sum queue lengths across all categories for each nationality
        # Example: India total = EB-1 India + EB-2 India + EB-3 India + EB-4 India + EB-5 India
        for (
            category,
            nationality,
        ), queue in self.category_nationality_queues.items():
            backlogs[nationality] += len(queue)

        return dict(backlogs)

    def _calculate_eb_queue_backlogs(
        self,
    ) -> Tuple[Dict[EBCategory, int], Dict[Tuple[EBCategory, str], int]]:
        """
        Count principal applicants waiting by EB category and category-nationality cells.

        This provides two complementary views of the backlog:

        1. By category: Total EB-2 backlog across all countries

        2. By (category, nationality): EB-2 India vs EB-2 China vs EB-2 ROW

        Returns:
            Tuple of two dictionaries:
            - by_category: Maps each EB category to total worker count
            - by_category_nationality: Maps each (category, nationality) pair to worker count
        """
        backlog_by_category = {cat: 0 for cat in EBCategory}
        backlog_by_category_nationality = {}

        # Count workers in each queue
        for (
            category,
            nationality,
        ), queue in self.category_nationality_queues.items():
            queue_size = len(queue)

            # Aggregate by category (sum across nationalities)
            backlog_by_category[category] += queue_size

            # Track by (category, nationality) pair for granular view
            backlog_by_category_nationality[(category, nationality)] = queue_size

        return backlog_by_category, backlog_by_category_nationality

    def step(self) -> SimulationState:
        """
        Execute one simulation year--the main event loop.

        Order of operations:

        1. Add new workers using this year's synthetic entry totals

        2. Convert to green cards: Allocate visas via two-pass algorithm
           Must happen BEFORE saving children so we know which children were saved

        3. Save children: Remove children whose parents converted
           Must happen BEFORE aging so saved children don't age out

        4. Process queue exits: Emigration, death, marriage to citizens, etc.
           Must happen AFTER conversions so exits do not pre-empt visa allocation

        5. Calculate total family visa consumption by category

        6. Age all workers and children: Increment age by 1 year; some children will age out at 21
           Must happen AFTER conversions so we don't age children who were saved

        7. Calculate statistics: Backlog counts, age-out rates, demographic breakdowns
           Must happen AFTER all updates for accurate snapshots

        Returns:
            SimulationState object capturing complete metrics for this year
        """
        t_total_start = time.perf_counter()  # Start total timer for performance monitoring

        # Get the model visa budget from the stored historical reconstruction series.
        # Future years repeat the 2024 model budget (~167K visas).
        total_visa_pool_this_year = get_total_eb_visa_pool(self.current_year)

        # Calculate how visas are split across categories and countries this year
        # Base allocation: 28.6% each for EB-1/2/3, 7.1% each for EB-4/5
        # Per-country cap: 7% of each category's allocation per country
        annual_eb_caps = calculate_annual_eb_caps(total_visa_pool_this_year)
        per_country_caps_by_category = calculate_per_country_caps_by_category(annual_eb_caps)
        # The global budget is authoritative even if category calculations change.
        annual_sim_cap = total_visa_pool_this_year

        ##############################################################
        # Step 1: Add new workers using this year's synthetic entry totals
        ##############################################################
        t1 = time.perf_counter()
        self._add_new_workers_by_pathway(self.current_year)
        t_add_workers = time.perf_counter() - t1

        #########################################
        # Step 2: Process green card conversions
        #########################################
        t2 = time.perf_counter()

        (
            converted_temps,
            conversions_by_country,
            conversions_by_category,
            converted_worker_ids,
            total_visas_consumed,
            converted_spouses,
            visas_consumed_by_category_nationality,
            pass2_visas_by_category,
        ) = self._process_conversions(
            self.current_year,
            annual_sim_cap,
            annual_eb_caps,
            per_country_caps_by_category,
        )

        # Update cumulative counters
        self.cumulative_conversions += converted_temps
        self.cumulative_visas_consumed += total_visas_consumed
        t_conversions = time.perf_counter() - t2

        ###################################################################################
        # Step 3: Remove children of converted parents (they got green cards derivatively)
        ###################################################################################
        t3 = time.perf_counter()
        children_saved, children_saved_by_parent = self.child_processor.remove_children_of_converted_parents(
            self.current_year, converted_worker_ids, self.worker_lookup
        )
        t_remove_converted = time.perf_counter() - t3

        #############################################################################
        # Step 4: Process queue exits (people leaving without receiving green cards)
        #############################################################################
        t4 = time.perf_counter()

        # Calculate exit probability for each worker currently in queue
        worker_ids = list(self.temp_worker_ids)
        n_workers = len(worker_ids)

        # Pre-allocate exit probabilities
        queue_exit_rates = np.zeros(n_workers, dtype=np.float64)
        for i, worker_id in enumerate(worker_ids):
            worker = self.worker_lookup[worker_id]
            years_in_us = self.current_year - worker.entry_year
            queue_exit_rates[i] = get_queue_exit_rate(
                nationality=worker.nationality,
                eb_category=worker.eb_category,
                current_year=self.current_year,
                years_in_us=years_in_us,
                age=worker.age,
            )

        # Probabilistically determine whether a worker exits this year
        random_values = self.queue_exit_rng.random(n_workers)
        queue_exit_mask = random_values < queue_exit_rates
        queue_exit_ids = [worker_ids[i] for i in range(n_workers) if queue_exit_mask[i]]

        # Track exit statistics BEFORE removing workers (we need their attributes)
        (
            exited_by_nationality,
            exited_by_eb_category,
            exited_by_category_nationality,
        ) = self._track_exited(queue_exit_ids)

        # Now remove exited workers from all queue structures
        if queue_exit_ids:
            self._remove_queue_exits(queue_exit_ids, self.current_year)

        queue_exits_this_year = len(queue_exit_ids)

        # Update cumulative counter
        self.cumulative_queue_exits += queue_exits_this_year
        t_queue_exit = time.perf_counter() - t4

        ##############################################################
        # Step 5: Calculate total family visa consumption by category
        ##############################################################
        t5 = time.perf_counter()
        converted_by_eb_category_total = {cat: 0 for cat in EBCategory}
        for worker_id in converted_worker_ids:
            worker = self.worker_lookup[worker_id]
            if worker.eb_category:
                # Count principal worker + spouse + children
                converted_by_eb_category_total[worker.eb_category] += 1
                converted_by_eb_category_total[worker.eb_category] += worker.spouse_count
                num_children = children_saved_by_parent.get(worker_id, 0)
                converted_by_eb_category_total[worker.eb_category] += num_children
        t_eb_category = time.perf_counter() - t5

        ###################################################
        # Step 6: Age all workers and children by one year
        ###################################################
        t6 = time.perf_counter()

        # Age all workers
        for worker in self.worker_lookup.values():
            worker.update_age()

        # Age all children and process age-outs
        previous_ageouts = len(self.child_processor.aged_out_children)
        children_aged_out_this_year = self.child_processor.process_child_aging(
            self.current_year, self.worker_lookup
        )
        self.cumulative_children_aged += children_aged_out_this_year

        t_aging = time.perf_counter() - t6

        ########################################################
        # Step 7: Build list of children who aged out this year
        ########################################################
        t7 = time.perf_counter()
        children_aged_out_this_year_list = self.child_processor.aged_out_children[previous_ageouts:]
        t_aged_out_list = time.perf_counter() - t7

        ###########################################################
        # Step 8: Calculate all backlog and demographic statistics
        ###########################################################
        t8 = time.perf_counter()
        children_aged_out_by_nationality = self.child_processor.get_cumulative_aged_out_by_nationality()
        children_ageout_percentage_by_nationality = self.child_processor.get_ageout_percentage_by_nationality(
            COUNTRIES
        )
        pathway_totals = get_pathway_totals(self.current_year)
        total_new_entries = sum(pathway_totals.values())
        queue_backlogs_by_country = self._calculate_queue_backlogs()
        queue_backlogs_by_category, queue_backlogs_by_cat_nat = self._calculate_eb_queue_backlogs()
        t_backlogs = time.perf_counter() - t8

        # Log summary of this year's events
        logger.info(
            f"Year {self.current_year}: +{total_new_entries:,} new queue entries, "
            f"{queue_exits_this_year} exited queue, "
            f"{converted_temps} converted ({total_visas_consumed:,}/{total_visa_pool_this_year:,} visas consumed), "
            f"{children_saved} children saved, {children_aged_out_this_year} children aged out"
        )

        ###########################################################
        # Step 9: Build comprehensive state snapshot for this year
        ###########################################################
        t9 = time.perf_counter()

        # Calculate worker counts across different statuses
        n_temps = len(self.temp_worker_ids)  # Still waiting in queue
        n_perms = len(self.perm_worker_ids)  # Got green cards
        n_exited = self.cumulative_queue_exits
        n_total = len(self.worker_lookup)  # Everyone who ever entered

        # Calculate what fraction are still temporary (in queue)
        temporary_share = n_temps / n_total if n_total else 0.0

        # Get child age-out statistics from child processor
        child_stats = ChildAgeoutStatistics(
            total_aged_out=len(self.child_processor.aged_out_children),
            aged_out_this_year=children_aged_out_this_year,
            children_at_risk=len(self.child_processor.dependent_children),
            aged_out_by_nationality=dict(self.child_processor.aged_out_by_nationality),
            aged_out_by_eb_category=dict(self.child_processor.aged_out_by_eb_category),
        )

        # Package everything into a comprehensive state snapshot
        new_state = SimulationState(
            year=self.current_year,
            total_workers=n_total,
            permanent_workers=n_perms,
            temporary_workers=n_temps,
            exited_workers=n_exited,
            new_permanent=converted_temps,
            new_temporary=total_new_entries,
            converted_temps=converted_temps,
            cumulative_conversions=self.cumulative_conversions,
            temporary_share=temporary_share,
            children_aged_out_this_year=children_aged_out_this_year,
            cumulative_children_aged_out=self.cumulative_children_aged,
            children_at_risk=child_stats.children_at_risk,
            aged_out_by_nationality=child_stats.aged_out_by_nationality,
            converted_by_eb_category=conversions_by_category,
            converted_by_eb_category_total=converted_by_eb_category_total,
            queue_backlog_by_eb_category=queue_backlogs_by_category,
            queue_backlog_by_eb_category_nationality=queue_backlogs_by_cat_nat,
            aged_out_by_eb_category=child_stats.aged_out_by_eb_category,
            converted_by_country=conversions_by_country,
            queue_backlog_by_country=queue_backlogs_by_country,
            country_cap_enabled=self.country_cap_enabled,
            annual_conversion_cap=annual_sim_cap,
            visas_consumed_this_year=total_visas_consumed,
            cumulative_visas_consumed=self.cumulative_visas_consumed,
            visas_available_this_year=total_visa_pool_this_year,
            converted_spouses=converted_spouses,
            children_saved_this_year=children_saved,
            children_aged_out_by_nationality=children_aged_out_by_nationality,
            children_created_total_by_nationality=dict(
                self.child_processor.children_created_total_by_nationality
            ),
            children_ageout_percentage_by_nationality=children_ageout_percentage_by_nationality,
            visas_consumed_by_category_nationality=visas_consumed_by_category_nationality,
            children_aged_out_this_year_list=children_aged_out_this_year_list,
            exited_this_year=queue_exits_this_year,
            cumulative_exited=self.cumulative_queue_exits,
            exited_by_nationality=exited_by_nationality,
            exited_by_eb_category=exited_by_eb_category,
            exited_by_category_nationality=exited_by_category_nationality,
            visa_allocations_pass2_annual_by_eb_category=pass2_visas_by_category,
        )
        t_build_state = time.perf_counter() - t9

        # Append to simulation history and advance to next year
        self.states.append(new_state)
        self.current_year += 1

        # Calculate total time for this year
        t_total = time.perf_counter() - t_total_start

        # Log detailed performance timings
        logger.debug(f"Year {self.current_year - 1} step timings (seconds):")
        logger.debug(f"  Add workers:            {t_add_workers:.3f}s")
        logger.debug(f"  Queue exit:             {t_queue_exit:.3f}s")
        logger.debug(f"  Conversions:            {t_conversions:.3f}s")
        logger.debug(f"  Remove converted:       {t_remove_converted:.3f}s")
        logger.debug(f"  EB category calc:       {t_eb_category:.3f}s")
        logger.debug(f"  Aging (workers+children): {t_aging:.3f}s")
        logger.debug(f"  Aged out list:          {t_aged_out_list:.3f}s")
        logger.debug(f"  Calculate backlogs:     {t_backlogs:.3f}s")
        logger.debug(f"  Build state:            {t_build_state:.3f}s")
        logger.debug(f"  Total:                  {t_total:.3f}s")

        return new_state

    def run(self) -> List[SimulationState]:
        """
        Run the complete multi-year simulation from start to finish.

        Executes step() for each year in the simulation.
        Progress is logged (printed in the terminal) every 10 years.

        Returns:
            List of SimulationState objects, one per simulated year, in chronological order
        """
        logger.info(f"Starting {self.config.years}-year simulation...")

        # Execute one year at a time through the simulation period
        for year in range(self.config.years):
            self.step()

            # Log progress every 10 years
            # Also log year 0 to confirm simulation started properly
            if (year + 1) % 10 == 0 or year == 0:
                state = self.states[-1]  # Get the state we just created
                backlog = sum(state.queue_backlog_by_country.values())
                logger.info(
                    f"Year {state.year}: {state.cumulative_children_aged_out} children aged out, "
                    f"{backlog:,} in backlog, {state.cumulative_visas_consumed:,} visas consumed, "
                    f"{self.cumulative_queue_exits} total queue exits"
                )

        # Log final summary statistics to give high-level overview
        final_state = self.states[-1]
        logger.info(f"Simulation completed.")
        logger.info(
            f"  Final workers: {final_state.total_workers:,} "
            f"({final_state.temporary_workers:,} in queue, {final_state.exited_workers:,} exited)"
        )
        logger.info(f"  Total children aged out: {final_state.cumulative_children_aged_out:,}")
        logger.info(f"  Final backlog: {sum(final_state.queue_backlog_by_country.values()):,}")
        logger.info(f"  Total visas consumed: {final_state.cumulative_visas_consumed:,}")
        logger.info(f"  Total queue exits: {self.cumulative_queue_exits:,}")

        return self.states
