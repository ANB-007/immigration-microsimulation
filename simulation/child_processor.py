"""
Child lifecycle management for dependent children in the immigration microsimulation.

Creates, tracks, and updates dependent children attached to workers in the
green card queue, and classifies each child into one of four outcomes:
still dependent, saved through parental conversion, aged out after turning 21,
or exited with a parent who leaves the queue. Provides utilities to summarize
age-out incidence and outcome distributions by nationality, EB category, and
entry cohort for downstream analysis.
"""


import logging
from typing import List, Set, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .models import (
    Worker,
    DependentChild,
    AgedOutChild,
    SavedChild,
    ExitedChild,
    EBCategory,
)
from .empirical_params import (
    CHILD_AGEOUT_AGE,
    sample_children_count,
    sample_child_entry_age,
)

logger = logging.getLogger(__name__)


@dataclass
class ChildAgeoutStatistics:
    """
    Statistics for child age-out tracking.

    Tracks how many children "age out" (turn 21 while their parents wait in the queue)
    and lose eligibility for derivative green cards.
    """

    total_aged_out: int
    aged_out_this_year: int
    children_at_risk: int
    aged_out_by_nationality: Dict[str, int]
    aged_out_by_eb_category: Dict[EBCategory, int]

    @classmethod
    def calculate(
        cls,
        aged_out_children: List[AgedOutChild],
        dependent_children: List[DependentChild],
        aged_out_this_year: int,
    ) -> "ChildAgeoutStatistics":
        """
        Calculate child age-out statistics from current simulation state.

        Aggregates individual child outcomes into summary statistics,
        breaking down age-outs by nationality and EB category to identify
        which populations face the highest risk.

        Args:
            aged_out_children: All children who have aged out (cumulative)
            dependent_children: Children currently in queue (at risk)
            aged_out_this_year: Number who aged out this year

        Returns:
            ChildAgeoutStatistics snapshot for this year
        """
        aged_out_by_nationality = defaultdict(int)
        for child in aged_out_children:
            aged_out_by_nationality[child.nationality] += 1

        aged_out_by_eb_category = defaultdict(int)
        for child in aged_out_children:
            if child.parent_eb_category:
                aged_out_by_eb_category[child.parent_eb_category] += 1

        children_at_risk = len(dependent_children)

        return cls(
            total_aged_out=len(aged_out_children),
            aged_out_this_year=aged_out_this_year,
            children_at_risk=children_at_risk,
            aged_out_by_nationality=dict(aged_out_by_nationality),
            aged_out_by_eb_category=dict(aged_out_by_eb_category),
        )


class ChildProcessor:
    """
    Processes dependent child lifecycle events.

    This class manages all children whose parents are in the green card queue.
    Children are tracked from creation (when parent enters queue) through their
    final outcome (saved, aged out, or exited).
    """

    def __init__(self, seed_seq: np.random.SeedSequence, debug: bool = False):
        """
        Initialize child processor with deterministic random number generation.

        Sets up all data structures needed to track children through their
        lifecycle in the immigration queue.

        Args:
            seed_seq: numpy SeedSequence for reproducible random sampling
            debug: Enable detailed debug logging (default: False)
        """
        self.seed_seq = seed_seq
        self.debug = debug
        self.next_child_id = 1

        # Create independent RNG for child-related random decisions
        # This ensures reproducibility across simulation runs
        self.rng = np.random.default_rng(seed_seq)

        # Main tracking lists for the four possible child states:
        # 1. Dependent: Under 21, parent still waiting
        self.dependent_children: List[DependentChild] = []

        # 2. Saved: Parent got green card before child turned 21
        self.saved_children: List[SavedChild] = []

        # 3. Aged out: Turned 21 while parent still waiting
        self.aged_out_children: List[AgedOutChild] = []

        # 4. Exited: Parent left queue (marriage, death, voluntary emigration, etc.)
        self.exited_children: List[ExitedChild] = []

        # Performance optimization: Index children by birth year
        # This allows fast filtering when processing age-outs each year
        self.children_by_birth_year: Dict[int, List[DependentChild]] = defaultdict(list)

        # Performance optimization: Index children by parent ID
        # This allows O(1) lookup when parent converts or exits queue
        self.children_by_parent_id: Dict[int, List[int]] = defaultdict(list)

        # Track total children created per nationality for final statistics
        # Used to calculate age-out percentages by country
        self.children_created_total_by_nationality: Dict[str, int] = defaultdict(int)

        # This is the denominator for entry-year cohort outcome percentages
        self.children_created_by_entry_year_nationality: Dict[Tuple[int, str], int] = defaultdict(int)

        # This is the denominator for the nationality x EB-category breakdown
        self.children_created_by_entry_year_nationality_eb: Dict[Tuple[int, str, str], int] = defaultdict(int)

    def create_children_for_worker(
        self,
        worker: Worker,
        entry_year: int,
        pathway: str,
        rng: np.random.Generator,
    ) -> List[DependentChild]:
        """
        Create dependent children when a worker enters the green card queue.

        Uses empirically-derived distributions:
        - Sample realistic family sizes by nationality and marital status
        - Assign realistic entry ages for each child

        Children are created at the moment their parent enters the queue.
        Their ages at entry determine when (if ever) they'll age out at 21.

        Args:
            worker: Worker entering the queue (must be temporary status)
            entry_year: Year the worker entered the queue
            pathway: Immigration pathway (e.g. "EB-1")
            rng: Random number generator from parent simulation

        Returns:
            List of newly created DependentChild objects
        """
        # Safety check: Only temporary workers (on H-1B, etc.) have dependents
        # Permanent residents already have green cards
        if not worker.is_temporary:
            raise ValueError(
                f"ERROR: Attempted to create children for non-temporary worker {worker.id}. "
                f"Worker status: {worker.status}."
            )

        children = []

        # Sample number of children from empirical distribution
        # This accounts for nationality-specific family size patterns and marital status
        num_children = sample_children_count(worker.nationality, pathway, worker.spouse_count, rng)

        # No children case
        if num_children == 0:
            return []

        # Resolve EB category string once for this worker
        eb_str = worker.eb_category.value if worker.eb_category else "Unknown"

        # Create each child with realistic characteristics
        for child_index in range(num_children):
            # Sample entry age from empirical distribution
            # Different nationalities have different age patterns at queue entry
            entry_age = sample_child_entry_age(worker.nationality, pathway, rng=rng)

            # Calculate birth year from entry year and age
            # Example: If child enters queue in 2015 at age 10, they were born in 2005
            birth_year = entry_year - entry_age

            # Create child object with all necessary tracking info
            child = DependentChild(
                child_id=self.next_child_id,
                parent_worker_id=worker.id,
                nationality=worker.nationality,
                parent_eb_category=worker.eb_category,
                birth_year=birth_year,
                entry_year=entry_year,
            )
            children.append(child)

            # Add to all tracking structures for efficient processing
            self.children_by_birth_year[birth_year].append(child)
            self.dependent_children.append(child)
            self.children_by_parent_id[worker.id].append(child.child_id)

            # Track creation count for final statistics
            self.children_created_total_by_nationality[worker.nationality] += 1

            # Track creation count per (entry_year, nationality) cohort
            self.children_created_by_entry_year_nationality[(entry_year, worker.nationality)] += 1

            # Track creation count per (entry_year, nationality, eb_category) cohort
            self.children_created_by_entry_year_nationality_eb[(entry_year, worker.nationality, eb_str)] += 1

            self.next_child_id += 1

        return children

    def remove_children_of_converted_parents(
        self,
        current_year: int,
        converted_worker_ids: Set[int],
        worker_lookup: Dict[int, Worker],
    ) -> Tuple[int, Dict[int, int]]:
        """
        Save children whose parents got green cards this year.

        When a parent successfully converts from temporary to permanent status,
        their children are "saved" - they're no longer at risk of aging out.
        This is the positive outcome we're measuring in the simulation.

        Args:
            current_year: Current simulation year
            converted_worker_ids: Set of worker IDs that converted THIS YEAR
            worker_lookup: Dictionary mapping worker IDs to Worker objects

        Returns:
            Tuple of (total_saved_count, dict mapping parent_id to children_saved)
        """
        children_saved_this_year = 0
        children_saved_by_parent: Dict[int, int] = {}
        still_dependent = []
        saved_child_ids = set()

        # Track which birth years are affected (for efficient index updates)
        birth_years_to_update = set()

        # Process each dependent child to see if their parent converted
        for child in self.dependent_children:
            if child.parent_worker_id in converted_worker_ids:
                # Parent got green card before child turned 21
                # This child is now safe from aging out
                saved_child_ids.add(child.child_id)
                children_saved_this_year += 1

                birth_years_to_update.add(child.birth_year)

                # Count children per parent for tracking
                if child.parent_worker_id not in children_saved_by_parent:
                    children_saved_by_parent[child.parent_worker_id] = 0
                children_saved_by_parent[child.parent_worker_id] += 1

                # Create SavedChild record with full details
                if child.parent_worker_id not in worker_lookup:
                    raise ValueError(
                        f"ERROR: Child {child.child_id} has parent {child.parent_worker_id} "
                        f"not in worker_lookup."
                    )

                parent_worker = worker_lookup[child.parent_worker_id]
                age = child.age_in_year(current_year)
                parent_years_in_queue = current_year - parent_worker.entry_year

                saved_child = SavedChild(
                    child_id=child.child_id,
                    parent_worker_id=child.parent_worker_id,
                    nationality=child.nationality,
                    parent_eb_category=parent_worker.eb_category,
                    birth_year=child.birth_year,
                    entry_year=child.entry_year,
                    parent_conversion_year=current_year,
                    age_at_outcome=age,
                    parent_years_in_queue=parent_years_in_queue,
                )
                self.saved_children.append(saved_child)
            else:
                # Parent still waiting - child remains at risk
                still_dependent.append(child)

        # Update main dependent list with only remaining children
        self.dependent_children = still_dependent

        # Update birth year indices (only for affected years - more efficient)
        for birth_year in birth_years_to_update:
            new_children_list = [
                child
                for child in self.children_by_birth_year[birth_year]
                if child.child_id not in saved_child_ids
            ]
            if new_children_list:
                self.children_by_birth_year[birth_year] = new_children_list
            else:
                # No children left for this birth year - remove key
                del self.children_by_birth_year[birth_year]

        # Clean up parent index for converted workers
        for parent_id in converted_worker_ids:
            if parent_id in self.children_by_parent_id:
                del self.children_by_parent_id[parent_id]

        # Log how many children were saved and remain at risk
        logger.info(
            f"Year {current_year}: {children_saved_this_year} children saved, "
            f"{len(self.dependent_children)} still at risk"
        )

        return children_saved_this_year, children_saved_by_parent

    def process_child_aging(self, current_year: int, worker_lookup: Dict[int, Worker]) -> int:
        """
        Process children turning 21 and aging out of dependent status.

        Children who turn 21 while their parent is still waiting lose their dependent status.
        They must either:
        1. Return to home country
        2. Find their own visa (H-1B, etc.)
        3. Become undocumented if they stay

        This represents family separation caused by long green card wait times.

        Args:
            current_year: Current simulation year
            worker_lookup: Dictionary mapping worker IDs to Worker objects

        Returns:
            Number of children who aged out this year
        """
        children_aged_out_this_year = 0
        still_dependent = []

        for child in self.dependent_children:
            # Calculate child's current age
            age = child.age_in_year(current_year)

            # Check if child reached the legal age-out threshold (21 years old)
            if age >= CHILD_AGEOUT_AGE:
                # Child turned 21 while parent still waiting
                if child.parent_worker_id not in worker_lookup:
                    raise ValueError(
                        f"ERROR: Child {child.child_id} has parent {child.parent_worker_id} "
                        f"not in worker_lookup during age-out processing."
                    )

                parent_worker = worker_lookup[child.parent_worker_id]

                # Calculate how long parent has been waiting
                # This is key for understanding which backlogs cause age-outs
                parent_years_in_queue = current_year - parent_worker.entry_year

                # Create aged-out record with full context
                aged_out_child = AgedOutChild(
                    child_id=child.child_id,
                    parent_worker_id=child.parent_worker_id,
                    nationality=child.nationality,
                    parent_eb_category=parent_worker.eb_category,
                    birth_year=child.birth_year,
                    entry_year=child.entry_year,
                    aged_out_year=current_year,
                    age_at_outcome=age,
                    parent_years_in_queue=parent_years_in_queue,
                )

                self.aged_out_children.append(aged_out_child)
                children_aged_out_this_year += 1
            else:
                # Child still under 21 - remains dependent and at risk
                still_dependent.append(child)

        # Update dependent list to exclude newly aged-out children
        self.dependent_children = still_dependent

        return children_aged_out_this_year

    def remove_children_of_queue_exits_batch(
        self,
        current_year: int,
        queue_exit_worker_ids: List[int],
        worker_lookup: Dict[int, Worker],
    ) -> int:
        """
        Process children of ALL exiting parents in a single batch operation.

        Much more efficient than calling remove_children_of_queue_exit() in a loop.
        Processes all exits with one pass through dependent_children list.

        This is the preferred method when multiple workers exit in the same year (which is always the case).

        Args:
            current_year: Current simulation year
            queue_exit_worker_ids: List of ALL worker IDs exiting queue this year
            worker_lookup: Dictionary mapping worker IDs to Worker objects

        Returns:
            Total number of children who exited with parents
        """
        # Handle empty case early
        if not queue_exit_worker_ids:
            return 0

        # Convert to set for O(1) membership checks
        queue_exit_id_set = set(queue_exit_worker_ids)

        # Collect all child IDs that need to be removed
        # Use parent ID index for fast lookup
        child_ids_to_remove = set()
        for queue_exit_id in queue_exit_worker_ids:
            child_ids = self.children_by_parent_id.get(queue_exit_id, [])
            child_ids_to_remove.update(child_ids)

        # No children case - none of the exiting workers had dependents
        if not child_ids_to_remove:
            return 0

        children_exited = len(child_ids_to_remove)

        # Track affected birth years for efficient index updates
        birth_years_to_update = set()

        # Single pass: filter children and create exit records
        new_dependent_children = []

        for child in self.dependent_children:
            if child.child_id in child_ids_to_remove:
                # This child is exiting with parent
                birth_years_to_update.add(child.birth_year)

                # Get parent info for record
                if child.parent_worker_id not in worker_lookup:
                    raise ValueError(
                        f"ERROR: Child {child.child_id} has parent {child.parent_worker_id} "
                        f"not in worker_lookup during queue exit processing."
                    )

                worker = worker_lookup[child.parent_worker_id]
                parent_years_in_queue = current_year - worker.entry_year

                age = child.age_in_year(current_year)
                exited_child = ExitedChild(
                    child_id=child.child_id,
                    parent_worker_id=child.parent_worker_id,
                    nationality=child.nationality,
                    parent_eb_category=worker.eb_category,
                    birth_year=child.birth_year,
                    entry_year=child.entry_year,
                    parent_exit_year=current_year,
                    age_at_outcome=age,
                    parent_years_in_queue=parent_years_in_queue,
                )
                self.exited_children.append(exited_child)
            else:
                # Keep this child (parent not exiting)
                new_dependent_children.append(child)

        self.dependent_children = new_dependent_children

        # Update birth year indices (only affected years)
        for birth_year in birth_years_to_update:
            new_children_list = [
                child
                for child in self.children_by_birth_year[birth_year]
                if child.child_id not in child_ids_to_remove
            ]
            if new_children_list:
                self.children_by_birth_year[birth_year] = new_children_list
            else:
                del self.children_by_birth_year[birth_year]

        # Clean up parent cache for all exiting workers
        for queue_exit_id in queue_exit_id_set:
            if queue_exit_id in self.children_by_parent_id:
                del self.children_by_parent_id[queue_exit_id]

        return children_exited

    def get_cumulative_aged_out_by_nationality(self) -> Dict[str, int]:
        """
        Count total children who aged out by nationality across all simulation years.

        This aggregates all children who turned 21 while their parents were still waiting,
        broken down by nationality.

        Returns:
            Dictionary mapping each nationality to cumulative aged-out count
        """
        aged_out_by_nat = defaultdict(int)

        # Loop through all aged-out children and tally by nationality
        for child in self.aged_out_children:
            aged_out_by_nat[child.nationality] += 1

        return dict(aged_out_by_nat)

    def get_ageout_percentage_by_nationality(self, countries: List[str]) -> Dict[str, float]:
        """
        Calculate what percentage of created children, per nationality, ultimately age out.

        Formula: (aged_out_children_per_nationality / total_children_created_per_nationality) x 100

        This percentage shows what fraction of children lose their path to a green card
        due to extended processing backlogs.

        Args:
            countries: List of country names to calculate percentages for

        Returns:
            Dictionary mapping each nationality to age-out percentage (0-100)
        """
        ageout_percentage = {}

        for nationality in countries:
            # Get total children created for this nationality
            total_created = self.children_created_total_by_nationality.get(nationality, 0)

            # Get total children who aged out for this nationality
            total_aged_out = sum(1 for child in self.aged_out_children if child.nationality == nationality)

            # Calculate percentage
            ageout_percentage[nationality] = (total_aged_out / total_created) * 100

        return ageout_percentage

    def get_total_children_count(self) -> int:
        """
        Get total count of all children across all four lifecycle states.

        The total should equal the sum of all children created throughout the simulation.
        If it doesn't match, we've lost children somewhere in our processing.

        Returns:
            Total children count (dependent + saved + aged out + exited)
        """
        return (
            len(self.dependent_children)
            + len(self.saved_children)
            + len(self.aged_out_children)
            + len(self.exited_children)
        )

    def get_outcomes_by_entry_year_nationality(
        self,
    ) -> Dict[Tuple[int, str], Dict[str, int]]:
        """
        Aggregate child outcomes by (entry_year, nationality) cohort.

        For each cohort, returns total_created, aged_out, saved, and exited.
        Children still dependent at simulation end are not counted in any
        outcome column; callers can derive still_dependent as:
            total_created - aged_out - saved - exited

        Returns:
            Dict mapping (entry_year, nationality) to a dict with keys:
                total_created, aged_out, saved, exited
        """
        # Seed results dict with denominators from the creation tracker
        results: Dict[Tuple[int, str], Dict[str, int]] = {
            key: {
                "total_created": count,
                "aged_out": 0,
                "saved": 0,
                "exited": 0,
            }
            for key, count in self.children_created_by_entry_year_nationality.items()
        }

        # Tally each terminal outcome -- one pass per list
        for child in self.aged_out_children:
            key = (child.entry_year, child.nationality)
            if key in results:
                results[key]["aged_out"] += 1

        for child in self.saved_children:
            key = (child.entry_year, child.nationality)
            if key in results:
                results[key]["saved"] += 1

        for child in self.exited_children:
            key = (child.entry_year, child.nationality)
            if key in results:
                results[key]["exited"] += 1

        return results

    def get_outcomes_by_entry_year_nationality_eb(
        self,
    ) -> Dict[Tuple[int, str, str], Dict[str, int]]:
        """
        Aggregate child outcomes by (entry_year, nationality, eb_category) cohort.

        Identical logic to get_outcomes_by_entry_year_nationality() but adds EB
        category as a third dimension.

        eb_category is the string value of EBCategory (e.g. "EB-1", "EB-2", "EB-3").
        Children whose parent had no EB category assigned are grouped under "Unknown".

        Children still dependent at simulation end are not counted in any outcome
        column; callers can derive still_dependent as:
            total_created - aged_out - saved - exited

        Returns:
            Dict mapping (entry_year, nationality, eb_category) to a dict with keys:
                total_created, aged_out, saved, exited
        """
        # Seed results dict with denominators from the creation tracker
        results: Dict[Tuple[int, str, str], Dict[str, int]] = {
            key: {
                "total_created": count,
                "aged_out": 0,
                "saved": 0,
                "exited": 0,
            }
            for key, count in self.children_created_by_entry_year_nationality_eb.items()
        }

        def _eb(child) -> str:
            return child.parent_eb_category.value if child.parent_eb_category else "Unknown"

        # Tally each terminal outcome
        for child in self.aged_out_children:
            key = (child.entry_year, child.nationality, _eb(child))
            if key in results:
                results[key]["aged_out"] += 1

        for child in self.saved_children:
            key = (child.entry_year, child.nationality, _eb(child))
            if key in results:
                results[key]["saved"] += 1

        for child in self.exited_children:
            key = (child.entry_year, child.nationality, _eb(child))
            if key in results:
                results[key]["exited"] += 1

        return results
