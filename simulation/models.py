"""
Core data models for the immigration microsimulation.

Defines typed representations of principal workers and their dependent
children, including status transitions (temporary, permanent, exited) and
child outcomes (saved, aged out, exited). These models provide a consistent
schema for tracking queue tenure, family composition, and visa usage across
all components of the simulation.
"""


from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .child_processor import ChildProcessor


class WorkerStatus(Enum):
    """
    Immigration status of a worker in the queue system.

    TEMPORARY: Worker is in the queue waiting for a green card (has H-1B, L-1, etc.)
    PERMANENT: Worker has received their green card and left the queue
    EXITED: Worker left the queue without receiving green card (emigration, death, etc.)
    """

    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    EXITED = "exited"


class EBCategory(Enum):
    """
    Employment-based green card preference categories.
    """

    EB1 = "EB-1"
    EB2 = "EB-2"
    EB3 = "EB-3"
    EB4 = "EB-4"
    EB5 = "EB-5"


@dataclass
class Worker:
    """
    Individual worker in green card queue.

    Represents a principal applicant who has filed an EB petition
    and is waiting for a green card under one of the five EB categories.

    Each worker may have a spouse and children who are also waiting in
    the queue as derivative beneficiaries.
    """

    id: int  # Unique identifier for this worker
    status: WorkerStatus  # Current immigration status
    nationality: str  # Country of birth
    age: int  # Age when entering the queue
    entry_year: int  # Year the worker entered the queue (priority date)
    eb_category: EBCategory  # EB category of the worker's petition
    pathway: str  # Petition pathway
    conversion_year: Optional[int] = None  # Year converted to permanent (if converted)
    exit_year: Optional[int] = None  # Year exited from queue (if exited)
    spouse_count: int = 0  # Number of spouses (0 or 1 in practice)

    @property
    def is_temporary(self) -> bool:
        """Check if worker is still waiting in the queue."""
        return self.status == WorkerStatus.TEMPORARY

    @property
    def is_permanent(self) -> bool:
        """Check if worker has received their green card."""
        return self.status == WorkerStatus.PERMANENT

    @property
    def is_exited(self) -> bool:
        """Check if worker left the queue without receiving green card."""
        return self.status == WorkerStatus.EXITED

    def years_in_system(self, current_year: int) -> int:
        """
        Calculate how long this worker has been waiting in the queue.

        Args:
            current_year: Current simulation year

        Returns:
            Number of years spent in queue system
        """
        return current_year - self.entry_year

    def update_age(self) -> None:
        """
        Update worker's age based on current simulation year.

        This should be called once per simulation year to keep age current.

        Args:
            current_year: Current simulation year
        """
        self.age += 1

    def convert_to_permanent(self, year: int) -> None:
        """
        Convert worker from temporary to permanent status.

        This happens when the worker receives their green card and exits
        the queue. Their family members also receive green cards at this time.

        Args:
            year: Year of conversion
        """
        self.status = WorkerStatus.PERMANENT
        self.conversion_year = year

    def mark_as_exited(self, year: int) -> None:
        """
        Mark worker as exited from the queue without receiving green card.

        Args:
            year: Year of exit
        """
        self.status = WorkerStatus.EXITED
        self.exit_year = year

    def visa_cost(self, child_processor: "ChildProcessor") -> int:
        """
        Calculate total visas consumed when this worker converts.

        Cost = 1 (principal) + spouse_count + num_dependent_children

        Args:
            child_processor: ChildProcessor instance to count dependent children

        Returns:
            Total number of visas consumed by this worker and their family
        """

        visa_cost = 1  # principal
        visa_cost += self.spouse_count

        # Count dependent children
        num_children = len([c for c in child_processor.dependent_children if c.parent_worker_id == self.id])
        visa_cost += num_children

        return visa_cost


@dataclass
class DependentChild:
    """
    Dependent child of a worker still in queue.

    Tracks a child from when their parent enters the queue until one of
    three outcomes occurs:
    - Parent converts → child becomes SavedChild (gets green card)
    - Parent leaves system → child becomes ExitedChild (leaves with parent)
    - Child turns 21 → child becomes AgedOutChild (loses eligibility)
    """

    child_id: int  # Unique identifier for this child
    parent_worker_id: int  # ID of parent worker in queue
    parent_eb_category: EBCategory  # Parent's EB category
    nationality: str  # Country of birth (inherited from parent)
    birth_year: int  # Year of birth
    entry_year: int  # Year parent entered queue

    def age_in_year(self, year: int) -> int:
        """
        Calculate child's age in a given simulation year.

        Args:
            year: Simulation year

        Returns:
            Child's age at end of year
        """
        return year - self.birth_year

    def years_since_entry(self, year: int) -> int:
        """
        Calculate how long the child has been waiting in the queue.

        This measures the time from when the parent entered the queue,
        not from the child's birth.

        Args:
            year: Simulation year

        Returns:
            Years spent in queue system
        """
        return year - self.entry_year


@dataclass
class AgedOutChild:
    """
    Child who aged out (turned 21) while parent was still in queue.

    Note: age_at_outcome is always 21 but included to maintain structural
    consistency with SavedChild and ExitedChild for analysis purposes.
    """

    child_id: int
    parent_worker_id: int
    parent_eb_category: EBCategory
    nationality: str
    birth_year: int
    entry_year: int
    aged_out_year: int  # Year the child turned 21
    age_at_outcome: int  # Child's age at age-out (always 21); included for consistency
    parent_years_in_queue: int


@dataclass
class SavedChild:
    """
    Child whose parent converted to permanent status before aging out.
    """

    child_id: int
    parent_worker_id: int
    parent_eb_category: EBCategory
    nationality: str
    birth_year: int
    entry_year: int
    parent_conversion_year: int  # Year parent received green card
    age_at_outcome: int  # Child's age when parent converted (< 21)
    parent_years_in_queue: int


@dataclass
class ExitedChild:
    """
    Child whose parent left the queue system without converting.
    """

    child_id: int
    parent_worker_id: int
    parent_eb_category: EBCategory
    nationality: str
    birth_year: int
    entry_year: int
    parent_exit_year: int  # Year parent left the system
    age_at_outcome: int  # Child's age when parent exited
    parent_years_in_queue: int
