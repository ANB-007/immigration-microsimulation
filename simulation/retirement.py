"""Logistic age retention and explicitly distinct queue-exit specifications.

The parameters describe a survival curve, not an annual exit probability:
S(median_age)=0.5 and S(near_zero_age)=tail_probability. Annual conditional
withdrawal is 1-S(a+1)/S(a), so repeated application recovers that curve.
This is a declared assumption, not an estimated EB retirement law. The catchall
specification takes the maximum of the conditional age hazard and the retained
tenure/nationality/category baseline. These describe overlapping reasons for
total cessation, not independent causes to be added together.
"""
from contextlib import contextmanager
from functools import lru_cache
import math


def _parameters(median_age, near_zero_age, tail_probability):
    values = (median_age, near_zero_age, tail_probability)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Retirement parameters must be finite numbers")
    if not 18 <= median_age < near_zero_age <= 99 or not 0 < tail_probability < .5:
        raise ValueError("Require 18 <= median_age < near_zero_age <= 99 and 0 < tail_probability < .5")
    return math.log((1 - tail_probability) / tail_probability) / (near_zero_age - median_age)


def retirement_survival(age, median_age=60.0, near_zero_age=65.0, tail_probability=.05):
    """Hypothetical willingness/ability survival; condition on entry separately."""
    slope = _parameters(median_age, near_zero_age, tail_probability)
    if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age):
        raise ValueError("Age must be finite")
    value = slope * (age - median_age)
    if value >= 0:
        e = math.exp(-value)
        return e / (1 + e)
    return 1 / (1 + math.exp(value))


@lru_cache(maxsize=1024)
def retirement_hazard(age, median_age=60.0, near_zero_age=65.0, tail_probability=.05):
    """P(withdraw between a and a+1 | still willing at a); no hard cutoff."""
    slope = _parameters(median_age, near_zero_age, tail_probability)
    if type(age) is not int or not 0 <= age < 120:
        raise ValueError("Modeled retirement age must be an integer in 0–119")
    # Algebraically 1 - (1+exp(x))/(1+exp(x+slope)), evaluated without overflow.
    next_age_cdf = 1 - retirement_survival(age + 1, median_age, near_zero_age, tail_probability)
    return -math.expm1(-slope) * next_age_cdf


@contextmanager
def catchall_retention_exit_rates(median_age=60.0, near_zero_age=65.0,
                                  tail_probability=.05,
                                  categories=("EB-1", "EB-2", "EB-3", "EB-4", "EB-5")):
    """One total cessation hazard: max(retained baseline, conditional age risk).

    The age schedule includes all reasons for permanently ending pursuit of
    the modeled route, including death. No separate mortality/retirement risk
    or legacy annual logistic floor is stacked onto it. Existing tenure,
    nationality and category effects remain a lower bound on this same total
    hazard. The maximum is an explicit overlap assumption, not cause-specific
    estimation or an independence assumption.

    All five EB categories receive the same age schedule by default. Category
    modifiers continue to apply to the baseline. All entrants are conditioned
    on being active when they enter. Only future annual hazards are applied,
    consistently in reconstruction and projection.

    With no visa service, total retention from a0 to a1 is at most S(a1)/S(a0),
    and equals that ratio wherever the age hazard exceeds the baseline in
    every interval. The baseline can additionally reduce early-age retention;
    the age-curve 50%/5% anchors do not describe the whole waiting stock.
    """
    from . import empirical_params as ep
    from . import sim
    from .models import EBCategory

    _parameters(median_age, near_zero_age, tail_probability)
    if not isinstance(categories, (tuple, list)) or not categories or len(set(categories)) != len(categories):
        raise ValueError("Catchall categories must be a nonempty unique list")
    if not set(categories).issubset({"EB-1", "EB-2", "EB-3", "EB-4", "EB-5"}):
        raise ValueError("Unknown catchall category")
    covered = frozenset(categories)
    original_ep, original_sim = ep.get_queue_exit_rate, sim.get_queue_exit_rate
    if any(getattr(function, "_catchall_adjusted", False)
           for function in (original_ep, original_sim)):
        raise ValueError("Catchall retention cannot be applied twice")
    rates = tuple(retirement_hazard(age, median_age, near_zero_age, tail_probability)
                  for age in range(120))
    # Snapshot all modifiers at entry to the isolated specification. There are
    # only three tenure bands; a small exact lookup avoids nested Python rate
    # calls, repeated multiplications and max evaluations for every principal.
    tenure_rates = tuple(ep.QUEUE_EXIT_RATE_PROJECTION_BY_TENURE[name]
                         for name in ("recently_arrived", "intermediate", "settled"))
    nationality_factors = {None: 1.0, **dict(ep.NATIONALITY_EMIGRATION_MULTIPLIERS)}
    category_factors = dict(ep.CATEGORY_EMIGRATION_MULTIPLIERS)
    rate_table = {}
    for nationality, nationality_factor in nationality_factors.items():
        for category in EBCategory:
            bands = []
            for tenure_rate in tenure_rates:
                baseline = tenure_rate * nationality_factor * category_factors.get(category, 1.0)
                if not 0.0 <= baseline <= 1.0:
                    raise ValueError("Catchall baseline must be a finite probability in [0,1]")
                bands.append(tuple(max(baseline, rate) for rate in rates)
                             if category.value in covered else (baseline,) * 120)
            rate_table[(nationality, category)] = tuple(bands)
    original_floor = ep.AGE_EXIT_ENABLED

    def adjusted(nationality, eb_category, current_year, years_in_us, age):
        if type(age) is not int or not 0 <= age < 120:
            raise ValueError("Modeled catchall age must be an integer in 0–119")
        band = 0 if years_in_us < 5 else 1 if years_in_us < 10 else 2
        try:
            return rate_table[(nationality, eb_category)][band][age]
        except KeyError:
            return rate_table[(None, eb_category)][band][age]

    adjusted._catchall_adjusted = True
    ep.AGE_EXIT_ENABLED = False
    ep.get_queue_exit_rate = sim.get_queue_exit_rate = adjusted
    try:
        yield
    finally:
        ep.get_queue_exit_rate, sim.get_queue_exit_rate = original_ep, original_sim
        ep.AGE_EXIT_ENABLED = original_floor
