"""Five public verification checks for the simulation engine.

Run from the repository root after installing the project dependencies:
    python -m unittest discover -s tests -p test_public_checks.py -v

The checks use the declared primary exit specification and small synthetic
inflows and visa budgets over FY2009-FY2040. They exercise the real engine
without generating publication outputs or importing private development tests.
"""

import json
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from simulation import experiments, sim
from simulation.models import EBCategory
from simulation.states import SimulationConfig


ROOT = Path(__file__).resolve().parents[1]
DESIGN = json.loads((ROOT / "data/experiment.json").read_text())
PRIMARY = next(case for case in DESIGN["cases"] if case["name"] == DESIGN["primary_case"])
START_YEAR = DESIGN["start_year"]
POLICY_YEAR, END_YEAR = DESIGN["event_years"]
SEED = DESIGN["seed"]
EXIT_SETTINGS = {
    "exit_specification": DESIGN["exit_specification"],
    "catchall_median_age": PRIMARY["median_age"],
    "catchall_near_zero_age": PRIMARY["near_zero_age"],
    "catchall_tail_probability": DESIGN["catchall_tail_probability"],
    "catchall_categories": DESIGN["catchall_categories"],
}


def visa_pool(year):
    """Include zero supply, very small budgets, and competing demand."""
    return (0, 5, 40, 70)[(year - START_YEAR) % 4]


def child_snapshot(processor):
    groups = {
        "waiting": processor.dependent_children,
        "saved": processor.saved_children,
        "aged_out": processor.aged_out_children,
        "exited": processor.exited_children,
    }
    return {
        "created_by_nationality": dict(processor.children_created_total_by_nationality),
        "observed_by_nationality": dict(Counter(
            child.nationality for children in groups.values() for child in children
        )),
        "ids": {
            outcome: tuple(sorted(child.child_id for child in children))
            for outcome, children in groups.items()
        },
    }


def run_pair():
    """Run fresh capped/uncapped models; no saved records are reused."""
    result = {}
    inflows = {pathway: 15 for pathway in sim.PATHWAY_TO_EB_CATEGORY}
    with (
        patch.object(sim, "get_pathway_totals", side_effect=lambda year: dict(inflows)),
        patch.object(sim, "get_total_eb_visa_pool", side_effect=visa_pool),
        experiments.parameter_override(EXIT_SETTINGS),
    ):
        for name, capped in (("Uncapped", False), ("Capped", True)):
            model = sim.Simulation(SimulationConfig(
                years=END_YEAR - START_YEAR + 1,
                seed=SEED,
                output_path="",
                country_cap_enabled=capped,
                debug=False,
                start_year=START_YEAR,
            ))
            children = []
            for _ in range(model.config.years):
                model.step()
                children.append(child_snapshot(model.child_processor))
            result[name] = {
                "states": model.states,
                "children": children,
                "workers": {key: asdict(worker) for key, worker in model.worker_lookup.items()},
            }
    return result


class PublicVerificationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pair = run_pair()

    def test_01_visa_allocations_stay_within_the_pool(self):
        for scenario, record in self.pair.items():
            total = 0
            for state in record["states"]:
                with self.subTest(scenario=scenario, year=state.year):
                    available = visa_pool(state.year)
                    self.assertEqual(state.visas_available_this_year, available)
                    self.assertGreaterEqual(state.visas_consumed_this_year, 0)
                    self.assertLessEqual(state.visas_consumed_this_year, available)
                    self.assertEqual(
                        state.visas_consumed_this_year,
                        state.converted_temps + state.converted_spouses + state.children_saved_this_year,
                    )
                    total += state.visas_consumed_this_year
                    self.assertEqual(state.cumulative_visas_consumed, total)
            self.assertGreater(total, 0)

    def test_02_subgroups_sum_to_totals(self):
        for scenario, record in self.pair.items():
            for state in record["states"]:
                with self.subTest(scenario=scenario, year=state.year):
                    for values, total in (
                        (state.queue_backlog_by_country, state.temporary_workers),
                        (state.queue_backlog_by_eb_category, state.temporary_workers),
                        (state.queue_backlog_by_eb_category_nationality, state.temporary_workers),
                        (state.converted_by_country, state.converted_temps),
                        (state.converted_by_eb_category, state.converted_temps),
                        (state.converted_by_eb_category_total, state.visas_consumed_this_year),
                        (state.visas_consumed_by_category_nationality, state.visas_consumed_this_year),
                        (state.exited_by_nationality, state.exited_this_year),
                        (state.exited_by_eb_category, state.exited_this_year),
                        (state.exited_by_category_nationality, state.exited_this_year),
                        (state.aged_out_by_nationality, state.cumulative_children_aged_out),
                        (state.aged_out_by_eb_category, state.cumulative_children_aged_out),
                    ):
                        self.assertTrue(all(value >= 0 for value in values.values()))
                        self.assertEqual(sum(values.values()), total)
                    for category in EBCategory:
                        self.assertEqual(
                            sum(value for (cat, _), value in
                                state.queue_backlog_by_eb_category_nationality.items() if cat == category),
                            state.queue_backlog_by_eb_category.get(category, 0),
                        )
                        self.assertEqual(
                            sum(value for (cat, _), value in
                                state.visas_consumed_by_category_nationality.items() if cat == category),
                            state.converted_by_eb_category_total.get(category, 0),
                        )
                    for country in sim.COUNTRIES:
                        self.assertEqual(
                            sum(value for (_, nat), value in
                                state.queue_backlog_by_eb_category_nationality.items() if nat == country),
                            state.queue_backlog_by_country.get(country, 0),
                        )

    def test_03_children_are_conserved_across_outcomes(self):
        for scenario, record in self.pair.items():
            for state, snapshot in zip(record["states"], record["children"]):
                with self.subTest(scenario=scenario, year=state.year):
                    ids = [child for group in snapshot["ids"].values() for child in group]
                    created = sum(snapshot["created_by_nationality"].values())
                    self.assertEqual(len(ids), created)
                    self.assertEqual(len(set(ids)), created)
                    self.assertEqual(
                        Counter(snapshot["observed_by_nationality"]),
                        Counter(snapshot["created_by_nationality"]),
                    )
                    self.assertEqual(len(snapshot["ids"]["waiting"]), state.children_at_risk)
                    self.assertEqual(len(snapshot["ids"]["aged_out"]), state.cumulative_children_aged_out)
            # Each outcome must actually occur in the fixture.
            for outcome, ids in record["children"][-1]["ids"].items():
                with self.subTest(scenario=scenario, outcome=outcome):
                    self.assertGreater(len(ids), 0)

    def test_04_historical_reconstruction_is_identical(self):
        uncapped, capped = self.pair["Uncapped"], self.pair["Capped"]
        for index, (left, right) in enumerate(zip(uncapped["states"], capped["states"])):
            if left.year >= POLICY_YEAR:
                continue
            with self.subTest(year=left.year):
                a, b = asdict(left), asdict(right)
                # This flag labels the chosen projection policy, even before reform.
                a.pop("country_cap_enabled")
                b.pop("country_cap_enabled")
                self.assertEqual(a, b)
                self.assertEqual(uncapped["children"][index], capped["children"][index])
        self.assertTrue(any(
            left.queue_backlog_by_eb_category_nationality != right.queue_backlog_by_eb_category_nationality
            for left, right in zip(uncapped["states"], capped["states"])
            if left.year >= POLICY_YEAR
        ), "The fixture must also exercise the policy change after reconstruction.")

    def test_05_fixed_seed_reproduces_results(self):
        # A second engine run exercises the RNGs again, rather than reloading results.
        repeated = run_pair()
        for scenario in ("Uncapped", "Capped"):
            with self.subTest(scenario=scenario):
                self.assertEqual(self.pair[scenario], repeated[scenario])


if __name__ == "__main__":
    unittest.main(verbosity=2)
