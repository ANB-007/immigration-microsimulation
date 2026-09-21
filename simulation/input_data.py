"""Load the model's historical petition inputs from the repository data folder."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _year_keys(values):
    """JSON object keys are strings; restore the integer fiscal-year keys."""
    return {int(key) if key.isdecimal() else key: value for key, value in values.items()}


def _petitions(name):
    with (DATA_DIR / "petitions" / f"{name}.json").open() as handle:
        return json.load(handle, object_hook=_year_keys)


_eb123 = _petitions("eb123")
I140_ANNUAL_APPROVALS = _eb123["I140_ANNUAL_APPROVALS"]
COUNTRY_APPROVALS = _eb123["COUNTRY_APPROVALS"]

_eb4 = _petitions("eb4")
EB4_ANNUAL_TOTALS_BY_YEAR = _eb4["EB4_ANNUAL_TOTALS_BY_YEAR"]
SIJS_NATIONALITY_DISTRIBUTION = _eb4["SIJS_NATIONALITY_DISTRIBUTION"]
OTHER_EB4_NATIONALITY_DISTRIBUTION_BY_YEAR = _eb4["OTHER_EB4_NATIONALITY_DISTRIBUTION_BY_YEAR"]

_eb5 = _petitions("eb5")
EB5_HISTORICAL = _eb5["EB5_HISTORICAL"]
EB5_NATIONALITY_DISTRIBUTION_BY_YEAR = _eb5["EB5_NATIONALITY_DISTRIBUTION_BY_YEAR"]
