"""Explicit adapters and accounting checks for simulation result tables."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_nationality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Read legacy Other columns as ROW, rejecting contradictory duplicates."""
    result = frame.copy()
    for old in list(result.columns):
        if "_Other_" not in old:
            continue
        new = old.replace("_Other_", "_ROW_")
        if new in result:
            if not np.array_equal(result[old].to_numpy(), result[new].to_numpy(), equal_nan=True):
                raise ValueError(f"Conflicting nationality columns: {old}, {new}")
            result = result.drop(columns=old)
        else:
            result = result.rename(columns={old: new})
    return result


def check_annual_exit_partitions(frame: pd.DataFrame) -> None:
    """Fail instead of silently summarizing old first-differenced exit exports."""
    for scenario in ("unc", "cap"):
        total = f"annual_exited_{scenario}"
        if total not in frame:
            continue
        for groups in (("India", "China", "ROW"), ("EB1", "EB2", "EB3", "EB4", "EB5")):
            columns = [f"annual_exited_{group}_{scenario}" for group in groups]
            if not all(column in frame for column in columns):
                raise ValueError(f"Missing exit partition columns: {columns}")
            if (frame[columns] < 0).any().any():
                raise ValueError("Negative annual subgroup exits: legacy export requires an explicit audited migration.")
            if not np.allclose(frame[columns].sum(axis=1), frame[total], atol=0, rtol=0):
                raise ValueError(f"Annual exits do not reconcile for {scenario}: {groups}")
