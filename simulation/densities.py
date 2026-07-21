"""
100% stacked area charts of cohort Child Outcome Proportions by entry year.
Styling mirrors visualization.py exactly.

Reads:  outputs/cohorts/outcomes_by_entry_year.csv
Output: outputs/cohorts/entry_year_density/
  overall/                  uncapped.png, capped.png
  by_nationality/           <nat>_uncapped.png, <nat>_capped.png
  by_eb_category/           <eb>_uncapped.png, <eb>_capped.png
  by_category_nationality/  <eb>_<nat>_uncapped.png, <eb>_<nat>_capped.png
  capped_period_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "outputs" / "cohorts" / "outcomes_by_entry_year.csv"
BASE_OUT = PROJECT_ROOT / "outputs" / "cohorts" / "entry_year_density"

# Global style -- mirrors visualization.py exactly
PLOT_DPI = 400
END_YEAR = 2040

sns.set_theme(style="white", context="paper")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = PLOT_DPI
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Outcome stack palette  (bottom -> top: Saved, Aged_Out, Exited, Still_Dependent)
#
# Good   -- Saved:           #2E7D32  forest green
#                           (distinct from teal #00897B and emerald #059669)
# Bad    -- Aged_Out:        #BF360C  burnt orange-red
#                           (distinct from crimson #C62828 and orange #F57C00)
# Bad    -- Exited:          #4E342E  dark brownish-red
#                           (unique region; nothing brown in visualization.py)
# Neutral-- Still_Dependent: #546E7A  slate blue-grey
#                           (darker/bluer than #90A4AE; distinct from #616161)
STACK_ORDER = ["Saved", "Aged_Out", "Exited", "Still_Dependent"]

OUTCOME_COLORS = {
    "Saved": "#009E73",           # bluish green (good outcome)
    "Aged_Out": "#D55E00",        # vermillion (adverse outcome)
    "Exited": "#E69F00",          # orange
    "Still_Dependent": "#0072B2", # blue (neutral / pending)
}

# Each stacked layer also carries a hatch pattern so the four outcomes stay
# distinguishable in greyscale/print, not by colour alone.
OUTCOME_HATCH = {
    "Saved": "",
    "Aged_Out": "///",
    "Exited": "..",
    "Still_Dependent": "xx",
}

OUTCOME_LABELS = {
    "Saved": "Saved",
    "Aged_Out": "Aged Out",
    "Exited": "Exited Queue",
    "Still_Dependent": "Still Dependent",
}

SCENARIO_DISPLAY = {
    "Uncapped": "Uncapped",
    "Capped": "Capped",
}

NATIONALITY_DISPLAY = {
    "Other": "ROW",
}

# Period windows for summary tables
PERIOD_ALL  = (2009, 2040)
PERIOD_LATE = (2025, 2040)

# Helpers


def _mkdir(sub: str) -> Path:
    d = BASE_OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, folder: Path, filename: str) -> None:
    p = folder / filename
    fig.savefig(p, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK]  {p.relative_to(PROJECT_ROOT)}")


def _build_proportions(df_group: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by Entry_Year, compute proportion of each outcome (0-100).
    Returns a DataFrame indexed by Entry_Year with columns matching STACK_ORDER,
    clipped to END_YEAR. Entry years with Total_Created == 0 are dropped
    entirely (not filled as 0) so they never cause spurious zero-drops.
    """
    agg = df_group.groupby("Entry_Year")[["Total_Created", "Aged_Out", "Saved", "Exited", "Still_Dependent"]].sum()
    agg = agg[agg["Total_Created"] > 0]
    if agg.empty:
        return pd.DataFrame(columns=STACK_ORDER)
    total = agg["Total_Created"]
    props = (
        pd.DataFrame(
            {
                "Saved":           agg["Saved"]           / total * 100,
                "Aged_Out":        agg["Aged_Out"]        / total * 100,
                "Exited":          agg["Exited"]          / total * 100,
                "Still_Dependent": agg["Still_Dependent"] / total * 100,
            }
        )
        .sort_index()
    )
    return props[props.index <= END_YEAR]


def _plot_stacked(
    props: pd.DataFrame,
    title: str,
    subtitle: str,
    folder: Path,
    filename: str,
) -> None:
    """
    Draw a 100% stacked area chart of cohort Child Outcome Proportions over entry year.
    Only layers with non-zero data are drawn.
    """
    active = [c for c in STACK_ORDER if c in props.columns and props[c].sum() > 0]
    if not active:
        return

    years = props.index.values

    fig, ax = plt.subplots(figsize=(6.5, 3.7))

    ax.stackplot(
        years,
        [props[c].values for c in active],
        labels=[OUTCOME_LABELS[c] for c in active],
        colors=[OUTCOME_COLORS[c] for c in active],
        alpha=0.85,
        linewidth=0,
    )
    for _poly, _c in zip(ax.collections, active):
        _poly.set_hatch(OUTCOME_HATCH.get(_c, ""))

    ax.set_xlim(int(years[0]), END_YEAR)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))

    full_title = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full_title, fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Entry Year", fontsize=13, weight="medium", labelpad=10)
    ax.set_ylabel("Proportion of Cohort", fontsize=13, weight="medium", labelpad=10)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=True,
        framealpha=1.0,
        edgecolor="#DADADA",
        loc="upper right",
        fontsize=12,
    )
    legend.get_frame().set_linewidth(1.5)

    fig.tight_layout()
    _save(fig, folder, filename)


def _period_avg(df_cap: pd.DataFrame, nat: str, eb: str, start: int, end: int):
    """
    Return avg_save_pct for the given nationality / EB category
    and entry-year window, averaged across individual cohort rows.
    """
    sub = df_cap[
        (df_cap["Nationality"] == nat) &
        (df_cap["EB_Category"] == eb) &
        (df_cap["Entry_Year"] >= start) &
        (df_cap["Entry_Year"] <= end)
    ]
    if sub.empty:
        return float("nan")
    return sub["Save_Pct"].mean()


def _fmt_pct(v: float) -> str:
    return f"{v:.1f}%" if not np.isnan(v) else "N/A"

# Load data
print("Loading data ...")
df = pd.read_csv(INPUT_CSV)

scenarios     = sorted(df["Scenario"].unique())
nationalities = sorted(df["Nationality"].unique())
eb_categories = sorted(df["EB_Category"].unique())

# 1. Overall
print("\n[1] Overall ...")
d = _mkdir("overall")

for scenario in scenarios:
    sub = df[df["Scenario"] == scenario]
    props = _build_proportions(sub)
    _plot_stacked(
        props,
        title=f"Child Outcome Proportions by Entry Cohort ({SCENARIO_DISPLAY.get(scenario, scenario)})",
        subtitle="All Nationalities - All EB Categories",
        folder=d,
        filename=f"{scenario.lower()}.png",
    )

# 2. By Nationality
print("\n[2] By Nationality ...")
d = _mkdir("by_nationality")

for nat in nationalities:
    nat_display = NATIONALITY_DISPLAY.get(nat, nat)
    for scenario in scenarios:
        sub = df[(df["Scenario"] == scenario) & (df["Nationality"] == nat)]
        if sub.empty:
            continue
        props = _build_proportions(sub)
        _plot_stacked(
            props,
            title=f"Child Outcome Proportions by Entry Cohort ({SCENARIO_DISPLAY.get(scenario, scenario)})",
            subtitle=f"Nationality: {nat_display} - All EB Categories",
            folder=d,
            filename=f"{nat.lower()}_{scenario.lower()}.png",
        )

# 3. By EB Category
print("\n[3] By EB Category ...")
d = _mkdir("by_eb_category")

for eb in eb_categories:
    eb_key = eb.replace("-", "").lower()  # "eb1", "eb2", ...
    for scenario in scenarios:
        sub = df[(df["Scenario"] == scenario) & (df["EB_Category"] == eb)]
        if sub.empty:
            continue
        props = _build_proportions(sub)
        _plot_stacked(
            props,
            title=f"Child Outcome Proportions by Entry Cohort ({SCENARIO_DISPLAY.get(scenario, scenario)})",
            subtitle=f"EB Category: {eb} - All Nationalities",
            folder=d,
            filename=f"{eb_key}_{scenario.lower()}.png",
        )

# 4. By EB Category x Nationality
print("\n[4] By EB Category x Nationality ...")
d = _mkdir("by_category_nationality")

for eb in eb_categories:
    eb_key = eb.replace("-", "").lower()
    for nat in nationalities:
        nat_display = NATIONALITY_DISPLAY.get(nat, nat)
        for scenario in scenarios:
            sub = df[
                (df["Scenario"] == scenario) &
                (df["EB_Category"] == eb) &
                (df["Nationality"] == nat)
            ]
            if sub.empty:
                continue
            props = _build_proportions(sub)
            _plot_stacked(
                props,
                title=f"Child Outcome Proportions by Entry Cohort ({SCENARIO_DISPLAY.get(scenario, scenario)})",
                subtitle=f"EB Category: {eb} - Nationality: {nat_display}",
                folder=d,
                filename=f"{eb_key}_{nat.lower()}_{scenario.lower()}.png",
            )

# 5. Capped period-average save rates (by nationality)
print("\n[5] Capped period-average save rates ...")

df_cap = df[df["Scenario"] == "Capped"]
summary_rows = []

A0, A1 = PERIOD_ALL
L0, L1 = PERIOD_LATE

COL_W = 14
VAL_W = 26

header = (
    f"{'EB Category':<{COL_W}}"
    f"{'Avg. Saved (2009-2040)':>{VAL_W}}"
    f"{'Avg. Saved (2025-2040)':>{VAL_W}}"
)
divider = "-" * len(header)

for nat in nationalities:
    nat_display = NATIONALITY_DISPLAY.get(nat, nat)
    print(f"\n-- {nat_display} (Capped) {'-' * max(0, 62 - len(nat_display))}")
    print(header)
    print(divider)
    for eb in eb_categories:
        s_all  = _period_avg(df_cap, nat, eb, A0, A1)
        s_late = _period_avg(df_cap, nat, eb, L0, L1)
        print(
            f"{eb:<{COL_W}}"
            f"{_fmt_pct(s_all):>{VAL_W}}"
            f"{_fmt_pct(s_late):>{VAL_W}}"
        )
        summary_rows.append({
            "Nationality":          nat,
            "EB_Category":          eb,
            "Avg_Saved_2009_2040": round(s_all, 4)  if not np.isnan(s_all)  else None,
            "Avg_Saved_2025_2040": round(s_late, 4) if not np.isnan(s_late) else None,
        })

summary_csv = BASE_OUT / "capped_period_summary.csv"
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
print(f"\n  [OK]  capped_period_summary.csv -> {summary_csv.relative_to(PROJECT_ROOT)}")

n_overall = len(scenarios)
n_nat     = len(nationalities) * len(scenarios)
n_eb      = len(eb_categories) * len(scenarios)
n_pairs   = len(eb_categories) * len(nationalities) * len(scenarios)

print(
    f"""
[OK]  All figures saved -> outputs/cohorts/entry_year_density/
   overall/                  {n_overall} files
   by_nationality/           {n_nat} files
   by_eb_category/           {n_eb} files
   by_category_nationality/  {n_pairs} files
   capped_period_summary.csv
"""
)
