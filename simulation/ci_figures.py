"""
CI-backed policy figures for the aged-out paper.
Styling is identical to visualization.py.

Output: outputs/ci/policy_figures/
  annual_aged_out/   overall.png, india.png, china.png, other.png
  cumulative/        overall.png, india.png, china.png, other.png
  totals/            overall.png, india.png, china.png, other.png
  differential/      overall.png, india.png, china.png, other.png
  backlog/           overall.png, india.png, china.png, other.png
  eb_category/       eb1.png, eb2.png, eb3.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CI_RAW_TS = PROJECT_ROOT / "outputs" / "ci" / "ci_raw_timeseries.csv"
CI_RAW_SC = PROJECT_ROOT / "outputs" / "ci" / "ci_raw_scalars.csv"
SEG_CSV = PROJECT_ROOT / "outputs" / "cohorts" / "children_aged_out_segmentation.csv"
BASE_OUT = PROJECT_ROOT / "outputs" / "ci" / "policy_figures"

# Global style -- mirrors visualization.py exactly
PLOT_DPI = 400

sns.set_theme(style="white", context="paper")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = PLOT_DPI
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Colour palettes -- from visualization.py
# Okabe-Ito colour-blind-safe hues, matched to simulation/visualization.py.
# Scenario bars also carry a hatch (SCENARIO_HATCH) so the two scenarios stay
# distinguishable in greyscale/print, not by colour alone.
SCENARIO_COLORS = {"Uncapped": "#56B4E9", "Capped": "#D55E00"}
SCENARIO_HATCH = {"Uncapped": "", "Capped": "///"}
EB_COLORS = {
    "EB-1": "#0072B2",
    "EB-2": "#E69F00",
    "EB-3": "#009E73",
    "EB-4": "#CC79A7",
    "EB-5": "#D55E00",
}
DIFF_COLOR = "#CC79A7"

NATS = ["India", "China", "Other"]
POLICY_YEAR = 2024

BAR_WIDTH = 0.35
CAPSIZE = 4
ERR_KW = {"ecolor": "#444444", "lw": 1.0, "capthick": 1.0}

# Helpers


def _mkdir(sub: str) -> Path:
    d = BASE_OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ci_ts(grouped, col):
    return (
        grouped[col].mean().values,
        grouped[col].quantile(0.025).values,
        grouped[col].quantile(0.975).values,
    )


def _ci_sc(series: pd.Series):
    return (
        float(series.mean()),
        float(series.quantile(0.025)),
        float(series.quantile(0.975)),
    )


def _yerr(m, lo, hi):
    return np.array([np.asarray(m) - np.asarray(lo), np.asarray(hi) - np.asarray(m)])


def _fmt(v: float) -> str:
    av = abs(v)
    if av >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if av >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.0f}"


def _add_shift_annotation(ax, x_val) -> None:
    xlim = ax.get_xlim()
    if not (xlim[0] <= x_val <= xlim[1]):
        return
    ax.axvline(x_val, color="#555555", linewidth=1.5, linestyle="--", zorder=5, alpha=0.7)
    ylim = ax.get_ylim()
    x_off = (xlim[1] - xlim[0]) * 0.012
    y_text = ylim[0] + (ylim[1] - ylim[0]) * 0.96
    ax.text(
        x_val + x_off,
        y_text,
        "Policy shift",
        fontsize=11,
        color="#444444",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", linewidth=1.0, alpha=0.88),
        zorder=6,
    )


def _style_ts(ax, years, ylabel: str, title: str) -> None:
    n = len(years)
    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [str(list(years)[i]) for i in ticks],
        rotation=45,
        ha="right",
        fontsize=12,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, weight="medium", labelpad=10)
    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=True, framealpha=1.0, edgecolor="#DADADA", loc="upper left", fontsize=12)
    legend.get_frame().set_linewidth(1.5)


def _save(fig, folder: Path, filename: str) -> None:
    p = folder / filename
    fig.savefig(p, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK]  {p.relative_to(PROJECT_ROOT)}")

# Load data
print("Loading data ...")
df = pd.read_csv(CI_RAW_TS)
sc = pd.read_csv(CI_RAW_SC)
seg = pd.read_csv(SEG_CSV)

by_year = df.groupby("year")
years = sorted(df["year"].unique())
x = np.arange(len(years))
yr_min = int(min(years))
yr_max = int(max(years))

# 1. Annual aged-out  --  one file per group
print("\n[1] Annual aged-out ...")
d = _mkdir("annual_aged_out")

annual_groups = [
    ("overall", "annual_aged_out_unc", "annual_aged_out_cap", "Children Aged Out Per Year: Policy Comparison"),
    ("india", "annual_aged_out_India_unc", "annual_aged_out_India_cap", "Children Aged Out Per Year -- India: Policy Comparison"),
    ("china", "annual_aged_out_China_unc", "annual_aged_out_China_cap", "Children Aged Out Per Year -- China: Policy Comparison"),
    ("other", "annual_aged_out_Other_unc", "annual_aged_out_Other_cap", "Children Aged Out Per Year -- Other Nationalities: Policy Comparison"),
]

# Clip the annual aged-out charts (incl. Figure 5, overall.png) to the post-policy
# window FY2025-2040: capped and uncapped are identical before FY2025, so pre-2025
# bars carry no policy signal. Other sections keep the full FY2009-2040 range.
ANNUAL_START = 2025
_annual_keep = np.array([yr >= ANNUAL_START for yr in years])
annual_years = [yr for yr in years if yr >= ANNUAL_START]
annual_x = np.arange(len(annual_years))

for fname, uc_col, cc_col, title in annual_groups:
    unc_m, unc_lo, unc_hi = _ci_ts(by_year, uc_col)
    cap_m, cap_lo, cap_hi = _ci_ts(by_year, cc_col)
    unc_m, unc_lo, unc_hi = unc_m[_annual_keep], unc_lo[_annual_keep], unc_hi[_annual_keep]
    cap_m, cap_lo, cap_hi = cap_m[_annual_keep], cap_lo[_annual_keep], cap_hi[_annual_keep]

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(
        annual_x - BAR_WIDTH / 2,
        unc_m,
        BAR_WIDTH,
        label="No Per-Country Cap",
        color=SCENARIO_COLORS["Uncapped"],
        hatch=SCENARIO_HATCH["Uncapped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(unc_m, unc_lo, unc_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )
    ax.bar(
        annual_x + BAR_WIDTH / 2,
        cap_m,
        BAR_WIDTH,
        label="7% Per-Country Cap",
        color=SCENARIO_COLORS["Capped"],
        hatch=SCENARIO_HATCH["Capped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(cap_m, cap_lo, cap_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )

    _style_ts(ax, annual_years, "Children Aged Out", f"{title}\nMean +/- 95% CI  (50 paired Monte Carlo runs)")

    if POLICY_YEAR in annual_years:
        _add_shift_annotation(ax, list(annual_years).index(POLICY_YEAR) + 0.5)

    fig.tight_layout()
    # The overall annual chart is Figure 5 in the paper. Also emit it under the
    # paper-facing filename so the manuscript figure regenerates directly with
    # the same Okabe-Ito palette as every other policy figure (no stale copies).
    if fname == "overall":
        fig.savefig(
            d / "annual_age_outs_by_scenario.png",
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    _save(fig, d, f"{fname}.png")

# 2. Cumulative aged-out  --  one file per group
print("\n[2] Cumulative aged-out ...")
d = _mkdir("cumulative")

unc_m, unc_lo, unc_hi = _ci_ts(by_year, "cumul_aged_out_unc")
cap_m, cap_lo, cap_hi = _ci_ts(by_year, "cumul_aged_out_cap")

fig, ax = plt.subplots(figsize=(6.5, 3.7))
ax.bar(
    x - BAR_WIDTH / 2,
    unc_m,
    BAR_WIDTH,
    label="No Per-Country Cap",
    color=SCENARIO_COLORS["Uncapped"],
    hatch=SCENARIO_HATCH["Uncapped"],
    edgecolor="white",
    linewidth=1.5,
    alpha=0.9,
    yerr=_yerr(unc_m, unc_lo, unc_hi),
    capsize=CAPSIZE,
    error_kw=ERR_KW,
    zorder=3,
)
ax.bar(
    x + BAR_WIDTH / 2,
    cap_m,
    BAR_WIDTH,
    label="7% Per-Country Cap",
    color=SCENARIO_COLORS["Capped"],
    hatch=SCENARIO_HATCH["Capped"],
    edgecolor="white",
    linewidth=1.5,
    alpha=0.9,
    yerr=_yerr(cap_m, cap_lo, cap_hi),
    capsize=CAPSIZE,
    error_kw=ERR_KW,
    zorder=3,
)
_style_ts(ax, years, "Cumulative Children Aged Out", "Cumulative Children Aged Out: Policy Comparison\n" "Mean +/- 95% CI  (50 paired MC runs)  --  Scenarios identical <= 2024")
if POLICY_YEAR in years:
    _add_shift_annotation(ax, list(years).index(POLICY_YEAR) + 0.5)
fig.tight_layout()
_save(fig, d, "overall.png")

for nat in NATS:
    runs_unc, runs_cap = [], []
    for _, grp in df.groupby("run_index"):
        s = grp.sort_values("year")
        runs_unc.append(s[f"annual_aged_out_{nat}_unc"].cumsum().values)
        runs_cap.append(s[f"annual_aged_out_{nat}_cap"].cumsum().values)
    mat_u = np.array(runs_unc)
    mat_c = np.array(runs_cap)
    unc_m = mat_u.mean(0)
    unc_lo = np.percentile(mat_u, 2.5, 0)
    unc_hi = np.percentile(mat_u, 97.5, 0)
    cap_m = mat_c.mean(0)
    cap_lo = np.percentile(mat_c, 2.5, 0)
    cap_hi = np.percentile(mat_c, 97.5, 0)

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(
        x - BAR_WIDTH / 2,
        unc_m,
        BAR_WIDTH,
        label="No Per-Country Cap",
        color=SCENARIO_COLORS["Uncapped"],
        hatch=SCENARIO_HATCH["Uncapped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(unc_m, unc_lo, unc_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )
    ax.bar(
        x + BAR_WIDTH / 2,
        cap_m,
        BAR_WIDTH,
        label="7% Per-Country Cap",
        color=SCENARIO_COLORS["Capped"],
        hatch=SCENARIO_HATCH["Capped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(cap_m, cap_lo, cap_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )
    _style_ts(ax, years, "Cumulative Children Aged Out", f"Cumulative Children Aged Out -- {nat}: Policy Comparison\n" f"Mean +/- 95% CI  (50 paired MC runs, {yr_min}-{yr_max})")
    if POLICY_YEAR in years:
        _add_shift_annotation(ax, list(years).index(POLICY_YEAR) + 0.5)
    fig.tight_layout()
    _save(fig, d, f"{nat.lower()}.png")

# 3. Totals (headline)  --  one file per group
print("\n[3] Totals (headline) ...")
d = _mkdir("totals")


def _total_chart(unc_col, cap_col, group_label, fname, ylabel, title_prefix):
    m_u, lo_u, hi_u = _ci_sc(sc[unc_col])
    m_c, lo_c, hi_c = _ci_sc(sc[cap_col])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.bar(
        0,
        m_u,
        BAR_WIDTH * 2.2,
        label="No Per-Country Cap",
        color=SCENARIO_COLORS["Uncapped"],
        hatch=SCENARIO_HATCH["Uncapped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        yerr=[[m_u - lo_u], [hi_u - m_u]],
        capsize=6,
        error_kw=ERR_KW,
    )
    ax.bar(
        1,
        m_c,
        BAR_WIDTH * 2.2,
        label="7% Per-Country Cap",
        color=SCENARIO_COLORS["Capped"],
        hatch=SCENARIO_HATCH["Capped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        yerr=[[m_c - lo_c], [hi_c - m_c]],
        capsize=6,
        error_kw=ERR_KW,
    )

    ymax = max(hi_u, hi_c) * 1.42
    ax.set_ylim(0, ymax)

    for xi, (mean, lo, hi) in enumerate([(m_u, lo_u, hi_u), (m_c, lo_c, hi_c)]):
        ax.text(xi, hi + ymax * 0.025, _fmt(mean), ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(xi, hi + ymax * 0.10, f"[{_fmt(lo)}, {_fmt(hi)}]", ha="center", va="bottom", fontsize=10, color="#444444")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Per-Country Cap", "7% Per-Country Cap"], fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylabel(ylabel, fontsize=13, weight="medium", labelpad=10)
    ax.set_title(
        f"{title_prefix} ({yr_min}-{yr_max}): {group_label}\n" "Mean +/- 95% CI  (50 paired Monte Carlo runs)",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    legend = ax.legend(frameon=True, framealpha=1.0, edgecolor="#DADADA", loc="upper right", fontsize=12)
    legend.get_frame().set_linewidth(1.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, d, fname)

_total_chart(
    "final_cumul_aged_out_unc",
    "final_cumul_aged_out_cap",
    "Overall",
    "overall.png",
    "Total Children Aged Out",
    "Total Children Aged Out",
)
for nat in NATS:
    _total_chart(
        f"final_aged_out_{nat}_unc",
        f"final_aged_out_{nat}_cap",
        nat,
        f"{nat.lower()}.png",
        "Total Children Aged Out",
        "Total Children Aged Out",
    )

# 4. Differential  --  one file per group
print("\n[4] Differential ...")
d = _mkdir("differential")


def _diff_chart(diff_col, group_label, fname, ylabel, title_prefix, mean_override: float = None, lo_override: float = None, hi_override: float = None):
    if mean_override is not None:
        m, lo, hi = mean_override, lo_override, hi_override
    else:
        m, lo, hi = _ci_sc(sc[diff_col])

    outer_tip = lo if m <= 0 else hi
    data_abs = abs(outer_tip)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.bar(0, m, BAR_WIDTH * 2.2, color=DIFF_COLOR, edgecolor="white", linewidth=1.5, alpha=0.9, zorder=3, yerr=[[m - lo], [hi - m]], capsize=6, error_kw=ERR_KW)
    ax.axhline(0, color="#333333", lw=0.8)

    if m <= 0:
        ymin = outer_tip * 1.30
        ymax = data_abs * 0.15
    else:
        ymin = -data_abs * 0.15
        ymax = outer_tip * 1.30
    ax.set_ylim(ymin, ymax)

    tr = ax.get_xaxis_transform()

    if m <= 0:
        mean_y_ax = 0.13
        ci_y_ax = 0.06
    else:
        mean_y_ax = 0.87
        ci_y_ax = 0.94

    ax.text(0, mean_y_ax, f"{m:+,.0f}", transform=tr, ha="center", va="center", fontsize=14, fontweight="bold", color=DIFF_COLOR)
    ax.text(0, ci_y_ax, f"95% CI: [{lo:,.0f}, {hi:,.0f}]", transform=tr, ha="center", va="center", fontsize=11, color="#444444")

    ax.set_xticks([])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}"))
    ax.set_ylabel(ylabel, fontsize=13, weight="medium", labelpad=10)
    ax.set_title(
        f"{title_prefix}: {group_label}\n" "Mean +/- 95% CI  (50 paired Monte Carlo runs)",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, d, fname)

_diff_chart(
    "diff_cumul_aged_out",
    "Overall",
    "overall.png",
    "Change in Children Aged Out\n(7% Per-Country Cap - No Cap)",
    "Cap-Attributable Change in Aged-Out Children",
)
for nat in NATS:
    _diff_chart(
        f"diff_aged_out_{nat}",
        nat,
        f"{nat.lower()}.png",
        "Change in Children Aged Out\n(7% Per-Country Cap - No Cap)",
        "Cap-Attributable Change in Aged-Out Children",
    )

# 5. Final-year queue backlog  --  one file per group
print("\n[5] Final-year backlog ...")
d = _mkdir("backlog")


def _backlog_chart(unc_col, cap_col, group_label, fname):
    m_u, lo_u, hi_u = _ci_sc(sc[unc_col])
    m_c, lo_c, hi_c = _ci_sc(sc[cap_col])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.bar(
        0,
        m_u,
        BAR_WIDTH * 2.2,
        label="No Per-Country Cap",
        color=SCENARIO_COLORS["Uncapped"],
        hatch=SCENARIO_HATCH["Uncapped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        yerr=[[m_u - lo_u], [hi_u - m_u]],
        capsize=6,
        error_kw=ERR_KW,
    )
    ax.bar(
        1,
        m_c,
        BAR_WIDTH * 2.2,
        label="7% Per-Country Cap",
        color=SCENARIO_COLORS["Capped"],
        hatch=SCENARIO_HATCH["Capped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        yerr=[[m_c - lo_c], [hi_c - m_c]],
        capsize=6,
        error_kw=ERR_KW,
    )

    ymax = max(hi_u, hi_c) * 1.42
    ax.set_ylim(0, ymax)

    for xi, (mean, lo, hi) in enumerate([(m_u, lo_u, hi_u), (m_c, lo_c, hi_c)]):
        ax.text(xi, hi + ymax * 0.025, _fmt(mean), ha="center", va="bottom", fontsize=13, fontweight="bold")
        ax.text(xi, hi + ymax * 0.10, f"[{_fmt(lo)}, {_fmt(hi)}]", ha="center", va="bottom", fontsize=10, color="#444444")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Per-Country Cap", "7% Per-Country Cap"], fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylabel("Queue Backlog (Principals Only)", fontsize=13, weight="medium", labelpad=10)
    ax.set_title(
        f"Final-Year Queue Backlog ({yr_max}): {group_label}\n" "Mean +/- 95% CI  (50 paired Monte Carlo runs)",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    legend = ax.legend(frameon=True, framealpha=1.0, edgecolor="#DADADA", loc="upper right", fontsize=12)
    legend.get_frame().set_linewidth(1.5)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, d, fname)

_backlog_chart("final_backlog_total_unc", "final_backlog_total_cap", "Overall", "overall.png")
for nat in NATS:
    _backlog_chart(f"final_backlog_{nat}_unc", f"final_backlog_{nat}_cap", nat, f"{nat.lower()}.png")

# 6. EB category breakdown  --  one file per category
print("\n[6] EB category ...")
d = _mkdir("eb_category")

seg_eb = seg[(seg["Nationality"] == "Overall") & (seg["EB_Category"] != "Overall")].copy()
eb_cats = sorted(seg_eb["EB_Category"].unique())

for cat in eb_cats:
    col_key = cat.replace("-", "")  # "EB-1" -> "EB1"
    unc_col = f"annual_aged_out_{col_key}_unc"
    cap_col = f"annual_aged_out_{col_key}_cap"

    unc_m, unc_lo, unc_hi = _ci_ts(by_year, unc_col)
    cap_m, cap_lo, cap_hi = _ci_ts(by_year, cap_col)

    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(
        x - BAR_WIDTH / 2,
        unc_m,
        BAR_WIDTH,
        label="No Per-Country Cap",
        color=SCENARIO_COLORS["Uncapped"],
        hatch=SCENARIO_HATCH["Uncapped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(unc_m, unc_lo, unc_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )
    ax.bar(
        x + BAR_WIDTH / 2,
        cap_m,
        BAR_WIDTH,
        label="7% Per-Country Cap",
        color=SCENARIO_COLORS["Capped"],
        hatch=SCENARIO_HATCH["Capped"],
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
        yerr=_yerr(cap_m, cap_lo, cap_hi),
        capsize=CAPSIZE,
        error_kw=ERR_KW,
        zorder=3,
    )

    _style_ts(ax, years, "Children Aged Out (Annual)", f"Children Aged Out Per Year -- {cat}: Policy Comparison\n" f"Mean +/- 95% CI  (50 paired Monte Carlo runs)")

    if POLICY_YEAR in years:
        _add_shift_annotation(ax, list(years).index(POLICY_YEAR) + 0.5)

    fig.tight_layout()
    _save(fig, d, f"{cat.replace('-', '').lower()}.png")

print(
    f"""
[OK]  All figures saved -> outputs/ci/policy_figures/
   annual_aged_out/      {len(annual_groups)} files
   cumulative/           {1 + len(NATS)} files
   totals/               {1 + len(NATS)} files
   differential/         {1 + len(NATS)} files
   backlog/              {1 + len(NATS)} files
   eb_category/          {len(eb_cats)} files
"""
)
