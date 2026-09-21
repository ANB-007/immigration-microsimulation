"""Shared manuscript styling, following the original color PNG figures.

Keep palette meanings aligned with simulation.visualization and densities
without importing those modules and their plotting side effects.
"""
from pathlib import Path

PLOT_DPI = 400
FONT_SIZE = 10.2
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SCENARIO_COLORS = {"Uncapped": "#56B4E9", "Capped": VERMILLION}
NATIONALITY_COLORS = {"India": BLUE, "China": ORANGE, "ROW": GREEN}
APPLICANT_COLORS = {"principals": BLUE, "spouses": ORANGE, "children": GREEN}
OUTCOME_COLORS = {"saved": GREEN, "aged_out": VERMILLION,
                  "exited": ORANGE, "still_dependent": BLUE}


def manuscript_style():
    """Return one font, palette, and print-size style for all manuscript PNGs."""
    from matplotlib import font_manager
    from cycler import cycler

    for name in ("Arial.ttf", "Arial Bold.ttf"):
        path = Path("/System/Library/Fonts/Supplemental") / name
        if path.exists():
            font_manager.fontManager.addfont(path)
    family = "DejaVu Sans"
    for candidate in ("Arial", "Helvetica"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
            family = candidate
            break
        except ValueError:
            pass
    return {
        "font.family": family, "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
        "axes.titleweight": "bold", "axes.labelweight": "medium",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": .8,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "savefig.edgecolor": "none",
        "figure.dpi": 100, "savefig.dpi": PLOT_DPI,
        "axes.prop_cycle": cycler(color=[BLUE, ORANGE, GREEN, VERMILLION, "#CC79A7"]),
        "axes.axisbelow": True, "axes.grid": True,
        "grid.color": "#CCCCCC", "grid.linewidth": .6, "grid.alpha": .4,
        "lines.linewidth": 1.8, "lines.markersize": 4,
        "legend.frameon": False, "hatch.linewidth": .6,
    }
