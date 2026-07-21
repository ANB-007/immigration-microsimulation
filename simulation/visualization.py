"""
Visualization utilities for the immigration microsimulation.

Generates figures comparing capped and uncapped scenarios,
including age-out trajectories, EB-category conversions, and the composition of
conversions by applicant type. Provides helpers to reshape yearly
simulation states into DataFrames and to apply a consistent visual style
and policy-shift annotations across all exported charts.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .models import EBCategory

OUTPUT_DIR = "output"
PLOT_DPI = 400

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_theme(style="white", context="paper")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Colour palettes: Okabe-Ito colour-blind-safe hues, validated for CVD separation.
# Greyscale/print legibility never relies on colour alone: bar and stacked-area
# fills also carry a hatch pattern (the *_HATCH maps) and line series carry a
# distinct linestyle + marker, so every figure stays readable in black-and-white.
SCENARIO_COLORS = {"Uncapped": "#56B4E9", "Capped": "#D55E00"}
SCENARIO_HATCH = {"Uncapped": "", "Capped": "///"}

NATIONALITY_COLORS = {"India": "#0072B2", "China": "#E69F00", "ROW": "#009E73"}
NATIONALITY_HATCH = {"India": "", "China": "///", "ROW": ".."}

EB_COLORS = {
    "EB-1": "#0072B2",
    "EB-2": "#E69F00",
    "EB-3": "#009E73",
    "EB-4": "#CC79A7",
    "EB-5": "#D55E00",
}

EB_HATCH = {
    "EB-1": "",
    "EB-2": "///",
    "EB-3": "..",
    "EB-4": "xx",
    "EB-5": "\\\\",
}

EB_LINE_STYLES = {
    "EB-1": "-",
    "EB-2": "--",
    "EB-3": "-.",
    "EB-4": ":",
    "EB-5": (0, (3, 1, 1, 1)),
}

EB_MARKERS = {
    "EB-1": "o",
    "EB-2": "s",
    "EB-3": "^",
    "EB-4": "D",
    "EB-5": "v",
}

APPLICANT_COLORS = {
    "principals": "#0072B2",
    "spouses": "#E69F00",
    "children": "#009E73",
}
APPLICANT_HATCH = {"principals": "", "spouses": "///", "children": ".."}


class SimulationVisualizer:
    """Visualization class for workforce simulation results."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_age_out_charts(self, states_uncapped, states_capped) -> Dict[str, str]:
        """
        Generate children age-out charts segmented by nationality and EB category.

        Creates 6 chart files in output/age_outs/:
        - uncapped_by_nationality.png
        - uncapped_by_eb_category.png
        - capped_by_nationality.png
        - capped_by_eb_category.png
        - annual_comparison.png
        - cumulative_comparison.png
        """
        aged_out_dir = self.output_dir / "age_outs"
        aged_out_dir.mkdir(parents=True, exist_ok=True)
        chart_files = {}

        nat_data_uncapped = self._prepare_nationality_data(states_uncapped)
        nat_data_capped = self._prepare_nationality_data(states_capped)

        eb_data_uncapped = self._prepare_eb_category_data(states_uncapped)
        eb_data_capped = self._prepare_eb_category_data(states_capped)

        # filter to 2025 and beyond for nationality and EB-category data
        nat_data_uncapped = nat_data_uncapped[nat_data_uncapped["Year"] >= 2025]
        nat_data_capped = nat_data_capped[nat_data_capped["Year"] >= 2025]
        eb_data_uncapped = eb_data_uncapped[eb_data_uncapped["Year"] >= 2025]
        eb_data_capped = eb_data_capped[eb_data_capped["Year"] >= 2025]

        # Uncapped by Nationality
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        pivot_data = nat_data_uncapped.pivot(index="Year", columns="Nationality", values="Count")
        pivot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.75,
            color=[NATIONALITY_COLORS.get(col, None) for col in pivot_data.columns],
            edgecolor="white",
            linewidth=1.5,
        )
        for _container, _col in zip(ax.containers, pivot_data.columns):
            for _bar in _container:
                _bar.set_hatch(NATIONALITY_HATCH.get(_col, ""))
        ax.set_title(
            "Distribution of Age-Outs by Nationality (Uncapped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Children Aged Out", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            title="Nationality",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        years_list = list(pivot_data.index)
        if 2024 in years_list:
            self._add_shift_annotation(ax, years_list.index(2024))

        plt.tight_layout()

        chart_file = aged_out_dir / "uncapped_by_nationality.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["uncapped_by_nationality"] = str(chart_file)

        # Uncapped by EB Category
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        pivot_data = eb_data_uncapped.pivot(index="Year", columns="Category", values="Count")
        pivot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.75,
            color=[EB_COLORS.get(col, None) for col in pivot_data.columns],
            edgecolor="white",
            linewidth=1.5,
        )
        for _container, _col in zip(ax.containers, pivot_data.columns):
            for _bar in _container:
                _bar.set_hatch(EB_HATCH.get(_col, ""))
        ax.set_title(
            "Distribution of Age-Outs by EB Category (Uncapped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Children Aged Out", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            title="EB Category",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        years_list = list(pivot_data.index)
        if 2024 in years_list:
            self._add_shift_annotation(ax, years_list.index(2024))

        plt.tight_layout()

        chart_file = aged_out_dir / "uncapped_by_eb_category.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["uncapped_by_eb_category"] = str(chart_file)

        # Capped by Nationality
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        pivot_data = nat_data_capped.pivot(index="Year", columns="Nationality", values="Count")
        pivot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.75,
            color=[NATIONALITY_COLORS.get(col, None) for col in pivot_data.columns],
            edgecolor="white",
            linewidth=1.5,
        )
        for _container, _col in zip(ax.containers, pivot_data.columns):
            for _bar in _container:
                _bar.set_hatch(NATIONALITY_HATCH.get(_col, ""))
        ax.set_title(
            "Distribution of Age-Outs by Nationality (Capped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Children Aged Out", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            title="Nationality",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        years_list = list(pivot_data.index)
        if 2024 in years_list:
            self._add_shift_annotation(ax, years_list.index(2024))

        plt.tight_layout()

        chart_file = aged_out_dir / "capped_by_nationality.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["capped_by_nationality"] = str(chart_file)

        # Capped by EB Category
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        pivot_data = eb_data_capped.pivot(index="Year", columns="Category", values="Count")
        pivot_data.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.75,
            color=[EB_COLORS.get(col, None) for col in pivot_data.columns],
            edgecolor="white",
            linewidth=1.5,
        )
        for _container, _col in zip(ax.containers, pivot_data.columns):
            for _bar in _container:
                _bar.set_hatch(EB_HATCH.get(_col, ""))
        ax.set_title(
            "Distribution of Age-Outs by EB Category (Capped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Children Aged Out", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            title="EB Category",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
            title_fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        years_list = list(pivot_data.index)
        if 2024 in years_list:
            self._add_shift_annotation(ax, years_list.index(2024))

        plt.tight_layout()

        chart_file = aged_out_dir / "capped_by_eb_category.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["capped_by_eb_category"] = str(chart_file)

        # Annual Comparison
        annual_df = pd.DataFrame(
            {
                "Year": [state.year for state in states_uncapped],
                "Uncapped": [state.children_aged_out_this_year for state in states_uncapped],
                "Capped": [state.children_aged_out_this_year for state in states_capped],
            }
        )

        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        x = np.arange(len(annual_df))
        width = 0.35

        ax.bar(
            x - width / 2,
            annual_df["Uncapped"],
            width,
            label="Uncapped",
            color=SCENARIO_COLORS["Uncapped"],
            hatch=SCENARIO_HATCH["Uncapped"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.bar(
            x + width / 2,
            annual_df["Capped"],
            width,
            label="Capped",
            color=SCENARIO_COLORS["Capped"],
            hatch=SCENARIO_HATCH["Capped"],
            edgecolor="white",
            linewidth=1.5,
            alpha=0.9,
        )

        ax.set_title(
            "Annual Age-Outs by Scenario",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Children Aged Out", fontsize=13, weight="medium", labelpad=10)
        ax.set_xticks(x[:: max(1, len(x) // 10)])
        ax.set_xticklabels(
            annual_df["Year"].iloc[:: max(1, len(x) // 10)],
            rotation=45,
            ha="right",
            fontsize=12,
        )

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        plt.yticks(fontsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        # 2024 policy-shift annotation
        years_list = annual_df["Year"].tolist()
        if 2024 in years_list:
            self._add_shift_annotation(ax, years_list.index(2024))

        plt.tight_layout()

        chart_file = aged_out_dir / "annual_comparison.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["annual_comparison"] = str(chart_file)

        # Cumulative Comparison
        cumulative_df = pd.DataFrame(
            {
                "Year": [state.year for state in states_uncapped],
                "Uncapped": [state.cumulative_children_aged_out for state in states_uncapped],
                "Capped": [state.cumulative_children_aged_out for state in states_capped],
            }
        )

        fig, ax = plt.subplots(figsize=(6.5, 3.7))

        sns.lineplot(
            data=cumulative_df,
            x="Year",
            y="Uncapped",
            linewidth=3.5,
            marker="o",
            markersize=10,
            label="Uncapped",
            color=SCENARIO_COLORS["Uncapped"],
            markevery=max(1, len(cumulative_df) // 15),
            ax=ax,
            markeredgewidth=0,
        )
        sns.lineplot(
            data=cumulative_df,
            x="Year",
            y="Capped",
            linewidth=3.5,
            marker="s",
            markersize=9,
            label="Capped",
            color=SCENARIO_COLORS["Capped"],
            linestyle="--",
            markevery=max(1, len(cumulative_df) // 15),
            ax=ax,
            markeredgewidth=0,
        )

        ax.fill_between(
            cumulative_df["Year"],
            cumulative_df["Uncapped"],
            cumulative_df["Capped"],
            alpha=0.15,
            color=SCENARIO_COLORS["Capped"],
        )

        ax.set_title(
            "Cumulative Age Outs by Scenario",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Total Children Aged Out", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
            handlelength=2.5,
            handletextpad=0.8,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        # 2024 policy-shift annotation
        self._add_shift_annotation(ax, 2024)

        plt.tight_layout()

        chart_file = aged_out_dir / "cumulative_comparison.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["cumulative_comparison"] = str(chart_file)

        return chart_files

    def generate_eb_conversion_charts(self, states_uncapped, states_capped) -> Dict[str, str]:
        """
        Generate EB category conversion charts showing visas allocated per year.

        Creates 2 chart files in output/conversions/:
        - uncapped_by_eb_category.png
        - capped_by_eb_category.png
        """
        conversions_dir = self.output_dir / "conversions"
        conversions_dir.mkdir(parents=True, exist_ok=True)
        chart_files = {}

        uncapped_df = self._prepare_conversion_data(states_uncapped)
        capped_df = self._prepare_conversion_data(states_capped)

        # Uncapped Conversions
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        for category in sorted(uncapped_df["Category"].unique()):
            cat_data = uncapped_df[uncapped_df["Category"] == category]
            ax.plot(
                cat_data["Year"],
                cat_data["Count"],
                marker=EB_MARKERS.get(category, "o"),
                linestyle=EB_LINE_STYLES.get(category, "-"),
                linewidth=3.5,
                markersize=10,
                label=category,
                color=EB_COLORS.get(category),
                markeredgewidth=0,
                markevery=max(1, len(cat_data) // 15),
            )

        ax.set_title(
            "EB Category Conversions Over Time (Uncapped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel(
            "Annual Conversions",
            fontsize=13,
            weight="medium",
            labelpad=10,
        )

        legend = ax.legend(
            title="EB Category",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
            title_fontsize=12,
            handlelength=2.5,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        # 2024 policy-shift annotation
        self._add_shift_annotation(ax, 2024)

        plt.tight_layout()

        chart_file = conversions_dir / "uncapped_by_eb_category.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["uncapped_by_eb_category"] = str(chart_file)

        # Capped Conversions
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        for category in sorted(capped_df["Category"].unique()):
            cat_data = capped_df[capped_df["Category"] == category]
            ax.plot(
                cat_data["Year"],
                cat_data["Count"],
                marker=EB_MARKERS.get(category, "o"),
                linestyle=EB_LINE_STYLES.get(category, "-"),
                linewidth=3.5,
                markersize=10,
                label=category,
                color=EB_COLORS.get(category),
                markeredgewidth=0,
                markevery=max(1, len(cat_data) // 15),
            )

        ax.set_title(
            "EB Category Conversions Over Time (Capped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel(
            "Annual Conversions",
            fontsize=13,
            weight="medium",
            labelpad=10,
        )

        legend = ax.legend(
            title="EB Category",
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
            title_fontsize=12,
            handlelength=2.5,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        # 2024 policy-shift annotation
        self._add_shift_annotation(ax, 2024)

        plt.tight_layout()

        chart_file = conversions_dir / "capped_by_eb_category.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["capped_by_eb_category"] = str(chart_file)

        return chart_files

    def generate_applicant_type_charts(self, states_uncapped, states_capped) -> Dict[str, str]:
        """
        Generate applicant type conversion charts showing principals, spouses, and children.

        Creates 2 chart files in output/conversions/:
        - applicant_type_uncapped.png
        - applicant_type_capped.png
        """
        conversions_dir = self.output_dir / "conversions"
        conversions_dir.mkdir(parents=True, exist_ok=True)
        chart_files = {}

        # Uncapped by Applicant Type
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        principals_uncapped = [state.converted_temps for state in states_uncapped]
        spouses_uncapped = [state.converted_spouses for state in states_uncapped]
        children_uncapped = [state.children_saved_this_year for state in states_uncapped]
        years_uncapped = [state.year for state in states_uncapped]

        ax.stackplot(
            years_uncapped,
            principals_uncapped,
            spouses_uncapped,
            children_uncapped,
            labels=["Principals", "Spouses", "Children"],
            colors=[
                APPLICANT_COLORS["principals"],
                APPLICANT_COLORS["spouses"],
                APPLICANT_COLORS["children"],
            ],
            alpha=0.85,
            linewidth=0,
        )
        for _poly, _h in zip(ax.collections, APPLICANT_HATCH.values()):
            _poly.set_hatch(_h)

        final_principals = principals_uncapped[-1]
        final_spouses = spouses_uncapped[-1]
        final_children = children_uncapped[-1]
        final_total = final_principals + final_spouses + final_children

        if final_total > 0:
            pct_principals = (final_principals / final_total) * 100
            pct_spouses = (final_spouses / final_total) * 100
            pct_children = (final_children / final_total) * 100
        else:
            pct_principals = pct_spouses = pct_children = 0

        total_conversions = [
            p + s + c for p, s, c in zip(principals_uncapped, spouses_uncapped, children_uncapped)
        ]
        max_conversions = max(total_conversions) if total_conversions else 0
        ax.set_ylim(0, max_conversions * 1.1)

        proportion_text = f"Final Year Proportions\nPrincipals: {pct_principals:.1f}%\nSpouses: {pct_spouses:.1f}%\nChildren: {pct_children:.1f}%"
        ax.text(
            0.97,
            0.97,
            proportion_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="white",
                edgecolor="#DADADA",
                linewidth=2,
                alpha=0.95,
            ),
        )

        ax.set_title(
            "Conversions by Applicant Type (Uncapped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Annual Conversions", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        plt.tight_layout()

        chart_file = conversions_dir / "applicant_type_uncapped.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["applicant_type_uncapped"] = str(chart_file)
        logger.info(f"Saved {chart_file}")

        # Capped by Applicant Type
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        principals_capped = [state.converted_temps for state in states_capped]
        spouses_capped = [state.converted_spouses for state in states_capped]
        children_capped = [state.children_saved_this_year for state in states_capped]
        years_capped = [state.year for state in states_capped]

        ax.stackplot(
            years_capped,
            principals_capped,
            spouses_capped,
            children_capped,
            labels=["Principals", "Spouses", "Children"],
            colors=[
                APPLICANT_COLORS["principals"],
                APPLICANT_COLORS["spouses"],
                APPLICANT_COLORS["children"],
            ],
            alpha=0.85,
            linewidth=0,
        )
        for _poly, _h in zip(ax.collections, APPLICANT_HATCH.values()):
            _poly.set_hatch(_h)

        final_principals = principals_capped[-1]
        final_spouses = spouses_capped[-1]
        final_children = children_capped[-1]
        final_total = final_principals + final_spouses + final_children

        if final_total > 0:
            pct_principals = (final_principals / final_total) * 100
            pct_spouses = (final_spouses / final_total) * 100
            pct_children = (final_children / final_total) * 100
        else:
            pct_principals = pct_spouses = pct_children = 0

        total_conversions = [p + s + c for p, s, c in zip(principals_capped, spouses_capped, children_capped)]
        max_conversions = max(total_conversions) if total_conversions else 0
        ax.set_ylim(0, max_conversions * 1.1)

        proportion_text = f"Final Year Proportions\nPrincipals: {pct_principals:.1f}%\nSpouses: {pct_spouses:.1f}%\nChildren: {pct_children:.1f}%"
        ax.text(
            0.97,
            0.97,
            proportion_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.8",
                facecolor="white",
                edgecolor="#DADADA",
                linewidth=2,
                alpha=0.95,
            ),
        )

        ax.set_title(
            "Conversions by Applicant Type (Capped)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Annual Conversions", fontsize=13, weight="medium", labelpad=10)

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="upper left",
            fontsize=12,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        plt.tight_layout()

        chart_file = conversions_dir / "applicant_type_capped.png"
        plt.savefig(
            chart_file,
            dpi=PLOT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        chart_files["applicant_type_capped"] = str(chart_file)
        logger.info(f"Saved {chart_file}")

        return chart_files

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _add_shift_annotation(self, ax, x_val) -> None:
        """
        Draw a dashed vertical line and labelled callout at x_val.

        Silently does nothing if 2024 falls outside the current x-axis range.
        """
        xlim = ax.get_xlim()
        if not (xlim[0] <= x_val <= xlim[1]):
            return

        ax.axvline(
            x_val,
            color="#555555",
            linewidth=1.5,
            linestyle="--",
            zorder=5,
            alpha=0.7,
        )
        ylim = ax.get_ylim()
        x_offset = (xlim[1] - xlim[0]) * 0.012
        y_text = ylim[0] + (ylim[1] - ylim[0]) * 0.96
        ax.text(
            x_val + x_offset,
            y_text,
            "Policy shift",
            fontsize=11,
            color="#444444",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#CCCCCC",
                linewidth=1.0,
                alpha=0.88,
            ),
            zorder=6,
        )

    def _prepare_nationality_data(self, states) -> pd.DataFrame:
        """Convert state data to DataFrame for nationality charts."""
        data = []
        for state in states:
            counts = {"China": 0, "India": 0, "ROW": 0}
            for child in state.children_aged_out_this_year_list:
                if child.nationality in ["China", "India"]:
                    counts[child.nationality] += 1
                else:
                    counts["ROW"] += 1

            for nat, count in counts.items():
                data.append({"Year": state.year, "Nationality": nat, "Count": count})

        return pd.DataFrame(data)

    def _prepare_eb_category_data(self, states) -> pd.DataFrame:
        """Convert state data to DataFrame for EB category charts."""
        data = []
        for state in states:
            counts = {
                cat: 0
                for cat in [
                    EBCategory.EB1,
                    EBCategory.EB2,
                    EBCategory.EB3,
                    EBCategory.EB4,
                    EBCategory.EB5,
                ]
            }
            for child in state.children_aged_out_this_year_list:
                if child.parent_eb_category:
                    counts[child.parent_eb_category] += 1

            for cat, count in counts.items():
                data.append({"Year": state.year, "Category": cat.value, "Count": count})

        return pd.DataFrame(data)

    def _prepare_conversion_data(self, states) -> pd.DataFrame:
        """Convert state data to DataFrame for conversion charts."""
        data = []
        for state in states:
            for category in [
                EBCategory.EB1,
                EBCategory.EB2,
                EBCategory.EB3,
                EBCategory.EB4,
                EBCategory.EB5,
            ]:
                count = state.converted_by_eb_category_total.get(category, 0)
                data.append(
                    {
                        "Year": state.year,
                        "Category": category.value,
                        "Count": count,
                    }
                )

        return pd.DataFrame(data)
