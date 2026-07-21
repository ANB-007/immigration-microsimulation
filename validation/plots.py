"""
Validation plots comparing simulated visa allocations to DOS data.

Loads model and DOS series, then produces side-by-side line charts by
nationality and EB category to visually assess fit over FY2009-FY2024. Outputs
figures with consistent styling to visualization.py. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="white", context="paper")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Okabe-Ito colour-blind-safe hues, matched to simulation/visualization.py.
# Each validation figure shows one entity's Model (solid, circle) vs Actual
# (dashed, square), so the two series stay distinct in greyscale via linestyle
# and marker; colour is a per-figure accent only.
NATIONALITY_COLORS = {"India": "#0072B2", "China": "#E69F00", "ROW": "#009E73"}

EB_COLORS = {
    "EB-1": "#0072B2",
    "EB-2": "#E69F00",
    "EB-3": "#009E73",
    "EB-4": "#CC79A7",
    "EB-5": "#D55E00",
}

os.makedirs("results/nationality", exist_ok=True)
os.makedirs("results/category", exist_ok=True)


def load_data():
    """Load model and actual DOS data."""
    model = pd.read_csv("model_data/visa_consumption.csv")
    actual = pd.read_csv("real_data/dos_visa_consumption_data.csv")

    model = model[model["scenario"] == "Uncapped"].copy()
    model = model.rename(columns={"visas_consumed": "visas"})
    actual = actual.rename(columns={"visas_issued": "visas"})

    model["source"] = "Model"
    actual["source"] = "Actual (DOS)"

    combined = pd.concat(
        [
            model[["year", "category", "nationality", "visas", "source"]],
            actual[["year", "category", "nationality", "visas", "source"]],
        ],
        ignore_index=True,
    )

    return combined, model, actual


def plot_nationality_comparison(combined):
    """Compare visa issuance by nationality -- one figure per nationality."""
    nationalities = ["India", "China", "ROW"]

    for nat in nationalities:
        fig, ax = plt.subplots(figsize=(6.5, 3.9))

        data = combined[(combined["category"] == "Overall") & (combined["nationality"] == nat)]
        pivoted = data.pivot(index="year", columns="source", values="visas").reset_index()

        sns.lineplot(
            data=pivoted,
            x="year",
            y="Model",
            ax=ax,
            marker="o",
            linewidth=3.5,
            markersize=10,
            label="Model",
            color=NATIONALITY_COLORS[nat],
            markeredgewidth=0,
            zorder=3,
        )

        sns.lineplot(
            data=pivoted,
            x="year",
            y="Actual (DOS)",
            ax=ax,
            marker="s",
            linewidth=3,
            markersize=7.5,
            label="Actual",
            color=NATIONALITY_COLORS[nat],
            linestyle="--",
            alpha=0.75,
            markeredgewidth=0,
            zorder=2,
        )

        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Visas Issued", fontsize=13, weight="medium", labelpad=10)
        ax.set_title(
            f"Employment-Based Visas - {nat}: Model vs Actual (2009-2024)",
            fontsize=14,
            weight="bold",
            pad=15,
        )

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
            handlelength=2.5,
            handletextpad=0.8,
            borderpad=1,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.set_xticks(range(2009, 2025, 2))
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(
            f"results/nationality/{nat.lower()}_validation.png",
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def plot_category_comparison(combined):
    """Compare visa issuance by EB category -- one figure per category."""
    categories = ["EB-1", "EB-2", "EB-3", "EB-4", "EB-5"]

    for cat in categories:
        fig, ax = plt.subplots(figsize=(6.5, 3.9))

        data = combined[combined["category"] == cat]
        grouped = data.groupby(["year", "source"])["visas"].sum().reset_index()
        pivoted = grouped.pivot(index="year", columns="source", values="visas").reset_index()

        sns.lineplot(
            data=pivoted,
            x="year",
            y="Model",
            ax=ax,
            marker="o",
            linewidth=3.5,
            markersize=10,
            label="Model",
            color=EB_COLORS[cat],
            markeredgewidth=0,
            zorder=3,
        )

        sns.lineplot(
            data=pivoted,
            x="year",
            y="Actual (DOS)",
            ax=ax,
            marker="s",
            linewidth=3,
            markersize=7.5,
            label="Actual",
            color=EB_COLORS[cat],
            linestyle="--",
            alpha=0.75,
            markeredgewidth=0,
            zorder=2,
        )

        ax.set_xlabel("Year", fontsize=13, weight="medium", labelpad=10)
        ax.set_ylabel("Visas Issued", fontsize=13, weight="medium", labelpad=10)
        ax.set_title(
            f"Employment-Based Visas - {cat}: Model vs Actual (2009-2024)",
            fontsize=14,
            weight="bold",
            pad=15,
        )

        legend = ax.legend(
            frameon=True,
            framealpha=1.0,
            edgecolor="#DADADA",
            loc="best",
            fontsize=12,
            handlelength=2.5,
            handletextpad=0.8,
            borderpad=1,
        )
        legend.get_frame().set_linewidth(1.5)

        ax.set_xticks(range(2009, 2025, 2))
        ax.tick_params(axis="x", rotation=45, labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(True, alpha=0.2, linewidth=0.8, color="#CCCCCC", linestyle="-")
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        plt.tight_layout()
        plt.savefig(
            f"results/category/{cat.replace('-', '').lower()}_validation.png",
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def main():
    combined, model, actual = load_data()
    plot_nationality_comparison(combined)
    plot_category_comparison(combined)


if __name__ == "__main__":
    main()
