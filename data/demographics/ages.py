"""
Empirical age distribution extraction for principals and children.

Builds pathway-specific proxy populations from ACS microdata, then derives
weighted age distributions for employment-based principals and entry-age
distributions for their foreign-born children. Outputs JSON files encoding
principal age and child entry-age distributions by pathway and nationality
for use in the immigration microsimulation.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

if __package__:
    from .income_adjustment import adjust_wages
else:
    from income_adjustment import adjust_wages

# Matplotlib configuration
sns.set_theme(style="white", context="talk")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 400
plt.rcParams["axes.facecolor"] = "#FFFFFF"
plt.rcParams["figure.facecolor"] = "#FFFFFF"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

NATIONALITY_COLORS = {
    "India": "#1E88E5",
    "China": "#8E24AA",
    "ROW": "#616161",
}

BPLD_CODES = {
    50000: "China",
    52100: "India",
}

# SIJS excluded
PATHWAYS = ["EB-1", "EB-2", "EB-3", "Other_EB4", "EB-5"]
MAJOR_NATIONALITIES = ["China", "India", "ROW"]

# Pathways whose distributions are pooled across nationalities due to sparse per-country samples
POOLED_NATIONALITY_PATHWAYS = {"Other_EB4"}

def get_nat_data(df_proxy: pd.DataFrame, pathway: str, nationality: str) -> pd.DataFrame:
    """
    Return the nationality-filtered proxy subset for standard pathways, or the
    full pooled proxy for pathways in POOLED_NATIONALITY_PATHWAYS.

    Other_EB4 is pooled because per-nationality cell sizes (<200 observations)
    are too small for reliable empirical distribution estimation. The pooled
    distribution is exported to all nationality keys to preserve uniform JSON structure.
    """
    if pathway in POOLED_NATIONALITY_PATHWAYS:
        return df_proxy
    return df_proxy[df_proxy["Nationality"] == nationality]


EB_PROFESSIONAL_OCC = (
    list(range(10, 431))  # Management 
    + list(range(500, 951))  # Business and Financial Operations 
    + list(range(1000, 1241))  # Computer and Mathematical 
    + list(range(1300, 1561))  # Architecture and Engineering 
    + list(range(1600, 1761))  # Life, Physical, and Social Sciences
    + list([2200])  # Professors
    + list(range(3000, 3541))  # Healthcare Practitioners and Technical
)

EB4_RELIGIOUS_OCC = [
    2040,
    2050,
    2060,
]

# Load data
df = pd.read_csv(Path(__file__).with_name("acs_2014-2024.csv"))

# Annual survey-year CPI-U to September 2024 dollars, without factor rounding.
# All wage-based proxy filters below operate on INCWAGE_REAL.
df["INCWAGE_REAL"] = adjust_wages(df["INCWAGE"], df["YEAR"])

##########################
# Build proxy populations
##########################

# Base filters shared across all professional pathways (EB-1/2/3/EB-5)
# EB-4 does not use this base mask - defined separately below
_base = (
    (df["RELATE"] == 1)
    & (df["BPLD"] >= 15000)
    & (df["CITIZEN"] == 3)
    & (df["GQ"] == 1)
    & (df["CLASSWKRD"].isin([22, 23]))
    & ((df["YEAR"] - df["YRIMMIG"]) <= 5)
    & (df["AGE"] >= 22)
    & (df["AGE"] <= 55)
    & (df["UHRSWORK"] >= 35)
    & (df["WKSWORK2"] >= 4)
)

# EB-1
df_eb1 = df[
    _base
    & (df["EDUC"] >= 11)
    & (df["OCC2010"].isin(EB_PROFESSIONAL_OCC))
    & (df["AGE"] >= 30)
    & (df["INCWAGE_REAL"] >= 130000)
].copy()

# EB-2
df_eb2 = df[
    _base & (df["EDUC"] >= 10) & (df["OCC2010"].isin(EB_PROFESSIONAL_OCC)) & (df["INCWAGE_REAL"] >= 100000)
].copy()

# EB-3
df_eb3 = df[
    _base & (df["EDUC"] == 10) & (df["OCC2010"].isin(EB_PROFESSIONAL_OCC)) & (df["INCWAGE_REAL"] >= 65000)
].copy()

# EB-4:
df_eb4 = df[
    (df["RELATE"] == 1)
    & (df["BPLD"] >= 15000)
    & (df["CITIZEN"] == 3)
    & (df["GQ"] == 1)
    & (df["CLASSWKRD"].isin([22, 23]))
    & ((df["YEAR"] - df["YRIMMIG"]) <= 5)
    & (df["AGE"] >= 22)
    & (df["AGE"] <= 55)
    & (df["OCC2010"].isin(EB4_RELIGIOUS_OCC))
].copy()

# EB-5
df_eb5 = df[_base & (df["AGE"] >= 35) & (df["INCWAGE_REAL"] >= 200000)].copy()

# Add nationality to all proxy DataFrames
for df_proxy in [df_eb1, df_eb2, df_eb3, df_eb4, df_eb5]:
    df_proxy["Nationality"] = df_proxy["BPLD"].map(BPLD_CODES).fillna("ROW")

PROXY_FRAMES = [df_eb1, df_eb2, df_eb3, df_eb4, df_eb5]
PATHWAY_TO_PROXY = dict(zip(PATHWAYS, PROXY_FRAMES))

#######################################################
# Build children populations (married households only)
#######################################################


def get_children_for_proxy(df_proxy: pd.DataFrame) -> pd.DataFrame:
    """
    Get foreign-born children linked to MARRIED households in the proxy population.
    Nationality is inherited from the parent via merge on (SERIAL, YEAR).

    The married filter (MARST in [1, 2]) is applied here rather than in the base
    proxy so that principal age distributions can use the full proxy population
    while child entry-age distributions remain restricted to married households.
    """
    married_proxy = df_proxy[df_proxy["MARST"].isin([1, 2])]
    all_households = married_proxy[["SERIAL", "YEAR", "Nationality"]].copy()
    children = df.merge(all_households, on=["SERIAL", "YEAR"], how="inner")
    children = children[
        (children["AGE"] < 21)
        & (children["BPLD"] >= 15000)
        & (children["CITIZEN"] == 3)
        & (children["RELATE"].isin([3, 4]))
        & (children["GQ"] == 1)
    ].copy()
    children = children[(children["YRSUSA1"] >= 0) & (children["YRSUSA1"] <= children["AGE"])].copy()
    children["Entry_Age"] = (children["AGE"] - children["YRSUSA1"]).clip(lower=0, upper=20)
    return children


PATHWAY_TO_CHILDREN = {
    pathway: get_children_for_proxy(df_proxy) for pathway, df_proxy in zip(PATHWAYS, PROXY_FRAMES)
}

######################################
# Extract principal age distributions
######################################

principal_age_distributions = {}

for pathway in PATHWAYS:
    principal_age_distributions[pathway] = {}
    df_proxy = PATHWAY_TO_PROXY[pathway]
    is_pooled = pathway in POOLED_NATIONALITY_PATHWAYS

    for nationality in MAJOR_NATIONALITIES:
        nat_data = get_nat_data(df_proxy, pathway, nationality)

        ages = nat_data["AGE"].values
        weights = nat_data["PERWT"].values

        min_age = int(ages.min()) if len(ages) > 0 else 22
        max_age = 55

        age_bins = np.arange(min_age, max_age + 1)
        weighted_freq = np.array([np.sum(weights[ages == age]) for age in age_bins])
        probabilities = weighted_freq / weighted_freq.sum()

        principal_age_distributions[pathway][nationality] = {
            "ages": age_bins.tolist(),
            "probabilities": probabilities.tolist(),
        }

with open(Path(__file__).resolve().parent / "principal_age_empirical_distributions.json", "w") as f:
    json.dump(principal_age_distributions, f, indent=2)

##################################################################
# Extract child entry-age distributinos (married principals only)
##################################################################

child_entry_age_distributions = {}

for pathway in PATHWAYS:
    child_entry_age_distributions[pathway] = {}
    children = PATHWAY_TO_CHILDREN[pathway]
    is_pooled = pathway in POOLED_NATIONALITY_PATHWAYS

    for nationality in MAJOR_NATIONALITIES:
        nat_data = get_nat_data(children, pathway, nationality)

        ages = nat_data["Entry_Age"].values
        weights = nat_data["PERWT"].values

        age_bins = np.arange(0, 21)
        weighted_freq = np.array([np.sum(weights[ages == age]) for age in age_bins])
        probabilities = weighted_freq / weighted_freq.sum()

        child_entry_age_distributions[pathway][nationality] = {
            "ages": age_bins.tolist(),
            "probabilities": probabilities.tolist(),
        }

with open(Path(__file__).resolve().parent / "child_entry_age_empirical_distributions.json", "w") as f:
    json.dump(child_entry_age_distributions, f, indent=2)
