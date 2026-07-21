"""
Empirical spouse and child count extraction for EB principal proxies.

Constructs pathway-specific proxy populations from ACS microdata and derives
weighted distributions for spouse presence and numbers of foreign-born
children in married households. Outputs JSON files encoding spouse
probabilities and child count distributions by pathway and nationality for
use in configuring family structures in the immigration microsimulation.
"""


import pandas as pd
import numpy as np
import json

BPLD_CODES = {
    50000: "China",
    52100: "India",
}

# https://www.bls.gov/data/inflation_calculator.html
CPI_DEFLATORS = {
    2014: 1.32,
    2015: 1.33,
    2016: 1.31,
    2017: 1.28,
    2018: 1.25,
    2019: 1.23,
    2020: 1.21,
    2021: 1.15,
    2022: 1.06,
    2023: 1.02,
    2024: 1.00,
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
df = pd.read_csv("../acs_2014-2024.csv")

# All wage-based proxy filters below operate on INCWAGE_REAL.
df["INCWAGE_REAL"] = df["INCWAGE"] * df["YEAR"].map(CPI_DEFLATORS)

###########################
# Build proxy populations
###########################

# Base filters shared across all professional pathways (EB-1/2/3/EB-5)
# EB-4 does not use this base mask -- defined separately below
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

# EB-4
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

# Add nationality and spouse flag to all proxy DataFrames
# HasSpouse is computed over the full (unfiltered-by-marital-status) proxy so that
# spouse presence probabilities reflect the real married/unmarried split
for df_proxy in [df_eb1, df_eb2, df_eb3, df_eb4, df_eb5]:
    df_proxy["Nationality"] = df_proxy["BPLD"].map(BPLD_CODES).fillna("ROW")
    df_proxy["HasSpouse"] = df_proxy["MARST"].isin([1, 2]).astype(int)

#########################################################
# Count foreign-born children - married households only
#########################################################


def count_children_for_married_proxy(df_proxy: pd.DataFrame) -> pd.DataFrame:
    """
    Count foreign-born non-citizen children under 21 per MARRIED household.

    Restricted to MARST in [1, 2] (HasSpouse == 1) before child matching.
    Unmarried principals are excluded entirely.

    Returns the married subset of df_proxy with NCHILD_FOREIGN_BORN added.
    Households with no qualifying children receive a count of 0.
    """
    married_proxy = df_proxy[df_proxy["HasSpouse"] == 1].copy()
    proxy_households = married_proxy[["SERIAL", "YEAR"]].copy()

    foreign_born_children = df.merge(proxy_households, on=["SERIAL", "YEAR"], how="inner")
    foreign_born_children = foreign_born_children[
        (foreign_born_children["RELATE"].isin([3, 4]))
        & (foreign_born_children["AGE"] < 21)
        & (foreign_born_children["BPLD"] >= 15000)
        & (foreign_born_children["CITIZEN"] == 3)
        & (foreign_born_children["GQ"] == 1)
    ][["SERIAL", "YEAR"]]

    foreign_born_counts = foreign_born_children.groupby(["SERIAL", "YEAR"]).size()
    df_out = married_proxy.set_index(["SERIAL", "YEAR"]).copy()
    df_out["NCHILD_FOREIGN_BORN"] = foreign_born_counts
    df_out["NCHILD_FOREIGN_BORN"] = df_out["NCHILD_FOREIGN_BORN"].fillna(0).astype(int)
    return df_out.reset_index()


PROXY_FRAMES = [df_eb1, df_eb2, df_eb3, df_eb4, df_eb5]

PATHWAY_TO_PROXY = dict(zip(PATHWAYS, PROXY_FRAMES))

PATHWAY_TO_MARRIED_PROXY = {
    pathway: count_children_for_married_proxy(df_proxy) for pathway, df_proxy in zip(PATHWAYS, PROXY_FRAMES)
}

######################################################
# Extract spouse presence distributions (all workers)
######################################################

empirical_spouse_distributions = {}

for pathway in PATHWAYS:
    empirical_spouse_distributions[pathway] = {}
    df_proxy = PATHWAY_TO_PROXY[pathway]
    is_pooled = pathway in POOLED_NATIONALITY_PATHWAYS

    for nationality in MAJOR_NATIONALITIES:
        # Full proxy -- both married and unmarried -- so the distribution reflects
        # the genuine spouse presence probability
        nat_data = get_nat_data(df_proxy, pathway, nationality)

        spouse_values = nat_data["HasSpouse"].values
        weights = nat_data["PERWT"].values

        spouse_bins = np.array([0, 1])
        weighted_freq = np.zeros(len(spouse_bins))
        for value in spouse_bins:
            mask = spouse_values == value
            weighted_freq[value] = np.sum(weights[mask])

        probabilities = weighted_freq / np.sum(weighted_freq)

        empirical_spouse_distributions[pathway][nationality] = {
            "values": spouse_bins.tolist(),
            "probabilities": probabilities.tolist(),
        }

with open("spouse_presence_empirical_distributions.json", "w") as f:
    json.dump(empirical_spouse_distributions, f, indent=2)

##############################################################
# Extract child count distributions (married principals only)
##############################################################


def extract_child_count_distributions(pathway_to_proxy: dict) -> dict:
    """
    Extract weighted child count distributions from married proxy DataFrames.

    Note: For pathways in POOLED_NATIONALITY_PATHWAYS, all nationality keys receive
    identical pooled distributions (see module docstring for rationale).
    """
    distributions = {}

    for pathway in PATHWAYS:
        distributions[pathway] = {}
        df_proxy = pathway_to_proxy[pathway]
        is_pooled = pathway in POOLED_NATIONALITY_PATHWAYS

        for nationality in MAJOR_NATIONALITIES:
            nat_data = get_nat_data(df_proxy, pathway, nationality)

            child_counts = nat_data["NCHILD_FOREIGN_BORN"].values
            weights = nat_data["PERWT"].values

            # Cap at 5 -- counts above this are extremely rare and unreliable to estimate
            max_count = min(5, int(child_counts.max())) if len(child_counts) > 0 else 0
            count_bins = np.arange(0, max_count + 1)
            weighted_freq = np.zeros(len(count_bins))
            for count in count_bins:
                mask = child_counts == count
                weighted_freq[count] = np.sum(weights[mask])

            probabilities = weighted_freq / np.sum(weighted_freq)

            distributions[pathway][nationality] = {
                "counts": count_bins.tolist(),
                "probabilities": probabilities.tolist(),
            }

    return distributions


children_married = extract_child_count_distributions(PATHWAY_TO_MARRIED_PROXY)

with open("children_count_empirical_distributions_married.json", "w") as f:
    json.dump(children_married, f, indent=2)
