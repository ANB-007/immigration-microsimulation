"""Convert nominal ACS wages to September 2024 purchasing power.

Use survey-year annual-average CPI-U for these annual ACS samples. ACS income
refers to the preceding 12 months; this annual convention does not reconstruct
respondent-specific interview or earnings dates. Inputs must be nominal INCWAGE,
not an already inflation-adjusted IPUMS variable or a pooled multi-year extract.

Source: BLS CPI-U, U.S. city average, all items, not seasonally adjusted
(CUUR0000SA0), monthly and annual tables:
https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202412.pdf
IPUMS income definitions and special codes:
https://usa.ipums.org/usa-action/variables/INCWAGE
"""

import pandas as pd


CPI_U_ANNUAL = {
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}
CPI_U_SEPTEMBER_2024 = 315.301


def adjust_wages(wages: pd.Series, years: pd.Series) -> pd.Series:
    """Return September 2024 wages, retaining missing values as missing.

    Do not round factors or wages before threshold comparisons. Reject unknown
    years rather than silently excluding their records through a missing factor.
    IPUMS N/A (999999) and missing (999998) codes are not high wages.
    """
    if not wages.index.equals(years.index):
        raise ValueError("Wages and survey years must have matching indexes")
    annual_cpi = years.map(CPI_U_ANNUAL)
    if annual_cpi.isna().any():
        invalid = years[annual_cpi.isna()].unique().tolist()
        raise ValueError(f"Missing CPI-U for survey year(s): {invalid}")
    nominal = wages.mask(wages.isin([999998, 999999]))
    return nominal * (CPI_U_SEPTEMBER_2024 / annual_cpi)
