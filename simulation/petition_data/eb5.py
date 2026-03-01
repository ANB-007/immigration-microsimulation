"""
Historical EB-5 approvals and evolving nationality distributions (FY2009-FY2024).

Defines annual EB-5 petition approval totals and associated nationality shares,
combining early years with published issuance data and later years with
interpolated China/India/ROW proportions. These series are used to assign EB-5
principals by country of birth in the microsimulation's reconstruction and
projection periods.
"""


EB5_ANNUAL_TOTALS_BY_YEAR = {
    2009: 1262,  # https://www.uscis.gov/sites/default/files/document/presentations/EB-5-presentation-March-2011.pdf
    2010: 1369,  # https://www.uscis.gov/sites/default/files/document/presentations/EB-5-presentation-March-2011.pdf
    2011: 1571,  # https://www.fosterglobal.com/blog/u-s-citizenship-and-immigration-services-releases-performance-data-on-i-526-and-i-829-petitions/#:~:text=Table_title:%20U.S.%20Citizenship%20and%20Immigration%20Services%20Releases,%7C%20:%206%2C346%20%7C%20:%203%2C699%20%7C
    2012: 3677,
    2013: 3699,
    2014: 5115,
    2015: 8683,
    2016: 7632,
    2017: 11321,
    2018: 13571,
    2019: 3659,
    2020: 2577,
    2021: 2398,
    2022: 590,
    2023: 2212,
    2024: 5385,
}

EB5_NATIONALITY_DISTRIBUTION_BY_YEAR = {
    # Pre-retrogression: Actual visa issuance data
    2009: {"China": 0.4692, "India": 0.0171, "ROW": 0.5138},
    2010: {"China": 0.4095, "India": 0.0329, "ROW": 0.5576},
    2011: {"China": 0.6954, "India": 0.0107, "ROW": 0.294},
    2012: {"China": 0.8016, "India": 0.0099, "ROW": 0.1885},
    2013: {"China": 0.8051, "India": 0.0102, "ROW": 0.1847},
    # Linear interpolation from 2013 to 2024
    # https://iiusa.org/wp-content/uploads/2024/09/I526E-Filings-and-Ajudication-by-Country-Analysis-final-full-report.pdf
    2014: {"China": 0.7808, "India": 0.0264, "ROW": 0.1928},
    2015: {"China": 0.7565, "India": 0.0425, "ROW": 0.2009},
    2016: {"China": 0.7323, "India": 0.0587, "ROW": 0.2091},
    2017: {"China": 0.708, "India": 0.0749, "ROW": 0.2172},
    2018: {"China": 0.6837, "India": 0.091, "ROW": 0.2253},
    2019: {"China": 0.6594, "India": 0.1072, "ROW": 0.2334},
    2020: {"China": 0.6351, "India": 0.1233, "ROW": 0.2415},
    2021: {"China": 0.6108, "India": 0.1395, "ROW": 0.2496},
    2022: {"China": 0.5866, "India": 0.1557, "ROW": 0.2578},
    2023: {"China": 0.5623, "India": 0.1718, "ROW": 0.2659},
    2024: {"China": 0.538, "India": 0.188, "ROW": 0.274},
}
