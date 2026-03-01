"""
Historical I-140 approvals and nationality-specific EB-2/EB-3 adjustments (FY2009-FY2024).

Defines annual I-140 approval counts for EB-1, EB-2, and EB-3, including
backlog-based corrections to early-period USCIS noise, and decomposes EB-1/2/3
approvals by country of birth for India and China. These series anchor the
microsimulation's reconstruction of petition inflows and nationality mixes
across employment-based categories.
"""

I140_ANNUAL_APPROVALS = {
    "EB-1": {
        2009: 13546
        + 3148,  # adjust for USCIS noise; without this, a mathematically impossible difference occurs between model and actual
        2010: 14269
        + 2767,  # adjust for USCIS noise; without this, a mathematically impossible difference occurs between model and actual
        2011: 13989,
        2012: 15029,
        2013: 17602,
        2014: 19833,
        2015: 21807,
        2016: 25074,
        2017: 27121,
        2018: 23106,
        2019: 19322,
        2020: 17430,
        2021: 18517,
        2022: 21875,
        2023: 30935,
        2024: 27570,
    },
    "EB-2": {
        2009: 18270
        + (9546 + 9992)
        + 5716,  # calculated using NFAP backlog estimates in 2011; India + China backlogs (https://nfap.com/pdf/0507brief-greencard-backlog.pdf); necessary adjustment to account for USCIS noise
        2010: 35912,
        2011: 44000,
        2012: 42601,
        2013: 38272,
        2014: 45890,
        2015: 54912,
        2016: 77522,
        2017: 65855,
        2018: 65878,
        2019: 73536,
        2020: 63631,
        2021: 66429,
        2022: 78613,
        2023: 81618,
        2024: 74819,
    },
    "EB-3": {
        2009: 16558
        + (
            80386 + 30780 + 83490
        ),  # calculated using NFAP backlog estimates in 2011; India + China + ROW backlogs (https://nfap.com/pdf/0507brief-greencard-backlog.pdf)
        2010: 20168,
        2011: 17098,
        2012: 9817,
        2013: 8631,
        2014: 13631,
        2015: 17301,
        2016: 31665,
        2017: 32042,
        2018: 34749,
        2019: 41701,
        2020: 38210,
        2021: 83553,
        2022: 52920,
        2023: 57047,
        2024: 55066,
    },
}

COUNTRY_APPROVALS = {
    "India": {
        "EB-1": {
            2009: 2261,
            2010: 2627,
            2011: 3038,
            2012: 3627,
            2013: 4634,
            2014: 6370,
            2015: 6126,
            2016: 7737,
            2017: 8498,
            2018: 7576,
            2019: 6880,
            2020: 6194,
            2021: 7248,
            2022: 8130,
            2023: 10942,
            2024: 8780,
        },
        "EB-2": {
            2009: 6674
            + 9546,  # calculated using NFAP backlog estimates in 2011 (https://nfap.com/pdf/0507brief-greencard-backlog.pdf)
            2010: 15313,
            2011: 22286,
            2012: 22311,
            2013: 20936,
            2014: 25009,
            2015: 31543,
            2016: 47474,
            2017: 40902,
            2018: 39056,
            2019: 43320,
            2020: 35004,
            2021: 37630,
            2022: 45251,
            2023: 39290,
            2024: 38842,
        },
        "EB-3": {
            2009: 4422
            + 80386,  # calculated using NFAP backlog estimates in 2011 (https://nfap.com/pdf/0507brief-greencard-backlog.pdf)
            2010: 6682,
            2011: 6929,
            2012: 4110,
            2013: 3403,
            2014: 3825,
            2015: 6246,
            2016: 9947,
            2017: 8611,
            2018: 8066,
            2019: 11189,
            2020: 9046,
            2021: 48051,
            2022: 16601,
            2023: 12580,
            2024: 10113,
        },
    },
    "China": {
        "EB-1": {
            2009: 2127,
            2010: 2771,
            2011: 2432,
            2012: 2707,
            2013: 3250,
            2014: 3823,
            2015: 4308,
            2016: 4614,
            2017: 4938,
            2018: 4493,
            2019: 3707,
            2020: 2967,
            2021: 3398,
            2022: 4363,
            2023: 7151,
            2024: 6288,
        },
        "EB-2": {
            2009: 2515
            + 9992,  # calculated using NFAP backlog estimates in 2011 (https://nfap.com/pdf/0507brief-greencard-backlog.pdf)
            2010: 3559,
            2011: 3814,
            2012: 3910,
            2013: 3356,
            2014: 3964,
            2015: 5051,
            2016: 7336,
            2017: 6170,
            2018: 7897,
            2019: 10411,
            2020: 10546,
            2021: 9610,
            2022: 9979,
            2023: 12420,
            2024: 12228,
        },
        "EB-3": {
            2009: 543
            + 30780,  # calculated using NFAP backlog estimates in 2011 (https://nfap.com/pdf/0507brief-greencard-backlog.pdf)
            2010: 677,
            2011: 554,
            2012: 358,
            2013: 378,
            2014: 3933,
            2015: 1345,
            2016: 3417,
            2017: 4531,
            2018: 4240,
            2019: 3858,
            2020: 4622,
            2021: 10213,
            2022: 3965,
            2023: 3370,
            2024: 7199,
        },
    },
}
