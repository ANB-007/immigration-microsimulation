# Historical petition inputs

These JSON files contain the historical input tables used by the model. Fiscal years are JSON object keys; `simulation/input_data.py` restores them to integers when loading the tables. Moving the tables from Python modules to JSON does not change their values, ordering or use in the simulation.

| File | Contents |
| --- | --- |
| `eb123.json` | EB-1–3 approved receipt-cohort counts and India/China counts, including the initialization additions below |
| `eb4.json` | SIJ and other EB-4 annual totals, SIJ nationality shares and annual other EB-4 nationality shares |
| `eb5.json` | EB-5 annual totals and annual nationality shares |

## EB-1–3

The USCIS tables group petitions by fiscal year received and report approved status at the query date. These are not annual adjudication flows. The model uses these receipt-cohort series as arrival inputs; the names `I140_ANNUAL_APPROVALS` and `COUNTRY_APPROVALS` are retained in the code for continuity. India EB-2 in FY2022 is 45,351, from `i140_rec_by_class_country_fy2024_q4.xlsx`, sheet `India FY24`, cell J21.

Early-period initialization additions are assumptions, not observed annual flows or directly quoted principal counts. The [NFAP October 2011 report](https://www.nfap.com/pdf/WAITING_NFAP_Policy_Brief_October_2011.pdf) describes stocks including principals and dependents; their conversion to the model's principal units remains an assumption. The existing additions are preserved:

| Series | Year | Construction |
| --- | --- | --- |
| Total EB-1 | 2009 | 13,546 + 3,148 |
| Total EB-1 | 2010 | 14,269 + 2,767 |
| Total EB-2 | 2009 | 18,270 + 9,546 + 9,992 + 5,716 |
| Total EB-3 | 2009 | 16,558 + 80,386 + 30,780 + 83,490 |
| India EB-2 | 2009 | 6,674 + 9,546 |
| China EB-2 | 2009 | 2,515 + 9,992 |
| India EB-3 | 2009 | 4,422 + 80,386 |
| China EB-3 | 2009 | 543 + 30,780 |

## EB-4

The model separates SIJs from other EB-4 principals. The FY2009 SIJ count comes from the [2011 Federal Register notice](https://www.federalregister.gov/documents/2011/09/06/2011-22625/special-immigrant-juvenile-petitions). Other EB-4 totals in 2009–2011 use the 2012 proportion as a proxy because direct data were unavailable.

SIJ nationality shares use unaccompanied-minor court data, with the ROW group primarily representing Central American countries. Other EB-4 nationality shares use R-1 religious-worker visa distributions. These are proxies for nationality composition, not direct observations of the full EB-4 population. The manuscript explains their implications for queue composition and age-outs.

## EB-5

Annual totals in 2009 and 2010 come from the [USCIS March 2011 presentation](https://www.uscis.gov/sites/default/files/document/presentations/EB-5-presentation-March-2011.pdf); the 2011 count follows the [published summary of USCIS petition-performance data](https://www.fosterglobal.com/blog/u-s-citizenship-and-immigration-services-releases-performance-data-on-i-526-and-i-829-petitions/).

Nationality shares through 2013 use visa issuance data. Shares for 2014–2024 interpolate between 2013 and 2024 using the later endpoint from the [IIUSA country analysis](https://iiusa.org/wp-content/uploads/2024/09/I526E-Filings-and-Ajudication-by-Country-Analysis-final-full-report.pdf). Annual totals are retained separately; they are not interpolated. The manuscript discusses how smoothing nationality shares can misrepresent the timing of country-specific demand and the resulting backlog.
