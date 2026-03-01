"""
Validation utilities for comparing simulated visa allocations to DOS benchmarks.

Provides functions to compute standard time-series fit metrics (MAPE, RMSE,
bias, correlation, turning-point accuracy) between model outputs and official
data, along with nationality- and EB-category-specific diagnostics. Includes a
CLI entry point that loads model and DOS visa consumption files, generates a
text validation report, and writes it to validation_metrics_report.txt.
"""

import pandas as pd
import sys
import os
from math import sqrt


def compute_metrics(sim, real, label=""):
    """
    Compute validation metrics for time series comparison.

    Args:
        sim: Series of simulated values indexed by year
        real: Series of actual values indexed by year
        label: Descriptive label for the comparison

    Returns:
        Dictionary containing:
            - T: Number of overlapping time periods
            - MAPE: Mean Absolute Percentage Error
            - RMSE: Root Mean Square Error
            - Bias: Mean error (model - actual)
            - Corr: Pearson correlation coefficient
            - TurningPointAcc: Fraction of correctly predicted direction changes
            - MeanReal: Mean of actual values
            - MeanSim: Mean of simulated values
            - TotalAbsError: Sum of absolute errors across all years
            - TotalReal: Sum of actual values across all years
    """
    df = pd.concat({"sim": sim, "real": real}, axis=1).dropna().copy()
    T = len(df)
    if T == 0:
        return {"label": label, "T": 0}

    # Calculate MAPE only on non-zero years, but other metrics on ALL years
    df_nonzero = df[df["real"] != 0].copy()

    if len(df_nonzero) > 0:
        mape = (df_nonzero.eval("abs((sim - real) / real)")).mean()
    else:
        mape = float("nan")

    # Calculate other metrics on ALL years (including years where real = 0)
    rmse = sqrt(((df["sim"] - df["real"]) ** 2).mean())
    bias = (df["sim"] - df["real"]).mean()

    # Correlation requires at least 2 points and variation
    if len(df) >= 2 and df["sim"].std() > 0 and df["real"].std() > 0:
        corr = df["sim"].corr(df["real"])
    else:
        corr = float("nan")

    # Calculate total absolute error (sum across all years)
    total_abs_error = (df["sim"] - df["real"]).abs().sum()
    total_real = df["real"].sum()

    df_sorted = df.sort_index()
    ds = df_sorted["sim"].diff()
    dr = df_sorted["real"].diff()
    mask = ds.notna() & dr.notna()
    ds = ds[mask]
    dr = dr[mask]

    sign_sim = ds.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    sign_real = dr.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    valid = sign_real != 0
    N = valid.sum()
    if N > 0:
        turning = (sign_sim[valid] == sign_real[valid]).mean()
    else:
        turning = float("nan")

    return {
        "label": label,
        "T": T,
        "MAPE": mape,
        "RMSE": rmse,
        "Bias": bias,
        "Corr": corr,
        "TurningPointAcc": turning,
        "MeanReal": df["real"].mean(),
        "MeanSim": df["sim"].mean(),
        "TotalAbsError": total_abs_error,
        "TotalReal": total_real,
    }


def print_metrics(m):
    """Print formatted metrics."""
    if m.get("T", 0) == 0:
        print(f"\n{m.get('label','')} - no overlapping data (T=0).")
        return

    print(f"\n{m['label']} (T={m['T']} years)")
    print(f"  Mean level (real):   {m['MeanReal']:>10,.1f}")
    print(f"  Mean level (model):  {m['MeanSim']:>10,.1f}")
    print(f"  MAPE:                {m['MAPE']*100:>10.2f}%")
    print(f"  RMSE:                {m['RMSE']:>10,.1f}")
    print(f"  Bias (model-real):   {m['Bias']:>10,.1f}")
    print(f"  Correlation:         {m['Corr']:>10.3f}")
    print(f"  Turning-point acc.:  {m['TurningPointAcc']*100:>10.2f}%")


def compute_summary_statistics(model_df, actual_df):
    """Compute overall summary statistics."""
    print("\n" + "#" * 100)
    print("Validation Summary Statistics")
    print("#" * 100)

    model_overall = model_df[model_df["category"] == "Overall"].groupby("year")["visas"].sum().reset_index()
    actual_overall = (
        actual_df[actual_df["category"] == "Overall"].groupby("year")["visas"].sum().reset_index()
    )

    diff_df = pd.merge(model_overall, actual_overall, on="year", suffixes=("_model", "_actual"))
    diff_df["difference"] = diff_df["visas_model"] - diff_df["visas_actual"]
    diff_df["pct_difference"] = (diff_df["difference"] / diff_df["visas_actual"] * 100).round(2)

    total_model = model_overall["visas"].sum()
    total_actual = actual_overall["visas"].sum()
    total_diff = total_model - total_actual
    total_pct_diff = total_diff / total_actual * 100

    print(f"\nTotal Visas (2009-2024):")
    print(f"  Model:      {total_model:>12,}")
    print(f"  Actual:     {total_actual:>12,}")
    print(f"  Difference: {total_diff:>12,} ({total_pct_diff:+.2f}%)")

    print(f"\nMean Absolute Error (MAE): {diff_df['difference'].abs().mean():,.0f}")
    print(f"Root Mean Square Error (RMSE): {(diff_df['difference']**2).mean()**0.5:,.0f}")
    print(f"Mean Percentage Error: {diff_df['pct_difference'].mean():+.2f}%")


def compute_nationality_metrics(model_df, actual_df):
    """Compute metrics by nationality (full period 2009-2024)."""
    nationalities = ["India", "China", "ROW"]

    print("\n" + "#" * 100)
    print("Metrics by Nationality (2009-2024)")
    print("#" * 100)

    for nationality in nationalities:
        nat_model = (
            model_df[(model_df["category"] == "Overall") & (model_df["nationality"] == nationality)]
            .groupby("year")["visas"]
            .sum()
        )
        nat_actual = (
            actual_df[(actual_df["category"] == "Overall") & (actual_df["nationality"] == nationality)]
            .groupby("year")["visas"]
            .sum()
        )
        print_metrics(compute_metrics(nat_model, nat_actual, label=nationality))


def compute_category_metrics(model_df, actual_df):
    """Compute metrics by EB category (full period 2009-2024)."""
    categories = ["EB-1", "EB-2", "EB-3", "EB-4", "EB-5"]

    print("\n" + "#" * 100)
    print("Metrics by EB Category (2009-2024)")
    print("#" * 100)

    for category in categories:
        cat_model = model_df[model_df["category"] == category].groupby("year")["visas"].sum()
        cat_actual = actual_df[actual_df["category"] == category].groupby("year")["visas"].sum()
        print_metrics(compute_metrics(cat_model, cat_actual, label=category))


def main():
    """Compute all validation metrics."""
    os.makedirs("results", exist_ok=True)
    output_file = "results/validation_metrics_report.txt"

    with open(output_file, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("Loading data...")
        model_df = pd.read_csv("model_data/visa_consumption.csv")
        actual_df = pd.read_csv("real_data/dos_visa_consumption_data.csv")

        model_df = model_df[model_df["scenario"] == "Uncapped"].copy()

        model_df = model_df.rename(columns={"visas_consumed": "visas"})
        actual_df = actual_df.rename(columns={"visas_issued": "visas"})

        print(f"Model data: {len(model_df)} rows")
        print(f"Actual data: {len(actual_df)} rows")

        compute_summary_statistics(model_df, actual_df)
        compute_nationality_metrics(model_df, actual_df)
        compute_category_metrics(model_df, actual_df)

        print("\n" + "#" * 100)
        print("All validation metrics computed successfully!")
        print("#" * 100)

        sys.stdout = original_stdout

    print(f"Validation metrics saved to: {output_file}")


if __name__ == "__main__":
    main()
