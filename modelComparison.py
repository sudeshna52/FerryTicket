import pandas as pd


# ============================================================
# 1. LOAD MODEL RESULTS
# ============================================================

baseline = pd.read_csv("baseline_model_results.csv")
ml = pd.read_csv("ml_model_results.csv")
arima = pd.read_csv("arima_results.csv")


# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================

baseline = baseline.rename(
    columns={"MAPE": "MAPE (%)"}
)


# ============================================================
# 3. COMBINE ALL RESULTS
# ============================================================

all_results = pd.concat(
    [baseline, ml, arima],
    ignore_index=True
)


# ============================================================
# 4. DISPLAY COMPLETE COMPARISON
# ============================================================

print("\n==============================================")
print("COMPLETE MODEL COMPARISON")
print("==============================================")

print(
    all_results.to_string(index=False)
)


# ============================================================
# 5. BEST MODEL BY MAE
# ============================================================

print("\n==============================================")
print("BEST MODEL BY HORIZON - MAE")
print("==============================================")

horizons = [
    "15 Minutes",
    "30 Minutes",
    "1 Hour",
    "2 Hours"
]

best_models = []

for horizon in horizons:

    horizon_data = all_results[
        all_results["Horizon"] == horizon
    ]

    best = horizon_data.loc[
        horizon_data["MAE"].idxmin()
    ]

    best_models.append({
        "Horizon": horizon,
        "Best Model": best["Model"],
        "MAE": best["MAE"],
        "RMSE": best["RMSE"],
        "MAPE (%)": best["MAPE (%)"]
    })

    print(f"\n{horizon}")
    print(f"Best Model : {best['Model']}")
    print(f"MAE        : {best['MAE']:.2f}")
    print(f"RMSE       : {best['RMSE']:.2f}")

    if pd.notna(best["MAPE (%)"]):
        print(f"MAPE       : {best['MAPE (%)']:.2f}%")
    else:
        print("MAPE       : Not available")


# ============================================================
# 6. BEST MODEL TABLE
# ============================================================

best_models_df = pd.DataFrame(
    best_models
)

print("\n==============================================")
print("BEST MODEL SUMMARY")
print("==============================================")

print(
    best_models_df.to_string(index=False)
)


# ============================================================
# 7. OVERALL BEST MODEL
# ============================================================

overall = (
    all_results
    .groupby("Model")["MAE"]
    .mean()
    .sort_values()
)

print("\n==============================================")
print("OVERALL MODEL RANKING BY AVERAGE MAE")
print("==============================================")

print(
    overall.to_string()
)

print(
    f"\nOverall best model: {overall.index[0]}"
)


# ============================================================
# 8. SAVE FINAL RESULTS
# ============================================================

all_results.to_csv(
    "complete_model_comparison.csv",
    index=False
)

best_models_df.to_csv(
    "best_model_by_horizon.csv",
    index=False
)

overall.reset_index(
    name="Average MAE"
).to_csv(
    "overall_model_ranking.csv",
    index=False
)


print("\n==============================================")
print("FILES SAVED")
print("==============================================")

print("complete_model_comparison.csv")
print("best_model_by_horizon.csv")
print("overall_model_ranking.csv")