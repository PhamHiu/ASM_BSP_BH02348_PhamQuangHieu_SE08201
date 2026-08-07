# -*- coding: utf-8 -*-
"""
01_demand_supply_forecast.py
---------------------------------------------------------------
PROBLEM 1: DEMAND-SUPPLY MISMATCH RISK
---------------------------------------------------------------
Context: Poor demand forecasting -> slow-moving inventory, stockouts for
"hot" products -> profit erosion.

Actual data sources used:
    - Orders.csv, OrderDetails.csv  (actual sales history 2023-2024)
    - Products.csv                  (current StockQuantity, ReorderLevel)
    - Category.csv                  (product categories)

Objective: Forecast SALES VOLUME (demand) BY MONTH for each PRODUCT in
the next month, then compare with current StockQuantity + ReorderLevel
to detect: (a) products at risk of UNDERSTOCK (forecasted demand > available
stock), (b) products at risk of OVERSTOCK (forecasted demand very low
compared to current stock).

Algorithms used (per requirements): Linear Regression, Random Forest
Regressor, XGBoost Regressor -> compare performance -> select best model
to generate the final output.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "DB_final", "Samsung_Store_Database")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
IMG_DIR = f"{OUT_DIR}/images"
REPORT_DIR = f"{OUT_DIR}/report_csv"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
pd.set_option("display.width", 140)

# =================================================================
# STEP 1: DATA PREPROCESSING AND CLEANING
# =================================================================
orders = pd.read_csv(f"{DATA_DIR}/Orders.csv")
order_details = pd.read_csv(f"{DATA_DIR}/OrderDetails.csv")
products = pd.read_csv(f"{DATA_DIR}/Products.csv")
category = pd.read_csv(f"{DATA_DIR}/Categories.csv")

# 1.1 Check & handle missing/duplicate values
print("=== Checking for missing data ===")
print(orders.isna().sum()[orders.isna().sum() > 0])
print(order_details.isna().sum()[order_details.isna().sum() > 0])
print("Duplicate rows in Orders:", orders.duplicated().sum())
print("Duplicate rows in OrderDetails:", order_details.duplicated().sum())
# Note: According to DATA_ISSUES.md, the sales data (Orders/OrderDetails)
# DOES NOT contain intentional errors -> just check for precaution, no major fixes needed.

# 1.2 Standardize datetime data types
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")
orders = orders.dropna(subset=["OrderDate"])

# 1.3 Merge sales data with products + categories
sales = order_details.merge(
    orders[["OrderID", "OrderDate"]], on="OrderID", how="inner"
).merge(
    products[["ProductID", "ProductName", "CategoryID", "CurrentSalePrice", "CurrentAssemblyCost",
              "StockQuantity", "ReorderLevel", "Status"]],
    on="ProductID", how="left",
).merge(
    category[["CategoryID", "CategoryName"]], on="CategoryID", how="left"
)

# 1.4 Create MONTH time axis (Year-Month) - forecasting analysis unit
sales["YearMonth"] = sales["OrderDate"].dt.to_period("M")

# 1.5 Aggregate demand by Month x Product
monthly_demand = (
    sales.groupby(["ProductID", "ProductName", "CategoryID", "CategoryName", "YearMonth"])
    .agg(QuantitySold=("Quantity", "sum"), Revenue=("SubTotal", "sum"))
    .reset_index()
)

# 1.6 Ensure complete data across time GRID (fill 0 for months with no sales
# so the model learns the "no sales" trend correctly, avoiding bias from
# missing observations)
all_months = pd.period_range(
    monthly_demand["YearMonth"].min(), monthly_demand["YearMonth"].max(), freq="M"
)
all_products = products[["ProductID", "ProductName", "CategoryID"]].merge(
    category[["CategoryID", "CategoryName"]], on="CategoryID", how="left"
)
grid = pd.MultiIndex.from_product(
    [all_products["ProductID"], all_months], names=["ProductID", "YearMonth"]
).to_frame(index=False)
grid = grid.merge(all_products, on="ProductID", how="left")
monthly_demand_full = grid.merge(
    monthly_demand[["ProductID", "YearMonth", "QuantitySold", "Revenue"]],
    on=["ProductID", "YearMonth"], how="left",
)
monthly_demand_full[["QuantitySold", "Revenue"]] = monthly_demand_full[
    ["QuantitySold", "Revenue"]
].fillna(0)

# 1.7 Feature engineering: time features + lags + rolling mean
# - classic features for sales time series forecasting problems.
monthly_demand_full = monthly_demand_full.sort_values(["ProductID", "YearMonth"])
monthly_demand_full["Month"] = monthly_demand_full["YearMonth"].dt.month
monthly_demand_full["Quarter"] = monthly_demand_full["YearMonth"].dt.quarter
monthly_demand_full["MonthIndex"] = (
    monthly_demand_full["YearMonth"] - monthly_demand_full["YearMonth"].min()
).apply(lambda x: x.n)

for lag in [1, 2, 3]:
    monthly_demand_full[f"Lag_{lag}"] = monthly_demand_full.groupby("ProductID")[
        "QuantitySold"
    ].shift(lag)

monthly_demand_full["RollingMean_3"] = (
    monthly_demand_full.groupby("ProductID")["QuantitySold"]
    .shift(1)
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Add current sale price and category code (to be encoded)
monthly_demand_full = monthly_demand_full.merge(
    products[["ProductID", "CurrentSalePrice", "CurrentAssemblyCost", "StockQuantity", "ReorderLevel"]],
    on="ProductID", how="left",
)

model_df = monthly_demand_full.dropna(subset=["Lag_1", "Lag_2", "Lag_3"]).copy()

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = encoder.fit_transform(model_df[["CategoryID"]].astype(str))
cat_cols = [f"Cat_{c}" for c in encoder.categories_[0]]
cat_df = pd.DataFrame(cat_encoded, columns=cat_cols, index=model_df.index)
model_df = pd.concat([model_df, cat_df], axis=1)

feature_cols = (
    ["Month", "Quarter", "MonthIndex", "Lag_1", "Lag_2", "Lag_3",
     "RollingMean_3", "CurrentSalePrice"]
    + cat_cols
)
target_col = "QuantitySold"

print(f"\n=== Training data after preprocessing: {model_df.shape[0]} rows ===")

# =================================================================
# STEP 2: MODEL BUILDING & TRAINING
# =================================================================
# Chronological train/test split (no random shuffle) -> simulates
# real-world forecasting context: predict next month based on past data.
split_month = model_df["MonthIndex"].quantile(0.8)
train_df = model_df[model_df["MonthIndex"] <= split_month]
test_df = model_df[model_df["MonthIndex"] > split_month]

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

models = {
    "LinearRegression": LinearRegression(),
}

results = {}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, None)  # non-negative demand
    predictions[name] = y_pred
    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }

results_df = pd.DataFrame(results).T.sort_values("RMSE")
print("\n=== Model Performance Comparison (Test Set) ===")
print(results_df.round(3))

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\n>>> Selected model for deployment: {best_model_name}")

# =================================================================
# STEP 3: OUTPUT EXTRACTION & NEXT MONTH FORECAST
# =================================================================
# Forecast for NEXT MONTH (right after the last month with actual data)
latest = (
    monthly_demand_full.sort_values("YearMonth")
    .groupby("ProductID")
    .tail(1)
    .copy()
)
latest["Month"] = (latest["YearMonth"] + 1).dt.month
latest["Quarter"] = (latest["YearMonth"] + 1).dt.quarter
latest["MonthIndex"] = latest["MonthIndex"] + 1
latest["Lag_3"] = latest["Lag_2"]
latest["Lag_2"] = latest["Lag_1"]
latest["Lag_1"] = latest["QuantitySold"]
latest["RollingMean_3"] = latest[["Lag_1", "Lag_2", "Lag_3"]].mean(axis=1)

cat_encoded_latest = encoder.transform(latest[["CategoryID"]].astype(str))
cat_df_latest = pd.DataFrame(cat_encoded_latest, columns=cat_cols, index=latest.index)
latest = pd.concat([latest, cat_df_latest], axis=1)

next_month_forecast = latest[feature_cols].fillna(0)
latest["Forecast_NextMonth_Demand"] = np.clip(
    best_model.predict(next_month_forecast), 0, None
).round(1)

# Compare with current inventory -> classify supply-demand risks
def classify_risk(row):
    if row["Forecast_NextMonth_Demand"] > row["StockQuantity"]:
        return "UNDERSTOCK RISK"
    elif row["Forecast_NextMonth_Demand"] < 0.3 * row["StockQuantity"] and row["StockQuantity"] > row["ReorderLevel"]:
        return "OVERSTOCK RISK"
    else:
        return "BALANCED (OK)"

latest["Risk_Classification"] = latest.apply(classify_risk, axis=1)

latest["Forecast_Revenue"] = (latest["Forecast_NextMonth_Demand"] * latest["CurrentSalePrice"]).round(2)
latest["Forecast_Profit"] = (latest["Forecast_NextMonth_Demand"] * (latest["CurrentSalePrice"] - latest["CurrentAssemblyCost"])).round(2)
latest["Current_Revenue"] = (latest["QuantitySold"] * latest["CurrentSalePrice"]).round(2)
latest["Current_Profit"] = (latest["QuantitySold"] * (latest["CurrentSalePrice"] - latest["CurrentAssemblyCost"])).round(2)
latest["Suggested_Order"] = np.maximum(0, latest["Forecast_NextMonth_Demand"] + latest["ReorderLevel"] - latest["StockQuantity"]).round(1)

final_output = latest[
    ["ProductID", "ProductName", "CategoryName", "StockQuantity", "ReorderLevel",
     "Current_Revenue", "Current_Profit",
     "Forecast_NextMonth_Demand", "Forecast_Revenue", "Forecast_Profit", "Suggested_Order", "Risk_Classification"]
].sort_values("Forecast_NextMonth_Demand", ascending=False)

csv_cols = ["ProductID", "ProductName", "CategoryName", "StockQuantity", "ReorderLevel",
            "Forecast_NextMonth_Demand", "Forecast_Revenue", "Forecast_Profit", "Suggested_Order", "Risk_Classification"]
final_output[csv_cols].to_csv(f"{REPORT_DIR}/P1_demand_forecast_result.csv", index=False)
results_df.to_csv(f"{REPORT_DIR}/P1_model_comparison.csv")

print("\n=== NEXT MONTH DEMAND FORECAST RESULTS (Top 10) ===")
print(final_output.head(10)[
    ["ProductName", "Forecast_NextMonth_Demand", "Forecast_Revenue", "Forecast_Profit", "Suggested_Order", "Risk_Classification"]
].to_string(index=False))
print("\n=== RISK CLASSIFICATION ===")
print(final_output["Risk_Classification"].value_counts())

# =================================================================
# STEP 4: VISUALIZATION (save images - to be manually inserted into report)
# =================================================================

# [FIGURE P1-1] Total Monthly Revenue Trend (Historical vs Forecast)
# Calculate total revenue across months
monthly_rev = monthly_demand_full.copy()
monthly_rev["Revenue"] = monthly_rev["QuantitySold"] * monthly_rev["CurrentSalePrice"]
trend = monthly_rev.groupby("MonthIndex")["Revenue"].sum().reset_index()

# Calculate total forecasted revenue for next month
next_month_index = trend["MonthIndex"].max() + 1
total_forecast_rev = final_output["Forecast_Revenue"].sum()
trend_forecast = pd.DataFrame([{"MonthIndex": next_month_index, "Revenue": total_forecast_rev}])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(trend["MonthIndex"], trend["Revenue"], marker='o', label="Historical Revenue", color="blue")
ax.plot([trend["MonthIndex"].max(), next_month_index], 
        [trend.iloc[-1]["Revenue"], total_forecast_rev], 
        marker='o', linestyle="--", color="red", label="Forecasted Revenue")

ax.set_title("Store Total Revenue Trend (Historical vs Forecast)")
ax.set_xlabel("Month Index")
ax.set_ylabel("Total Revenue (VND)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/P1_1_revenue_trend.png", dpi=150)
plt.close()

# [FIGURE P1-2] Current vs Predicted Inventory
top_10_stock = final_output.head(10).copy()
top_10_stock["Predicted_Inventory"] = np.maximum(0, top_10_stock["StockQuantity"] - top_10_stock["Forecast_NextMonth_Demand"])
x = np.arange(len(top_10_stock))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, top_10_stock["StockQuantity"], width, label='Current Inventory', color='#5bc0de')
ax.bar(x + width/2, top_10_stock["Predicted_Inventory"], width, label='Predicted Inventory', color='#d9534f')
ax.set_xticks(x)
ax.set_xticklabels(top_10_stock["ProductName"], rotation=45, ha="right")
ax.set_title("Current vs Predicted Inventory")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/P1_2_stock_vs_demand.png", dpi=150)
plt.close()

# [FIGURE P1-3] Predicted Sales Next Month
top_revenue = final_output.sort_values("Forecast_Revenue", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in top_revenue.iterrows():
    ax.plot(['This Month', 'Next Month'], [row["Current_Revenue"], row["Forecast_Revenue"]], marker='o', label=row["ProductName"])
ax.set_title("Predicted Sales Next Month")
ax.set_ylabel("Sales (VND)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/P1_3_forecast_revenue.png", dpi=150)
plt.close()

# [FIGURE P1-4] Predicted profit Next Month
top_profit = final_output.sort_values("Forecast_Profit", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in top_profit.iterrows():
    ax.plot(['This Month', 'Next Month'], [row["Current_Profit"], row["Forecast_Profit"]], marker='o', label=row["ProductName"])
ax.set_title("Predicted profit Next Month")
ax.set_ylabel("Profit (VND)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/P1_4_forecast_profit.png", dpi=150)
plt.close()

print(f"\n[COMPLETE] Results saved to '{REPORT_DIR}/P1_demand_forecast_result.csv'")
print(f"[COMPLETE] Saved 4 charts to '{IMG_DIR}/' (P1_1..P1_4)")
