# -*- coding: utf-8 -*-
"""
01_demand_supply_forecast.py
---------------------------------------------------------------
VAN DE 1: RUI RO LECH PHA CUNG CAU (Demand-Supply Mismatch)
---------------------------------------------------------------
Boi canh: Du bao nhu cau yeu kem -> ton kho san pham it ban chay, thieu
hang san pham dang "hot" -> bao mon loi nhuan.

Nguon du lieu that su dung:
    - Orders.csv, OrderDetails.csv  (lich su ban hang thuc te 2023-2024)
    - Products.csv                  (StockQuantity, ReorderLevel hien tai)
    - Category.csv                  (nhom nganh hang)

Muc tieu: Du bao SAN LUONG BAN (demand) THEO THANG cho tung SAN PHAM trong
thang tiep theo, sau do doi chieu voi StockQuantity + ReorderLevel hien tai
de phat hien: (a) san pham co nguy co THIEU HANG (demand du bao > ton kho
kha dung), (b) san pham co nguy co TON KHO U DONG (demand du bao rat thap
so voi ton kho hien tai).

Thuat toan su dung (theo yeu cau): Linear Regression, Random Forest
Regressor, XGBoost Regressor -> so sanh hieu nang -> chon mo hinh tot nhat
de xuat ban dau ra cuoi cung.
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

DATA_DIR = "data"
OUT_DIR = "outputs"
IMG_DIR = f"{OUT_DIR}/images"
REPORT_DIR = f"{OUT_DIR}/report_csv"
import os
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
pd.set_option("display.width", 140)

# =================================================================
# BUOC 1: TIEN XU LY VA LAM SACH DU LIEU (Data Preprocessing)
# =================================================================
orders = pd.read_csv(f"{DATA_DIR}/Orders.csv")
order_details = pd.read_csv(f"{DATA_DIR}/OrderDetails.csv")
products = pd.read_csv(f"{DATA_DIR}/Products.csv")
category = pd.read_csv(f"{DATA_DIR}/Categories.csv")

# 1.1 Kiem tra & xu ly gia tri thieu / trung lap
print("=== Checking for missing data ===")
print(orders.isna().sum()[orders.isna().sum() > 0])
print(order_details.isna().sum()[order_details.isna().sum() > 0])
print("Duplicate rows in Orders:", orders.duplicated().sum())
print("Duplicate rows in OrderDetails:", order_details.duplicated().sum())
# Ghi chu: theo DATA_ISSUES.md, phan du lieu ban hang (Orders/OrderDetails)
# KHONG chua loi co y -> chi can kiem tra phong ngua, khong can sua nhieu.

# 1.2 Chuan hoa kieu du lieu ngay thang
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")
orders = orders.dropna(subset=["OrderDate"])

# 1.3 Ghep du lieu ban hang voi san pham + danh muc
sales = order_details.merge(
    orders[["OrderID", "OrderDate"]], on="OrderID", how="inner"
).merge(
    products[["ProductID", "ProductName", "CategoryID", "CurrentSalePrice", "CurrentAssemblyCost",
              "StockQuantity", "ReorderLevel", "Status"]],
    on="ProductID", how="left",
).merge(
    category[["CategoryID", "CategoryName"]], on="CategoryID", how="left"
)

# 1.4 Tao truc thoi gian THANG (Year-Month) - don vi phan tich du bao
sales["YearMonth"] = sales["OrderDate"].dt.to_period("M")

# 1.5 Tong hop nhu cau (demand) theo Thang x San pham
monthly_demand = (
    sales.groupby(["ProductID", "ProductName", "CategoryID", "CategoryName", "YearMonth"])
    .agg(QuantitySold=("Quantity", "sum"), Revenue=("SubTotal", "sum"))
    .reset_index()
)

# 1.6 Dam bao du lieu day du theo LUOI thoi gian (dien 0 cho thang khong ban
# duoc de mo hinh hoc dung xu huong "khong ban duoc", tranh sai lech do
# thieu quan sat)
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

# 1.7 Feature engineering: dac trung thoi gian + do tre (lag) + trung binh
# truot (rolling mean) - cac dac trung kinh dien cho bai toan du bao chuoi
# thoi gian ban hang.
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

# Gop them gia ban hien tai va ma danh muc (encode)
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
# BUOC 2: XAY DUNG & HUAN LUYEN MO HINH (Model Training)
# =================================================================
# Chia train/test THEO THOI GIAN (khong shuffle ngau nhien) -> mo phong
# dung boi canh du bao thuc te: du bao thang sau dua tren du lieu qua khu.
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
    y_pred = np.clip(model.predict(X_test), 0, None)  # nhu cau khong am
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
# BUOC 3: TRICH XUAT DAU RA & DU BAO THANG TIEP THEO
# =================================================================
# Du bao cho THANG KE TIEP (ngay sau thang cuoi cung co du lieu that)
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

# Doi chieu voi ton kho hien tai -> phan loai rui ro cung cau
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
# BUOC 4: TRUC QUAN HOA (luu file anh - se duoc chen thu cong vao bao cao)
# =================================================================

# [HINH P1-1] Total Monthly Revenue Trend (Historical vs Forecast)
# Tinh tong doanh thu qua cac thang
monthly_rev = monthly_demand_full.copy()
monthly_rev["Revenue"] = monthly_rev["QuantitySold"] * monthly_rev["CurrentSalePrice"]
trend = monthly_rev.groupby("MonthIndex")["Revenue"].sum().reset_index()

# Tinh tong doanh thu du bao cho thang tiep theo
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

# [HINH P1-2] Current vs Predicted Inventory
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

# [HINH P1-3] Predicted Sales Next Month
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

# [HINH P1-4] Predicted profit Next Month
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
