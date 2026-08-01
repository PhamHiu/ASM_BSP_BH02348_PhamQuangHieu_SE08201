"""
==========================================================
MODULE 3: AI-BASED PRODUCT QUALITY PREDICTION SYSTEM
Samsung Electronics – Galaxy Smartphone Production Line
==========================================================
AI System to predict and support product quality inspection
at Samsung Electronics.

Architecture: 6 steps (Step 1-6)
Input:  ProductionData.csv + QualityTests.csv
Output: 3 CSV + 3 PNG
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PATH CONFIGURATION
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "..", "DB_final", "Samsung_Store_Database")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
IMG_DIR = os.path.join(OUT_DIR, "images")
REPORT_DIR = os.path.join(OUT_DIR, "report_csv")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ==========================================
# STEP 1: DATA INGESTION
# ==========================================
print("=== STEP 1: DATA INGESTION ===")

prod_df = pd.read_csv(os.path.join(DB_DIR, "ProductionData.csv"))
qt_df = pd.read_csv(os.path.join(DB_DIR, "QualityTests.csv"))

# Merge 2 tables via ProductionID
df = pd.merge(prod_df, qt_df, on='ProductionID', how='inner')

total = len(df)
defect_count = (df['QualityStatus'] == 'Defect').sum()
pass_count = (df['QualityStatus'] == 'Pass').sum()

print(f"-> Data loaded and merged: {total} products")
print(f"-> Pass: {pass_count} ({pass_count/total*100:.1f}%) | Defect: {defect_count} ({defect_count/total*100:.1f}%)")


# ==========================================
# STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("\n=== STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING ===")

# 2.1 Categorical Encoding
# MachineType: Label Encoding (Type_01 → 1, Type_02 → 2, ...)
le_machine = LabelEncoder()
df['MachineType_Encoded'] = le_machine.fit_transform(df['MachineType'])

# ProductionShift: One-Hot Encoding
shift_dummies = pd.get_dummies(df['ProductionShift'], prefix='Shift')
df = pd.concat([df, shift_dummies], axis=1)

# 2.2 Create Derived Features
df['AvgTestScore'] = (df['ScreenTestScore'] + df['BatteryTestScore'] +
                      df['CameraTestScore'] + df['PerformanceTestScore']) / 4

df['MinTestScore'] = df[['ScreenTestScore', 'BatteryTestScore',
                          'CameraTestScore', 'PerformanceTestScore']].min(axis=1)

# 2.3 Define Features (X) and Target (y)
feature_cols = [
    # Production context
    'MachineType_Encoded', 'ProductionSpeed_UPH',
    # Product physical metrics during test
    'Product_Temp_C', 'Product_Pressure_PSI',
    # 4 component test scores
    'ScreenTestScore', 'BatteryTestScore', 'CameraTestScore', 'PerformanceTestScore',
    # Derived features
    'AvgTestScore', 'MinTestScore',
]

# Add One-Hot shift columns
shift_cols = [c for c in df.columns if c.startswith('Shift_')]
feature_cols += shift_cols

X = df[feature_cols]
y = df['QualityStatus'].map({'Pass': 0, 'Defect': 1})  # Encode 0/1 for AI

print(f"-> Number of input features: {len(feature_cols)}")
print(f"-> Features: {feature_cols}")
print(f"-> Target: QualityStatus (Pass=0, Defect=1)")


# ==========================================
# STEP 3: TRAIN/TEST SPLIT
# ==========================================
print("\n=== STEP 3: TRAIN/TEST SPLIT ===")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"-> Training Set: {len(X_train)} products")
print(f"-> Test Set: {len(X_test)} products")
print(f"-> Defect rate in Training: {y_train.mean()*100:.1f}%")
print(f"-> Defect rate in Test: {y_test.mean()*100:.1f}%")


# ==========================================
# STEP 4: RANDOM FOREST MODEL TRAINING
# ==========================================
print("\n=== STEP 4: RANDOM FOREST MODEL TRAINING ===")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Defect (1)']))


# ==========================================
# STEP 5: MODEL EVALUATION & ANALYSIS
# ==========================================
print("\n=== STEP 5: MODEL EVALUATION ===")

# 5.1 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("--- Confusion Matrix ---")
print(f"   True Negative (TN): {tn}  |  False Positive (FP): {fp}")
print(f"   False Negative (FN): {fn}  |  True Positive (TP): {tp}")
print(f"\n-> AI correctly caught {tp} defective products out of {tp+fn} actual defects")
print(f"-> AI falsely alarmed on {fp} good products")
print(f"-> AI missed {fn} defective products")

# 5.2 Feature Importance
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance_Score': importances
}).sort_values('Importance_Score', ascending=False)

print("\n--- Top 10 Factors affecting Defects ---")
for i, row in feat_imp_df.head(10).iterrows():
    print(f"   {row['Feature']}: {row['Importance_Score']:.4f}")


# ==========================================
# STEP 6: REPORT EXPORT & VISUALIZATION
# ==========================================
print("\n=== STEP 6: REPORT EXPORT & VISUALIZATION ===")

# --- 6.1 CSV 1: Quality Prediction Report (all 10,000 products) ---
df['Defect_Probability'] = rf_model.predict_proba(X)[:, 1]

def get_risk_level(prob):
    if prob >= 0.7: return "High_Risk"
    elif prob >= 0.4: return "Needs_Inspection"
    else: return "Safe"

df['Risk_Level'] = df['Defect_Probability'].apply(get_risk_level)

report_cols = ['ProductionID', 'ProductID', 'MachineID', 'ProductionShift',
               'Product_Temp_C', 'Product_Pressure_PSI',
               'ScreenTestScore', 'BatteryTestScore', 'CameraTestScore', 'PerformanceTestScore',
               'QualityStatus', 'Defect_Probability', 'Risk_Level']
report_path = os.path.join(REPORT_DIR, "P3_Quality_Prediction_Report.csv")
df[report_cols].sort_values('Defect_Probability', ascending=False).to_csv(report_path, index=False)
print(f"-> CSV 1: Quality Prediction Report ({len(df)} products): {report_path}")

# --- 6.2 CSV 2: Root Cause Analysis ---
rc_path = os.path.join(REPORT_DIR, "P3_Root_Cause_Analysis.csv")
feat_imp_df.to_csv(rc_path, index=False)
print(f"-> CSV 2: Root Cause Analysis: {rc_path}")

# --- 6.3 CSV 3: Prediction vs Reality (Test Set) ---
compare_df = df.loc[X_test.index].copy()
compare_df['Actual_Result'] = y_test.map({0: 'Pass', 1: 'Defect'})
compare_df['AI_Prediction'] = pd.Series(y_pred, index=X_test.index).map({0: 'Pass', 1: 'Defect'})
compare_export = compare_df[['ProductionID', 'ProductID', 'MachineID', 'ProductionShift',
                              'Actual_Result', 'AI_Prediction', 'Defect_Probability']]
compare_path = os.path.join(REPORT_DIR, "P3_Prediction_vs_Reality.csv")
compare_export.to_csv(compare_path, index=False)
print(f"-> CSV 3: Prediction vs Reality ({len(compare_df)} products): {compare_path}")

# --- 6.4 VISUALIZATION ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Chart 1: Feature Importance (Horizontal Bar)
plt.figure(figsize=(10, 7))
top_features = feat_imp_df.head(12)
colors = ['#d62728' if imp > 0.1 else '#ff7f0e' if imp > 0.05 else '#1f77b4'
          for imp in top_features['Importance_Score']]
plt.barh(top_features['Feature'][::-1], top_features['Importance_Score'][::-1],
         color=colors[::-1], edgecolor='black', linewidth=0.5)
plt.title('Feature Importance for Quality Prediction (Gini)', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "P3_Feature_Importance.png"), dpi=150)
plt.close()

# Chart 2: Defect Risk Distribution (Histogram)
plt.figure(figsize=(10, 6))
plt.hist(df['Defect_Probability'], bins=20, color='#2ca02c', edgecolor='black', alpha=0.8)
plt.axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Inspection Threshold (0.4)')
plt.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Risk Threshold (0.7)')
plt.title('Product Defect Probability Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Defect Probability', fontsize=11)
plt.ylabel('Number of Products', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "P3_Defect_Risk_Distribution.png"), dpi=150)
plt.close()

# Chart 3: Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted: Pass', 'Predicted: Defect'], fontsize=11)
ax.set_yticklabels(['Actual: Pass', 'Actual: Defect'], fontsize=11)
ax.set_title('Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')

# Annotate cells
for i in range(2):
    for j in range(2):
        label = f"{cm[i, j]}"
        color = "white" if cm[i, j] > cm.max()/2 else "black"
        ax.text(j, i, label, ha="center", va="center", fontsize=18, fontweight='bold', color=color)

plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "P3_Confusion_Matrix_Heatmap.png"), dpi=150)
plt.close()

print(f"-> Exported 3 charts to: {IMG_DIR}")

# --- Summary ---
risk_counts = df['Risk_Level'].value_counts()
print("\n--- MODULE 3 SUMMARY ---")
print(f"Safe: {risk_counts.get('Safe', 0)} products | Needs_Inspection: {risk_counts.get('Needs_Inspection', 0)} products | High_Risk: {risk_counts.get('High_Risk', 0)} products")
print(f"Accuracy: {(y_pred == y_test).mean()*100:.1f}%")
print(f"Recall (Defect): {tp/(tp+fn)*100:.1f}%")
print(f"Precision (Defect): {tp/(tp+fp)*100:.1f}%")

print("\n[COMPLETE] Module 3 - AI Quality Prediction System finished successfully!")
