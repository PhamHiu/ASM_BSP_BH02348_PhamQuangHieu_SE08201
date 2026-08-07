import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

np.random.seed(42)

# ==========================================
# 1. DIRECTORY SETUP
# ==========================================
DATA_DIR = r"C:\Users\AQUA\Desktop\School\Part_IV\BusinessSP_Mr.Dong\ASM\DB_final\Samsung_Store_Database"
OUT_DIR = r"C:\Users\AQUA\Desktop\School\Part_IV\BusinessSP_Mr.Dong\ASM\python\outputs"
IMG_DIR = os.path.join(OUT_DIR, "images")
REPORT_DIR = os.path.join(OUT_DIR, "report_csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Manufacturer Specs table
specs_csv_path = os.path.join(DATA_DIR, "machine_specs.csv")
if not os.path.exists(specs_csv_path):
    raise FileNotFoundError(f"File not found: {specs_csv_path}")
machine_specs = pd.read_csv(specs_csv_path).set_index("MachineType").to_dict('index')

print("=== 1. CHECK/CREATE SENSOR DATA IN DATABASE ===")
csv_path = os.path.join(DATA_DIR, "IoT_Sensor_Data.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Data file not found at {csv_path}. Please check Database directory.")

# Load input data as source for Model (Matches problem structure)
df = pd.read_csv(csv_path)
print(f"-> Successfully loaded input data source: {df.shape[0]} rows.")


# ==========================================
# 2. PREPROCESSING & MODEL TRAINING
# ==========================================
print("\n=== 2. RANDOM FOREST TRAINING ===")
# Encode machine type variable (One-hot encoding)
df_encoded = pd.get_dummies(df, columns=['MachineType'])

features = [c for c in df_encoded.columns if c not in ['MachineID', 'Failure_Status']]
X = df_encoded[features]
y = df_encoded['Failure_Status']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))


# ==========================================
# 3. EXTRACT MACHINE STATUS REPORT
# ==========================================
print("\n=== 3. EXPORT STATUS REPORT ===")

# Get failure prediction probability on entire dataset
df['Predict_Failure_Prob'] = rf_model.predict_proba(X)[:, 1]

# --- Fix 2: Classify Risk Level into 4 tiers ---
def get_status(row):
    if row['Failure_Status'] == 1:
        return "Failed"
    elif row['Predict_Failure_Prob'] >= 0.8:
        return "Critical_Risk"
    elif row['Predict_Failure_Prob'] >= 0.6:
        return "Needs_Maintenance"
    else:
        return "Normal"

df['Current_Status'] = df.apply(get_status, axis=1)

# --- Calculate mean and standard deviation by machine type (Fleet Stats) ---
sensor_cols = ['Temperature_C', 'Vibration_mm_s', 'Voltage_V', 'Current_A',
               'Power_Consumption_kWh', 'Runtime_Hours']

fleet_stats = df.groupby('MachineType')[sensor_cols].agg(['mean', 'std']).reset_index()
fleet_stats.columns = ['MachineType'] + [f'{col}_{stat}' for col, stat in
                                          fleet_stats.columns[1:]]
fleet_lookup = fleet_stats.set_index('MachineType').to_dict('index')

# --- Fix 1: 2-tier Alert Reason System ---
alert_reasons = []
top_risk_factors = []

for _, row in df.iterrows():
    if row['Current_Status'] == "Normal":
        alert_reasons.append("Good")
        top_risk_factors.append("-")
        continue

    m_type = row['MachineType']
    spec = machine_specs.get(m_type)
    fleet = fleet_lookup.get(m_type, {})
    reasons = []
    anomaly_scores = {}  # To calculate Top Risk Factor

    # === Tier 1: Compare against Manufacturer Hard Specs ===
    if spec:
        if row['Temperature_C'] > spec['Max_Temp_C']:
            reasons.append(f"Temp {row['Temperature_C']:.1f}C > Spec Max {spec['Max_Temp_C']}")
        if row['Vibration_mm_s'] > spec['Max_Vib_mm_s']:
            reasons.append(f"Vib {row['Vibration_mm_s']:.2f} > Spec Max {spec['Max_Vib_mm_s']}")
        if row['Voltage_V'] > spec['Max_Voltage_V']:
            reasons.append(f"Volt {row['Voltage_V']:.1f} > Spec Max {spec['Max_Voltage_V']}")
        if row['Voltage_V'] < spec['Min_Voltage_V']:
            reasons.append(f"Volt {row['Voltage_V']:.1f} < Spec Min {spec['Min_Voltage_V']}")
        if row['Current_A'] > spec['Max_Current_A']:
            reasons.append(f"Curr {row['Current_A']:.1f} > Spec Max {spec['Max_Current_A']}")
        if row['Current_A'] < spec['Min_Current_A']:
            reasons.append(f"Curr {row['Current_A']:.1f} < Spec Min {spec['Min_Current_A']}")

    # === Tier 2: Compare against Fleet average (if Tier 1 not found) ===
    if not reasons and fleet:
        for col in sensor_cols:
            mean_key = f'{col}_mean'
            std_key = f'{col}_std'
            if mean_key in fleet and std_key in fleet:
                mean_val = fleet[mean_key]
                std_val = fleet[std_key] if fleet[std_key] > 0 else 1
                threshold = mean_val + 1.5 * std_val
                threshold_low = mean_val - 1.5 * std_val

                if row[col] > threshold:
                    z_score = (row[col] - mean_val) / std_val
                    reasons.append(f"{col} {row[col]:.1f} > fleet avg+1.5s ({threshold:.1f})")
                    anomaly_scores[col] = z_score
                elif row[col] < threshold_low and col in ['Voltage_V', 'Current_A']:
                    z_score = abs((row[col] - mean_val) / std_val)
                    reasons.append(f"{col} {row[col]:.1f} < fleet avg-1.5s ({threshold_low:.1f})")
                    anomaly_scores[col] = z_score

    # If still no reason, assign based on AI probability
    if not reasons:
        reasons.append(f"AI Prob {row['Predict_Failure_Prob']:.1%} - Combined sensor anomaly pattern")

    alert_reasons.append(" | ".join(reasons))

    # --- Fix 3: Top Risk Factor ---
    if anomaly_scores:
        top_factor = max(anomaly_scores, key=anomaly_scores.get)
        top_risk_factors.append(f"{top_factor} ({row[top_factor]:.1f}, z={anomaly_scores[top_factor]:.1f})")
    elif spec:
        # Calculate z-score based on distance to specs threshold
        spec_distances = {}
        if 'Max_Temp_C' in spec:
            spec_distances['Temperature_C'] = row['Temperature_C'] / spec['Max_Temp_C']
        if 'Max_Vib_mm_s' in spec:
            spec_distances['Vibration_mm_s'] = row['Vibration_mm_s'] / spec['Max_Vib_mm_s']
        if spec_distances:
            top_factor = max(spec_distances, key=spec_distances.get)
            top_risk_factors.append(f"{top_factor} ({row[top_factor]:.1f})")
        else:
            top_risk_factors.append("Combined_Pattern")
    else:
        top_risk_factors.append("Combined_Pattern")

df['Alert_Reason_vs_Specs'] = alert_reasons
df['Top_Risk_Factor'] = top_risk_factors

# Sort: Failed > Critical_Risk > Needs_Maintenance > Normal
status_order = {'Failed': 0, 'Critical_Risk': 1, 'Needs_Maintenance': 2, 'Normal': 3}
df['_sort'] = df['Current_Status'].map(status_order)
df = df.sort_values(by=['_sort', 'Predict_Failure_Prob'], ascending=[True, False])
df = df.drop(columns=['_sort'])

# Save to CSV
output_csv = os.path.join(REPORT_DIR, "P2_Machine_Status_Report.csv")
df.to_csv(output_csv, index=False)

# Statistics
print(f"-> Exported status report for {df.shape[0]} machines to file: {output_csv}")
print(f"   Failed: {(df['Current_Status']=='Failed').sum()}")
print(f"   Critical_Risk: {(df['Current_Status']=='Critical_Risk').sum()}")
print(f"   Needs_Maintenance: {(df['Current_Status']=='Needs_Maintenance').sum()}")
print(f"   Normal: {(df['Current_Status']=='Normal').sum()}")
print(f"   Multiple_Anomalies count: {(df['Alert_Reason_vs_Specs']=='Multiple_Anomalies').sum()}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("\n=== 4. VISUALIZATION ===")

import matplotlib
matplotlib.use('Agg') # Ensure no UI error display

# 4.1 Feature Importance Chart
importances = rf_model.feature_importances_
feature_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_imp = feature_imp.sort_values('Importance', ascending=True)

# Export Feature Importance data to CSV file for Power BI
csv_path_imp = os.path.join(REPORT_DIR, "P2_Feature_Importance.csv")
feature_imp.sort_values('Importance', ascending=False).to_csv(csv_path_imp, index=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_imp['Feature'], feature_imp['Importance'], color='#1f77b4')
plt.title('Feature Importance for Predictive Maintenance')
plt.xlabel('Gini Importance')
plt.ylabel('Sensor Feature')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "P2_Feature_Importance.png"), dpi=150)
plt.close()

# 4.2 Failure Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['Predict_Failure_Prob'], bins=15, color='#d62728', edgecolor='black')
plt.axvline(x=0.6, color='black', linestyle='--', linewidth=2, label='Warning Threshold (0.6)')
plt.title('Machine Failure Probability Distribution')
plt.xlabel('Predicted Probability of Failure')
plt.ylabel('Number of Machines')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "P2_Failure_Probability_Distribution.png"), dpi=150)
plt.close()

print(f"-> Exported 2 charts to directory: {IMG_DIR}")
print("\n[COMPLETE] Module 2 finished successfully.")
