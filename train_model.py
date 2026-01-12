import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Load dataset (Kaggle SDSS)
# ==============================
df = pd.read_csv("star_classification.csv")

# Normalize class labels
df["class"] = df["class"].str.upper()

# ==============================
# OPTIONAL: Subsample for deployable model size
# ==============================
MAX_TRAIN_SAMPLES = 150_000

if len(df) > MAX_TRAIN_SAMPLES:
    df = df.sample(MAX_TRAIN_SAMPLES, random_state=42)
# ==============================
# Create color indices
# ==============================
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

FEATURES = ["u_g", "g_r", "r_i", "i_z"]

X = df[FEATURES]

# ==============================
# Manual label encoding (SAFE)
# ==============================
CLASS_NAMES = ["GALAXY", "QSO", "STAR"]
class_to_int = {c: i for i, c in enumerate(CLASS_NAMES)}
int_to_class = {i: c for c, i in class_to_int.items()}

y = df["class"].map(class_to_int)

# ==============================
# Train / test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Scale features
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Train RANDOM FOREST (SIZE-CONTROLLED)
# ==============================
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# ==============================
# Evaluation
# ==============================
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.5f}\n")

print("Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES
))

# ==============================
# Save artifacts (DEPLOYABLE)
# ==============================
joblib.dump(model, "rf_star_classifier.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")

print("\n✅ Model and preprocessing artifacts saved successfully.")
