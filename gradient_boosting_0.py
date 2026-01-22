"""
Fahrer-Erkennung mit Gradient Boosting
-------------------------------------
- Lädt Fahrsimulator-Daten aus CSV
- Trainiert ein Gradient-Boosting-Modell
- Führt saubere Cross-Validation durch
- Gibt Klassifikationsmetriken aus

Abhängigkeiten:
pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


# ============================================================
# 1. Konfiguration
# ============================================================

CSV_PATH = r"C:\Users\Nutzer\Desktop\Meins\Studium\HochschuleHarz\Informatik\Semester9\ProgrammierenMobilerSysteme\JointRecordingsAggregations.csv"
TARGET_COLUMN = "driver_id"      # Fahrer-Label
GROUP_COLUMN = "session_id"      # optional (falls vorhanden)
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# 2. Daten laden
# ============================================================

df = pd.read_csv(CSV_PATH)
print("Daten geladen:", df.shape)


# ============================================================
# 3. Feature- & Label-Trennung
# ============================================================

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Zielspalte '{TARGET_COLUMN}' nicht in CSV gefunden!")

y = df[TARGET_COLUMN]

# Nur numerische Features verwenden
X = df.select_dtypes(include=[np.number]).drop(
    columns=[TARGET_COLUMN], errors="ignore"
)

print("Anzahl Features:", X.shape[1])
print("Anzahl Fahrer:", y.nunique())


# ============================================================
# 4. In NumPy umwandeln (robust!)
# ============================================================

X_values = X.to_numpy()
y_values = y.to_numpy()


# ============================================================
# 5. Cross-Validation-Strategie wählen
# ============================================================

if GROUP_COLUMN in df.columns:
    print("Verwende GroupKFold (sessionsicher)")
    groups_values = df[GROUP_COLUMN].to_numpy()
    cv = GroupKFold(n_splits=N_SPLITS)
    split_iterator = cv.split(X_values, y_values, groups_values)
else:
    print("Verwende StratifiedKFold (keine Sessions vorhanden)")
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    split_iterator = cv.split(X_values, y_values)


# ============================================================
# 6. Pipeline: Scaling + Gradient Boosting
# ============================================================

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        ))
    ]
)


# ============================================================
# 7. Cross-Validation durchführen
# ============================================================

y_true_all = []
y_pred_all = []

print("\nStarte Cross-Validation...")

for fold, (train_idx, test_idx) in enumerate(split_iterator):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X_values[train_idx], X_values[test_idx]
    y_train, y_test = y_values[train_idx], y_values[test_idx]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)


# ============================================================
# 8. Evaluation
# ============================================================

print("\n=== Klassifikationsbericht ===")
print(classification_report(y_true_all, y_pred_all))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true_all, y_pred_all))


# ============================================================
# 9. Feature Importance
# ============================================================

gb_model = pipeline.named_steps["gb"]

feature_importance = pd.Series(
    gb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n=== Top 15 wichtigste Features ===")
print(feature_importance.head(15))


# ============================================================
# 10. Optional: Modell speichern
# ============================================================

# from joblib import dump
# dump(pipeline, "gradient_boosting_driver_model.joblib")
