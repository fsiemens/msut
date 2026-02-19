

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pickle import TRUE


# ============================================================
# 1. CSV EINLESEN & BEREINIGEN
# ============================================================

def load_and_prepare(csv_path):
    """
    Lädt eine Simulatoraufnahme und bereitet sie vor.
    """

    df = pd.read_csv(csv_path)

    # Nur numerische Sensorwerte behalten
    df = df.select_dtypes(include=[np.number])

    # Fehlende Werte ersetzen
    df = df.fillna(df.median())

    # Ausreißer begrenzen (robuster gegen Sensorfehler)
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)

    return df


# ============================================================
# 2. DATEN AUS ORDNER LADEN
# ============================================================

def load_dataset(folder_path):
    """
    Lädt alle CSV-Dateien und erstellt Trainingsdaten.

    Fahrername wird aus Dateiname extrahiert:
    fabian_01.csv entspricht also Fabian Siemens
    """

    X_list = []
    y_list = []

    drivers = set()

    for file in os.listdir(folder_path):

        if not file.endswith(".csv"):
            continue

        driver_name = file.split("_")[0]   # Fahrer erkennen
        drivers.add(driver_name)

        full_path = os.path.join(folder_path, file)
        print(f"Lade {file}, also Fahrer: {driver_name}")

        df = load_and_prepare(full_path)

        X_list.append(df)
        y_list.append(np.full(len(df), driver_name))

    # gemeinsame Sensoren bestimmen
    common_cols = set(X_list[0].columns)
    for df in X_list[1:]:
        common_cols &= set(df.columns)

    common_cols = list(common_cols)

    print(f"\nGemeinsame Sensor-Spalten: {len(common_cols)}")

    X_list = [df[common_cols] for df in X_list]

    X = pd.concat(X_list, axis=0)
    y = np.concatenate(y_list)

    return X, y


# ============================================================
# 3. MODELL TRAINIEREN
# ============================================================

def train_model(data_folder, model_out="driver_model.pkl"):
    """
    Trainiert das Fahreridentifikationsmodell.
    """

    X, y = load_dataset(data_folder)

    print("\nNormalisiere Daten...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Trainings/Testaufteilung
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    print("\nTrainiere Gradient Boosting Modell...")

    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Bewertung
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== MODELLERGEBNIS ==========")
    print(f"Gesamtgenauigkeit: {accuracy:.3f}\n")

    print("Klassifikationsbericht:")
    print(classification_report(y_test, y_pred))

    # Modell speichern
    joblib.dump((model, scaler, X.columns.tolist()), model_out)

    print(f"\nModell gespeichert als: {model_out}")


# ============================================================
# 4. NEUE FAHRT IDENTIFIZIEREN
# ============================================================

def identify_driver(csv_file, model_file="driver_model.pkl"):
    """
    Bestimmt den Fahrer einer neuen Aufnahme.
    """

    model, scaler, train_columns = joblib.load(model_file)
    
    df = load_and_prepare(csv_file)
    
    # Fehlende Spalten ergänzen
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Zusätzliche Spalten entfernen
    df = df[train_columns]
    
    # gleiche Reihenfolge sicherstellen
    df = df[train_columns]
    
    # skalieren
    X = scaler.transform(df)


    # Wahrscheinlichkeiten je Fahrer
    probs = model.predict_proba(X)

    # Mittelwert über alle Frames
    mean_probs = probs.mean(axis=0)

    drivers = model.classes_

    print("\n========== FAHRERIDENTIFIKATION ==========")
    for driver, prob in zip(drivers, mean_probs):
        print(f"{driver}: {prob:.2%}")

    best_driver = drivers[np.argmax(mean_probs)]
    confidence = np.max(mean_probs)

    print("\nErkannt als:", best_driver)
    print(f"Sicherheit: {confidence:.2%}")

    if confidence < 0.5:
        print("Unbekannter Fahrer oder unsicheres Ergebnis")


# ============================================================
# 5. HAUPTPROGRAMM
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Fahrer-Identifikation")

    parser.add_argument("--train", help="Trainingsordner")
    parser.add_argument("--identify", help="CSV zur Fahreridentifikation")

    args = parser.parse_args()

    if args.train:
        train_model(args.train)

    if args.identify:
        identify_driver(args.identify)
