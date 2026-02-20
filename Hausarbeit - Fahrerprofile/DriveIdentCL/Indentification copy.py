import numpy as np
import pandas as pd
import questionary
import featuretools as ft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from woodwork.logical_types import Categorical, Double
from tqdm import tqdm
from hmmlearn.hmm import GaussianHMM

import warnings
#warnings.filterwarnings("ignore", message="Could not infer format")
warnings.filterwarnings("ignore", category=FutureWarning, module="featuretools")
# 1. import CSVs
#   1a. prepare Data
#       pd.to_numeric, errors="coerce", drop([0,1], interpolate(), dropna(axis=1, how="all")   # Dropping row 0 (Text) and row 1 (nan / zero values)
#       remove columns with constant values
#       add session index to data  
#   1b. Establish common ground 
#       drop unshared columns
#       abort if no columns remain
# 2. Expand Data
#   Aggregate all datasets the same way
# 3. Identify optimal features for identification
#   an optimal feature has a high inter-driver variance and a low intra-driver variance
#   3a. calculate feature scores
#   3b. select k best features --> K has to be a user input  
# 4. Validate (optional --> ask user)
#   4a. Test algorithm by identifying an already known dataset --> has to be 100% success
#   4b. Test training data by cross-validating (leave one out strategy)
# 5. identify samples
#   using nearest centroid
# 6. Evaluate and Display results in a informative way
#   Maybe use diagrams?

def identifyDriversCLI(datasets : list):
    input("Drücke ENTER um mit der Identifikation zu beginnen...")
    print("Starte Import der CSV-Dateien")
    sessions, data = loadRecordings(datasets)

    print("Starte Aggregation der Features")
    aggMatrix, aggDef = expandFeatures(sessions, data)
    
    print("Starte automatische Auswahl optimaler Features")
    k = int(questionary.text("Bitte gib die Anzahl (k) der besten Features an, die zur Identifikation genutzt werden sollen: ", validate=lambda text: text.isdigit()).ask())
    samples = aggMatrix[aggMatrix["driver"] == ""]
    trainData = aggMatrix[aggMatrix["driver"] != ""]
    featureScores = scoreFeatures(trainData)
    bestFeatures = featureScores.head(k)["feature"].tolist()
    print("Folgende Features wurden ausgewählt: ")
    print(bestFeatures)

    doValidate = questionary.confirm("Soll eine Validierung der Features und Trainingsdaten durchgeführt werden?").ask()
    if doValidate:
        print("Starte Validierung")
        # TODO Validierung

    print("Vorbereitung abgeschlossen")
    print("Starte Identifikation der Samples")
    return identify(trainData, samples, bestFeatures)


def loadRecordings(datasets : list) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = []
    data = []
    for entry in tqdm(datasets, desc="Importiere CSV-Dateien..."):
        df = pd.read_csv(entry["path"], low_memory=False)
        sessionId = str(f"{entry['driver']}_{entry['run']}")
        
        df = df.apply(pd.to_numeric, errors="coerce").drop([0,1]).interpolate().dropna(axis=1, how="all")   # Dropping row 0 (Text) and row 1 (nan / zero values)
        df = df.loc[:, (df != df.iloc[0]).any()]                                                            # Dropping constant columns values
        df["sessionId"] = sessionId
        
        data.append(df)
        sessions.append({
            "sessionId": sessionId,
            "driver": entry["driver"],
            "run": entry["run"]
        })

        # Establishing common ground --> dropping all columns not shared across every DataFrame in data
        commonColumns = set(data[0].columns)
        for df in data[1:]:
            commonColumns &= set(df.columns)
        data = [df.loc[:, sorted(commonColumns)] for df in data]

    print("Import abgeschlossen.")
    return pd.DataFrame(sessions), pd.concat(data, ignore_index=True)


def expandFeatures(sessions: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    sessions = sessions.copy()
    data = data.copy()

    sessions["sessionId"] = sessions["sessionId"].astype("category")
    data["sessionId"] = data["sessionId"].astype("category")
    data["timestamp"] = data["timestamp"].astype(float)

    es = ft.EntitySet(id="driver_analysis")

    es = es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions,
        index="sessionId",
        logical_types={
            "sessionId": Categorical,
            "driver": Categorical,
            "run": Categorical
        }
    )

    es = es.add_dataframe(
        dataframe_name="telemetry",
        dataframe=data,
        index="telemetryId",
        make_index=True,
        logical_types={
            "timestamp": Double,
            "sessionId": Categorical
        }
    )

    es = es.add_relationship(
        parent_dataframe_name="sessions",
        parent_column_name="sessionId",
        child_dataframe_name="telemetry",
        child_column_name="sessionId"
    )

    featureMatrix, featureDefs = ft.dfs(
        entityset=es,
        target_dataframe_name="sessions",
        ignore_columns={
            "telemetry": ["driver", "run"]
        },
        agg_primitives=["mean", "std", "min", "max", "skew", "count"],
        trans_primitives=["diff", "absolute"],
        max_depth=2,
        verbose=True
    )

    featureMatrix = featureMatrix.dropna(axis=1)
    print("Feature Aggregation abgeschlossen.")

    return featureMatrix, featureDefs


def scoreFeatures(data: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:

    excludedFeatures = ["driver", "run"]

    featureCols = [
        c for c in data.columns
        if c not in excludedFeatures
    ]

    results = []

    for feature in featureCols:
        values = data[excludedFeatures + [feature]].dropna()

        if values.empty:
            continue

        intraStds = []
        driverMeans = []

        for driver, g in values.groupby("driver", observed=True):
            driverMeans.append(g[feature].mean())

            if len(g) >= 2:
                intraStds.append(g[feature].std())

        # Mindestens 2 Fahrer nötig für Inter-Varianz
        if len(driverMeans) < 2:
            continue

        interVar = np.std(driverMeans)

        # Falls keine Intra-Varianz berechenbar --> 0
        intraVar = np.mean(intraStds) if intraStds else 0.0

        score = interVar / (interVar + intraVar + eps)

        results.append({
            "feature": feature,
            "score": score,
            "interDriverStd": interVar,
            "intraDriverStd": intraVar,
            "nDrivers": len(driverMeans),
            "nIntraDrivers": len(intraStds)
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    return df.sort_values("score", ascending=False).reset_index(drop=True)



def identify(trainData : pd.DataFrame, samples : pd.DataFrame, features : list[str]) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    XTrain = trainData.loc[:, features].values              # Nur k beste Features und bekannte Driver in den Trainingsdatensatz einfließen lassen
    yTrain = trainData["driver"].astype(str).to_numpy()     # Bekannte Driver (Y) des Trainingsdatensatzes auswählen
    XSamples = samples.loc[:, features].values

    # Skalierung
    scaler = StandardScaler()
    XTrain = scaler.fit_transform(XTrain)
    XSamples = scaler.transform(XSamples)

    # Zentroiden
    centroids = {
        driver: XTrain[yTrain == driver].mean(axis=0)
        for driver in np.unique(yTrain)
    }

    # Klassifikation per nächstem Zentroid
    distances = computeDistancesToCentroids(XSamples, centroids)
    predictions = classifyWithConfidence(distances)
    compactness = driverCompactness(XTrain, yTrain, centroids)
    centroidDists = centroidDistanceMatrix(centroids)

    print(distances)
    print(predictions)
    print(compactness)
    print(centroidDists)

    # Statistiken
    #   match score
    #   margin
    #   relative distance
    #   confidence
    #   reject-option
    #   intra driver compactness
    #   inter driver distance
    #   Diagrammme  
    #   --> Driver Distance
    #   --> Margin Histogramm
    #   --> PCA-Plot

    print("Identifikation abgeschlossen")
    return predictions, distances, compactness, centroidDists

def computeDistancesToCentroids(XSamples, centroids: dict[str, np.ndarray]):
    return {
        driver: np.linalg.norm(XSamples - centroid, axis=1)
        for driver, centroid in centroids.items()
    }

def classifyWithConfidence(distances: dict[str, np.ndarray], rejectionThreshold: float = 0.3):
    drivers = list(distances.keys())
    nSamples = len(next(iter(distances.values())))

    results = []

    for i in range(nSamples):
        dists = {d: distances[d][i] for d in drivers}
        sortedItems = sorted(dists.items(), key=lambda x: x[1])

        bestDriver, bestDist = sortedItems[0]
        secondDist = sortedItems[1][1] if len(sortedItems) > 1 else np.inf

        margin = secondDist - bestDist
        confidence = margin / (secondDist + 1e-6)

        prediction = (
            bestDriver if confidence >= rejectionThreshold else "UNKNOWN"
        )

        results.append({
            "sampleId": i,
            "predictedDriver": prediction,
            "bestDriver": bestDriver,
            "bestDistance": bestDist,
            "margin": margin,
            "confidence": confidence
        })

    return pd.DataFrame(results)

def driverCompactness(XTrain, yTrain, centroids):
    stats = []

    for driver, centroid in centroids.items():
        mask = yTrain == driver
        distances = np.linalg.norm(XTrain[mask] - centroid, axis=1)

        stats.append({
            "driver": driver,
            "meanDistance": distances.mean(),
            "stdDistance": distances.std(),
            "nSamples": mask.sum()
        })

    return pd.DataFrame(stats)

def centroidDistanceMatrix(centroids: dict[str, np.ndarray]) -> pd.DataFrame:
    drivers = list(centroids.keys())
    mat = np.zeros((len(drivers), len(drivers)))

    for i, d1 in enumerate(drivers):
        for j, d2 in enumerate(drivers):
            mat[i, j] = np.linalg.norm(centroids[d1] - centroids[d2])

    return pd.DataFrame(mat, index=drivers, columns=drivers)



def identifyDriversHMMCLI(datasets: list):

    input("Drücke ENTER um mit der Identifikation zu beginnen...")

    print("Starte Import der CSV-Dateien")
    sessions, data = loadRecordings(datasets)

    print("Starte Erweiterung der Features")
    seq = buildSequenceFeatures(data, ["car0_velocity", "throttle", "brakes"])

    print("Vorbereitung abgeschlossen")
    print("Starte Identifikation der Samples")

    predictions, models = identifyHMM(seq)

    print(predictions)
    return predictions


def buildSequenceFeatures(df: pd.DataFrame, baseCols: list[str]):

    df = df.copy()

    for col in baseCols:
        df[f"d_{col}"] = df.groupby("sessionId")[col].diff().fillna(0)
        df[f"abs_d_{col}"] = df[f"d_{col}"].abs()

    return df


def trainHMMPerDriver(sequencesByDriver):
    models = {}

    for driver, seqList in sequencesByDriver.items():
        X = np.vstack(seqList)
        lengths = [len(seq) for seq in seqList]

        hmm = GaussianHMM(n_components=5)
        hmm.fit(X, lengths)

        models[driver] = hmm

    return models

def identifyHMM(seq: pd.DataFrame,
                baseFeatures=["car0_velocity", "throttle", "brakes"],
                nStates: int = 5):
    """
    Identifiziert Fahrer mit einem Gaussian HMM pro Fahrer.

    seq: DataFrame mit Telemetrie + driver + sessionId
    baseFeatures: Rohfeatures
    nStates: Anzahl Hidden States

    Returns:
        predictions: DataFrame mit predictedDriver pro Sample-Session
        models: dict(driver -> trained HMM)
    """

    # -----------------------------
    # 1) Feature Columns definieren
    # -----------------------------
    featureCols = (
        baseFeatures
        + [f"d_{c}" for c in baseFeatures]
        + [f"abs_d_{c}" for c in baseFeatures]
    )

    # -----------------------------
    # 2) Train vs Samples trennen
    # -----------------------------
    trainData = seq[seq["driver"] != ""].copy()
    samples = seq[seq["driver"] == ""].copy()

    if trainData.empty:
        raise ValueError("Kein Trainingsdatensatz vorhanden!")

    if samples.empty:
        raise ValueError("Keine Samples zur Identifikation vorhanden!")

    # -----------------------------
    # 3) Skalierung fitten
    # -----------------------------
    scaler = StandardScaler()
    scaler.fit(trainData[featureCols])

    trainData[featureCols] = scaler.transform(trainData[featureCols])
    samples[featureCols] = scaler.transform(samples[featureCols])

    # -----------------------------
    # 4) Sequenzen pro Fahrer bauen
    # -----------------------------
    sequencesByDriver = {}

    for driver, gDriver in trainData.groupby("driver"):

        seqList = []

        for sessionId, gSession in gDriver.groupby("sessionId"):
            X = gSession[featureCols].values
            seqList.append(X)

        sequencesByDriver[driver] = seqList

    # -----------------------------
    # 5) Trainiere HMM pro Fahrer
    # -----------------------------
    models = trainHMMPerDriver(sequencesByDriver)

    print("HMM Training abgeschlossen.")

    # -----------------------------
    # 6) Identifikation pro Sample-Session
    # -----------------------------
    results = []

    for sessionId, gSession in samples.groupby("sessionId"):

        Xseq = gSession[featureCols].values

        # Likelihood pro Fahrer
        scores = {
            driver: model.score(Xseq)
            for driver, model in models.items()
        }

        # Best Driver auswählen
        if not scores:
            raise ValueError("Keine Modelle verfügbar!")

        bestDriver, bestScore = max(
            scores.items(),
            key=lambda item: item[1]
        )

        # Margin für Confidence
        sortedScores = sorted(scores.values(), reverse=True)
        secondScore = sortedScores[1] if len(sortedScores) > 1 else -np.inf

        margin = bestScore - secondScore

        results.append({
            "sessionId": sessionId,
            "predictedDriver": bestDriver,
            "logLikelihood": bestScore,
            "margin": margin
        })

    predictions = pd.DataFrame(results)

    print("Identifikation abgeschlossen.")
    return predictions, models