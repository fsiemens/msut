import pandas as pd
import questionary
from OsInteractions import clearTerminal, askForFiles, saveFile
from Indentification import identifyDriversCLI
import matplotlib.pyplot as plt
import numpy as np

def main():
    clearTerminal()
    print("===============================")
    print("| Willkommen zu DriveIdentCL! |")
    print("===============================\n")

    filePaths = selectTrainingFiles()
    if filePaths is None: return
    
    labels = labelTrainingData(filePaths)
    if labels is None: return

    print("Folgende Labels wurden vergeben: ")
    print(labels[['filename', 'driver']])
    print("")

    samples = selectSampleFiles()
    if samples is None: return
    labels = pd.concat([labels, samples], ignore_index=True)

    labels["run"] = labels.groupby("driver").cumcount() # Durchlaufs-ID (run-id) hinzufügen
    
    identifyDriversCLI(labels.to_dict(orient="records"))
    #predictions, distances, compactness, centroidDists = identifyDriversCLI(labels.to_dict(orient="records"))
    #displayResults(labels, predictions, distances, compactness, centroidDists)

    

def selectTrainingFiles() -> list[str] | None:
    input("Zunächst müssen Trainingsdaten ausgewählt werden. Drücke ENTER wenn du bereit bist...")
    filePaths = askForFiles("Wähle die Trainingsdaten aus", [("CSV Dateien", "*.csv")])
    print(f"Es wurden {len(filePaths)} Trainingsdateien ausgewählt.")

    if len(filePaths) <= 0:
        print("Es muss mindestens eine CSV-Datei zum Training ausgewählt werden.")
        return
    
    print("")
    return list(filePaths)



def labelTrainingData(paths : list[str]) -> pd.DataFrame | None:
    choice = questionary.select(
        "Wie soll das Labeling erfolgen?",
        choices=[
            "1 - Label-Datei auswählen",
            "2 - Manuelles Labeling"
        ]
    ).ask()

    choice = int(choice.split(" - ")[0])

    labels = pd.DataFrame()
    if choice == 1:
        labels = selectLabelFile(list(paths))
    elif choice == 2:
        labels = labelManually(list(paths))
    else:
        print("Invalid Choice")
        return
    
    if labels is None:
        return

    if not all(p in labels['path'].values for p in paths):
        missing = [p for p in paths if p not in labels['path'].values]

        print(f"{len(missing)} Trainingsdateien haben noch kein Label.")
        doAddMissing = questionary.confirm("Sollen die fehlenden Labels manuell hinzugefügt werden?").ask()

        if not doAddMissing:
            print("Vorgang wird abgebrochen, da nicht alle Trainingsdaten gelabelt sind.")
            return None
        
        labels = labelManually(missing, existing=labels)
        if labels is None: return None

    print("")
    return labels



def selectLabelFile(dataPaths : list[str]) -> pd.DataFrame | None:
    filePath = askForFiles("Wähle die Label-Datei aus", [("Label-Datei", "*.lbl")], False)
    
    if not filePath:
        print("Es wurde keine Label-Datei ausgewählt")
        return pd.DataFrame()

    labels = pd.read_csv(str(filePath))

    if not set(labels.columns) == {"path", "driver", "filename"}:
        print("Die ausgewählte Label-Datei ist ungültig.")
        return None

    return labels[labels["path"].isin(dataPaths)].reset_index()



def labelManually(trainingFiles : list[str], existing : pd.DataFrame = pd.DataFrame(columns=["path", "driver"])) -> pd.DataFrame | None:
    input(f"Du wirst nun für jede der {len(trainingFiles)} Trainingsdateien nach der Identifikation des Fahrers gefragt. Drücke ENTER um fortzufahren...")

    labels = pd.DataFrame(columns=["path", "driver", "filename"])

    for i,file in enumerate(trainingFiles):
        id = questionary.text(f"Benenne den Fahrer der Trainingsdatei '{file}': ", validate=validateNotEmpty).ask()
        if not id:
            print("Vorgang wird abgebrochen")
            return None

        labels.loc[i] = [file, id.strip(), file.split('/')[-1]]
    
    labels = pd.concat([existing, labels], ignore_index=True)

    doSaveAsFile = questionary.confirm("Möchtest du die Labels als Datei speichern?").ask()
    if doSaveAsFile:
        saveFile(labels)
        
    return labels



def selectSampleFiles() -> pd.DataFrame | None:
    input(  "Als nächstes müssen die Sample-Dateien ausgewählt werden. \n" \
            "Für diese Dateien soll ein Fahrer identifiziert werden.\n" \
            "Drücke ENTER wenn du bereit bist...")
    filePaths = askForFiles("Wähle die Sample-Dateien aus", [("CSV Dateien", "*.csv")])
    print(f"Es wurden {len(filePaths)} Sample-Dateien ausgewählt.")

    if len(filePaths) <= 0:
        print("Es muss mindestens eine CSV-Datei zum Identifizieren ausgewählt werden.")
        return
    
    samples = pd.DataFrame(columns=["path", "driver", "filename"])

    for i,file in enumerate(filePaths):
        samples.loc[i] = [file, "", file.split('/')[-1]]
    
    print("")
    return samples



def displayResults(labels : pd.DataFrame, predictions : pd.DataFrame, distances : dict, compactness : pd.DataFrame, centroidDists : pd.DataFrame):
    print("\n================= FAHRERIDENTIFIKATION =================\n")

    for i, prediction in predictions.iterrows():

        rows = labels.loc[
            (labels["run"] == prediction["sampleId"]) &
            (labels["driver"] == "")
        ]

        filename = rows.iloc[0]["filename"] if not rows.empty else "Unbekannt"

        print(f"Datei: {filename}")
        print(f"→ Identifizierter Fahrer : {prediction['predictedDriver']}")
        print(f"→ Confidence             : {prediction['confidence']*100:.2f}%")
        print(f"→ Margin                 : {prediction['margin']:.3f}")
        print(f"→ Beste Distanz          : {prediction['bestDistance']:.3f}")
        print("-" * 56)

    print("\nFAHRERPROFILE (Kompaktheit)\n")

    print(
        compactness
        .rename(columns={
            "driver": "Driver",
            "n_samples": "Samples",
            "mean_distance": "MeanDist",
            "std_distance": "StdDist"
        })
        .to_string(
            index=False,
            formatters={
                "MeanDist": "{:.3f}".format,
                "StdDist": "{:.3f}".format
            }
        )
    )

    print("\nABSTÄNDE ZWISCHEN FAHRERPROFILEN\n")

    print(
        centroidDists
        .round(3)
        .to_string()
    )

    plot_distances(distances)
    plot_confidence_stats(predictions)



def plot_distances(distances: dict[str, np.ndarray], sample_idx: int = 0):
    drivers = list(distances.keys())
    values = [distances[d][sample_idx] for d in drivers]

    plt.bar(drivers, values)
    plt.ylabel("Distanz zum Centroid")
    plt.title(f"Sample {sample_idx}: Distanz zu Fahrerprofilen")
    plt.show()



def plot_confidence_stats(predictions: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].hist(predictions["margin"], bins=20)
    axs[0].set_title("Margin-Verteilung")

    axs[1].hist(predictions["confidence"], bins=20)
    axs[1].set_title("Confidence-Verteilung")

    plt.tight_layout()
    plt.show()



def validateNotEmpty(input : str):
    return bool(input and input.strip())



if __name__ == "__main__":
    main()