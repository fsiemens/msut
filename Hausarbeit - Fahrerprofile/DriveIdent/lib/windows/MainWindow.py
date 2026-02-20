import tkinter as tk
from tkinter import messagebox
from DriveIdent.lib.components.StepProgressBar import StepProgress
from .frames.TrainingFrame import TrainingFrame
from .frames.ModelFrame import ModelFrame
from .frames.PredictionFrame import PredictionFrame
from .ValidationPopup import ValidationPopup
from .ProgressPopup import ProgressPopup
from DriveIdent.lib.Util import getPlotPaths, getModelAccuracyData
from DriveIdent.lib.FileImporter import selectFilesFromOS
from DriveIdent.lib.FileExporter import saveLabelFileOS
from DriveIdent.lib.core.backend_adapter import train, predict, validate_csv, set_config
from typing import Literal, Callable
import pandas as pd
import threading
from pathlib import Path
import os

class MainWindow(tk.Tk):

    def __init__(self, styleConfig : dict):
        super().__init__()

        self.styleConfig = styleConfig
        self.geometry("800x700")
        self.title("DriveIdent")
        self.resizable(False, False)
        #self.attributes('-topmost', True)

        self.dataDir = ""
        self.trainFiles : pd.DataFrame = pd.DataFrame(columns=["File", "Label"])
        self.testFiles = pd.DataFrame(columns=["File", "randomforest", "logreg"])
        self.options = {}
        #self.selectedModels = ["logreg"]
        self.modelAccuracy = pd.DataFrame(columns=["Model", "Precision"])

        self.stepProgress = StepProgress(self, self.styleConfig, labels=[("Training", self.openTrainingFrame), ("Modell", self.openModelFrame), ("Vorhersage", self.openPredictionFrame)], stepAccessValidationFct=lambda step:step <= self.stepProgress.getCurrentStep())
        self.stepProgress.pack(pady=self.styleConfig["paddings"]["wide"])

        self.currentFrame = None
        self.openTrainingFrame()

    def selectTrainFiles(self):
        self.selectRecordings(self.trainFiles, "Trainingsdateien auswählen", self.onPopupClose)

    def selectTestFiles(self):
        self.selectRecordings(self.testFiles, "Test-Dateien auswählen", self.onPopupClose)

    def selectRecordings(self, destination : pd.DataFrame, title : str, onCloseCallback : Callable):
        files = selectFilesFromOS(title, [("CSV-Recordings", "*.csv")])
        popup = ValidationPopup(self, self.styleConfig, destination, files, onPopupClose=onCloseCallback)

        def postValidation(faultyFiles : list[str]):
            popup.setFaultyFiles(faultyFiles)
            if len(faultyFiles) <= 0:
                popup.proceed()
            else:
                popup.showPostValidationFrame()

        def validateFile(index : int = 0, faultyFiles : list[str] = []):
            if index >= len(files):
                postValidation(faultyFiles)   # Once all files have been processed invoke callback
                return

            file = files[index]
            popup.updateProgress(index + 1, file)
            if not validate_csv(file):
                faultyFiles.append(file)
            # Since the verification operating might take some time, we will use an asynchronous recursion through tkinter to allow the UI to update
            self.after(100, lambda: validateFile(index +1, faultyFiles))
        
        validateFile()

    def onPopupClose(self, isSuccess : bool, destination : pd.DataFrame, files : list[str], faultyFiles : list[str]):
        files = [f for f in files if f not in faultyFiles]
        destination.drop(destination.index, inplace=True)

        if isSuccess:
            destination["File"] = files
            if "Label" in destination.columns:
                destination["Label"] = [""] * len(files)
            if len(files) > 0:
                self.dataDir = Path(files[0]).parent
        if isinstance(self.currentFrame, TrainingFrame) or isinstance(self.currentFrame, PredictionFrame):
            self.currentFrame.updateTables()

    def mapConfig(self) -> dict:
        config = {}
        models = []
        if bool(self.options["model_use_randomforest"].get()): models.append("randomforest")
        if bool(self.options["model_use_logreg"].get()): models.append("logreg") 
        
        useFeatureTools = bool(self.options["features_use_featuretools"].get())
        useTsFresh = bool(self.options["features_use_tsfresh"].get())

        featureSets = "both"
        if useFeatureTools and not useTsFresh:
            featureSets = "featuretools"
        elif useTsFresh and not useFeatureTools:
            featureSets = "tsfresh"

        config["models"] = models
        config["feature_set"] = featureSets
        for k, v in self.options.items():
            config[k] = v.get()
        return config

    def startTraining(self):
        proceed, reason = self.canStartTraining()
        if proceed:
            set_config(self.mapConfig())
            popup = ProgressPopup(self, self.styleConfig, "Training")
            threading.Thread(
                target=self.training,
                args=(popup.updateProgress,popup,),
                daemon=True
            ).start()
            return
        messagebox.showwarning("Unvollständiges Setup", reason)

    def training(self, callback, popup : ProgressPopup):
        success, out = train(self.dataDir, self.trainFiles, self.styleConfig["paths"]["artifacts"], progress_callback=callback)
        if not success:
            popup.close()
            messagebox.showerror("Kritischer Fehler", "Training fehlgeschlagen" if out is None else out)
            return
        
        self.modelAccuracy = getModelAccuracyData(self.styleConfig["paths"]["accuracyData"])
        print(self.modelAccuracy)
        self.openModelFrame()
        
    def startPrediction(self):
        popup = ProgressPopup(self, self.styleConfig, "Vorhersagen")

        threading.Thread(
            target=self.prediction,
            args=(popup.updateProgress,),
            daemon=True
        ).start()

    def prediction(self, callback):
        success, out, result = predict(self.dataDir, self.testFiles.copy(), self.styleConfig["paths"]["artifacts"], progress_callback=callback)
        
        if not success:
            messagebox.showerror("Kritischer Fehler", "Vorhersage fehlgeschlagen" if out is None else out)
            return

        for model, predictions in result.items():
            mapping = {p["recording"]: p["ist"] for p in predictions}

            self.testFiles.loc[:, model] = (
                self.testFiles["File"]
                .apply(os.path.basename)
                .map(mapping)
            )
        if isinstance(self.currentFrame, PredictionFrame):
            self.currentFrame.testFiles = self.testFiles
            self.currentFrame.updateTables()

    def loadLabelFile(self):
        labelFile = selectFilesFromOS("Label-Datei auswählen", [("Label-Datei", "*.lbl")], True)
        if not labelFile or len(labelFile) <= 0:
            return

        labels = pd.read_csv(labelFile[0])
        mapping = dict(zip(labels["File"].apply(os.path.basename),labels["Label"]))
        self.trainFiles["Label"] = self.trainFiles["File"].apply(os.path.basename).map(mapping).fillna("")
        if isinstance(self.currentFrame, TrainingFrame):
            self.currentFrame.updateTables()
    
    def saveLabelFile(self):
        data = pd.DataFrame(self.trainFiles.copy())
        data["File"] = data["File"].apply(lambda x: os.path.basename(str(x)))
        saveLabelFileOS(data)

    def openTrainingFrame(self):
        trainingFrame = TrainingFrame(self, self.styleConfig, self.options, self.selectTrainFiles, self.startTraining, self.loadLabelFile, self.saveLabelFile, self.trainFiles)
        self.openFrame(trainingFrame)

    def openModelFrame(self):
        self.stepProgress.next()
        modelFrame = ModelFrame(self, self.styleConfig, self.openPredictionFrame, modelAccuracyData=self.modelAccuracy, images=getPlotPaths(self.styleConfig["paths"]["plots"]))
        self.openFrame(modelFrame)

    def openPredictionFrame(self):
        predictionFrame = PredictionFrame(self, self.styleConfig, self.testFiles, self.selectTestFiles, self.startPrediction)
        self.stepProgress.next()
        self.openFrame(predictionFrame)

    def openFrame(self, frame):
        if self.currentFrame is not None:
            self.currentFrame.destroy()
        self.currentFrame = frame
        self.currentFrame.pack(expand=True, fill="both")

    def canStartTraining(self) -> tuple[bool, str]:
        trainFilesEmpty = self.trainFiles.empty
        if trainFilesEmpty:
            return False, "Es muss mindestens eine Trainingsdatei ausgewählt werden."

        trainFilesValid = (
            self.trainFiles["File"].astype(str).str.strip().ne("") &
            self.trainFiles["Label"].astype(str).str.strip().ne("")
        ).all()

        if not trainFilesValid:
            return False, "Allen Trainingsdateien muss ein Label zugewiesen sein."

        # TODO validate Options

        return True, ""