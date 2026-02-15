import tkinter as tk
from tkinter import messagebox
from lib.components.StepProgressBar import StepProgress
from .frames.FileSelectFrame import FileSelectFrame
from .frames.ModelOptionsFrame import ModelOptionsFrame
from .ValidationPopup import ValidationPopup
from lib.FileImporter import selectFilesFromOS
from lib.FileExporter import saveLabelFileOS
from lib.BackendStub import validate_feature_csv    # TODO BACKEND STUBS -- REMOVE BEFORE SHIPPING
from typing import Literal
import pandas as pd
import os


class MainWindow(tk.Tk):

    def __init__(self, styleConfig : dict):
        super().__init__()

        self.styleConfig = styleConfig
        self.geometry("800x700")
        self.title("DriveIdent")
        self.resizable(False, False)
        #self.attributes('-topmost', True)

        self.trainFiles = pd.DataFrame(columns=["File", "Label"])
        self.testFiles = pd.DataFrame(columns=["File"])
        self.options = {}
        self.selectedModels = ["logreg"]

        self.stepProgress = StepProgress(self, self.styleConfig, labels=[("File Selection", self.openFileSelection), ("Model Selection", self.openModelSelection), ("Evaluation", self.openEvaluation)], stepAccessValidationFct=lambda step:step <= self.stepProgress.getCurrentStep())
        self.stepProgress.pack(pady=styleConfig["paddings"]["wide"])

        self.currentFrame = None
        self.openFileSelection()

    def onPopupClose(self, isSuccess : bool, target : Literal["train", "test"], files : list[str], faultyFiles : list[str]):

        if not isinstance(self.currentFrame, FileSelectFrame):
            return

        if not isSuccess:
            if target == "train":
                self.trainFiles = []
            elif target == "test":
                self.testFiles = []
            else:
                raise ValueError("Target must be either 'train' or 'test'")
            return
        
        files = [f for f in files if f not in faultyFiles]
        if target == "train":
            self.trainFiles.drop(self.trainFiles.index, inplace=True)       # type: ignore
            self.trainFiles["File"] = files                                 # type: ignore
            self.trainFiles["Label"] = [""] * len(files)                    # type: ignore
        elif target == "test":
            self.testFiles.drop(self.testFiles.index, inplace=True)         # type: ignore
            self.testFiles["File"] = files                                  # type: ignore
        else:
            raise ValueError("Target must be either 'train' or 'test'")
        self.currentFrame.updateTables()

    def selectRecordings(self, target : Literal["train", "test"], title : str):
        files = selectFilesFromOS(title, [("CSV-Recordings", "*.csv")])
        popup = ValidationPopup(self, self.styleConfig, target, files, onPopupClose=self.onPopupClose)

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
            if not validate_feature_csv(file):
                faultyFiles.append(file)
            # Since the verification operating might take some time, we will use an asynchronous recursion through tkinter to allow the UI to update
            self.after(500, lambda: validateFile(index +1, faultyFiles)) # TODO DELAY ONLY FOR SIMULATION -- REMOVE BEFORE SHIPPING
        
        validateFile()


    def selectTrainFiles(self):
        self.selectRecordings("train", "Select Training Files")

    def selectTestFiles(self):
        self.selectRecordings("test", "Select Test Files")

    def loadLabelFile(self):
        labelFile = selectFilesFromOS("Select Label File", [("Label-File", "*.lbl")], True)
        if not labelFile or len(labelFile) <= 0:
            return

        labels = pd.read_csv(labelFile[0])
        mapping = dict(zip(labels["File"].apply(os.path.basename),labels["Label"]))
        self.trainFiles["Label"] = self.trainFiles["File"].apply(os.path.basename).map(mapping).fillna("")  # type: ignore
        if isinstance(self.currentFrame, FileSelectFrame):
            self.currentFrame.updateTables()
    
    def saveLabelFile(self):
        data = pd.DataFrame(self.trainFiles.copy())
        data["File"] = data["File"].apply(lambda x: os.path.basename(str(x)))
        saveLabelFileOS(data)

    def next(self):
        if isinstance(self.currentFrame, FileSelectFrame):
            proceed, reason = self.canProceed()
            if proceed:
                self.openModelSelection()
                self.stepProgress.next()
                print(self.trainFiles)
                print(self.testFiles)
                return
            print("Cannot proceed")
            messagebox.showwarning("Invalid Configuration", reason)

        elif isinstance(self.currentFrame, ModelOptionsFrame):
            self.openEvaluation()
            self.stepProgress.next()

    def openFileSelection(self):
        if self.currentFrame is not None:
            self.currentFrame.destroy()
        self.currentFrame = FileSelectFrame(self, self.styleConfig, self.selectTrainFiles, self.selectTestFiles, self.next, self.loadLabelFile, self.saveLabelFile, self.trainFiles, self.testFiles) # type: ignore
        self.currentFrame.pack(expand=True, fill="both")
        print("File Selection")

    def openModelSelection(self):
        if self.currentFrame is not None:
            self.currentFrame.destroy()
        self.currentFrame = ModelOptionsFrame(self, self.styleConfig, self.options, self.selectedModels, self.next, models=["extratrees", "logreg", "randomforest", "svm_rbf"])
        self.currentFrame.pack(expand=True, fill="both")
        print("Model Selection")

    def openEvaluation(self):
        print("Evaluation Selection")

    def canProceed(self) -> tuple[bool, str]:
        trainFilesEmpty = self.trainFiles.empty   # type: ignore
        if trainFilesEmpty:
            return False, "Please select at least one file for training"

        testFilesEmpty = self.testFiles.empty     # type: ignore
        if testFilesEmpty:
            return False, "Please select at least one file for testing / identification"

        trainFilesValid = (
            self.trainFiles["File"].astype(str).str.strip().ne("") & # type: ignore
            self.trainFiles["Label"].astype(str).str.strip().ne("")  # type: ignore
        ).all()

        if not trainFilesValid:
            return False, "Please make sure all training files are labeled"

        return True, ""

