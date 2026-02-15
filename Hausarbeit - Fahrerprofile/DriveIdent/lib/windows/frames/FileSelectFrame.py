import tkinter as tk
from tkinter import ttk
from typing import Callable, Literal
import pandas as pd
from lib.components.EditableTable import EditableTable
from lib.components.GenericButton import GenericButton

class FileSelectFrame(ttk.Frame):
    
    def __init__(self, parent, styleConfig : dict, onSelectTrainRecordings : Callable, onSelectTestRecordings : Callable, onNext : Callable, onLoadLabels : Callable, onSaveLabels : Callable, trainFiles : pd.DataFrame, testFiles : pd.DataFrame):
        super().__init__(parent)

        self.trainFiles = trainFiles
        self.testFiles = testFiles

        # Creating two columns in main frame
        leftFrame = tk.LabelFrame(self, text="Training Files", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        rightFrame = tk.LabelFrame(self, text="Test Files", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"])

        self.trainingFileTable = EditableTable(leftFrame, self.trainFiles)
        self.trainingFileTable.pack(expand=True, fill="both")

        self.testFileTable = EditableTable(rightFrame, self.testFiles)
        self.testFileTable.pack(expand=True, fill="both")

        # Creating containers for the buttons of both columns
        buttonFrameLeft = tk.Frame(leftFrame)
        buttonFrameLeft.pack(pady=styleConfig["paddings"]["tight"])

        buttonFrameRight = tk.Frame(rightFrame)
        buttonFrameRight.pack(pady=styleConfig["paddings"]["tight"])

        GenericButton(buttonFrameLeft, styleConfig, "Select", onSelectTrainRecordings).pack(side="left", padx=styleConfig["paddings"]["tight"])
        GenericButton(buttonFrameLeft, styleConfig, "Load Labels", onLoadLabels).pack(side="left", padx=styleConfig["paddings"]["tight"])
        GenericButton(buttonFrameLeft, styleConfig, "Save Labels", onSaveLabels).pack(side="left", padx=styleConfig["paddings"]["tight"])

        GenericButton(buttonFrameRight, styleConfig, "Select", onSelectTestRecordings).pack(side="left", padx=styleConfig["paddings"]["tight"])

        nextButton = GenericButton(self, styleConfig, text="Next", command=onNext, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])

        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=1, uniform="group1")
        self.grid_rowconfigure(0, weight=1)

    def updateTables(self):
        self.trainingFileTable.refresh()
        self.testFileTable.refresh()