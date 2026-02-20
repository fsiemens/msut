import tkinter as tk
from tkinter import ttk
from typing import Callable, Literal
import pandas as pd
from DriveIdent.lib.components.EditableTable import EditableTable
from DriveIdent.lib.components.GenericButton import GenericButton
from DriveIdent.lib.core.backend_adapter import get_config

class PredictionFrame(ttk.Frame):
    
    def __init__(self, parent, styleConfig : dict, testFiles : pd.DataFrame, onSelectTestRecordings : Callable, onPredict : Callable):
        super().__init__(parent)

        self.testFiles = testFiles

        # Creating two columns in main frame
        leftFrame = tk.LabelFrame(self, text="Test-Dateien", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        self.testFileTable = EditableTable(leftFrame, self.testFiles, editable=False)
        self.testFileTable.pack(expand=True, fill="both")

        # Creating containers for the buttons of both columns
        buttonFrameLeft = tk.Frame(leftFrame)
        buttonFrameLeft.pack(pady=styleConfig["paddings"]["tight"])

        GenericButton(buttonFrameLeft, styleConfig, "Wählen", onSelectTestRecordings).pack(side="left", padx=styleConfig["paddings"]["tight"])

        nextButton = GenericButton(self, styleConfig, text="Vorhersagen", command=onPredict, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])

        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_rowconfigure(0, weight=1)
        leftFrame.grid_propagate(False)

    def updateTables(self):
        self.testFileTable.refresh()