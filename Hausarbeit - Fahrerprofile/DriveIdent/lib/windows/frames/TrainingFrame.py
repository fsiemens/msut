import tkinter as tk
from tkinter import ttk
from typing import Callable, Literal
import pandas as pd
from DriveIdent.lib.components.EditableTable import EditableTable
from DriveIdent.lib.components.GenericButton import GenericButton
from DriveIdent.lib.core.backend_adapter import get_config

class TrainingFrame(ttk.Frame):
    
    def __init__(self, parent, styleConfig : dict, modelOptions : dict[str, tk.Variable], onSelectTrainRecordings : Callable, onTrain : Callable, onLoadLabels : Callable, onSaveLabels : Callable, trainFiles : pd.DataFrame):
        super().__init__(parent)

        self.options = modelOptions
        self.trainFiles = trainFiles
        entryValidation = self.register(self.validateNumber)

        # Creating two columns in main frame
        leftFrame = tk.LabelFrame(self, text="Trainingsdateien", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        rightFrame = tk.LabelFrame(self, text="Modelloptionen", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        # Using canvas as a scrollable container
        canvas = tk.Canvas(rightFrame, highlightthickness=0)
        scrollbar = tk.Scrollbar(rightFrame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        scrollFrame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=scrollFrame, anchor="nw")

        scrollFrame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        self.trainingFileTable = EditableTable(leftFrame, self.trainFiles)
        self.trainingFileTable.pack(expand=True, fill="both")

        # Creating containers for the buttons of both columns
        buttonFrameLeft = tk.Frame(leftFrame)
        buttonFrameLeft.pack(pady=styleConfig["paddings"]["tight"])

        GenericButton(buttonFrameLeft, styleConfig, "Wählen", onSelectTrainRecordings).pack(side="left", padx=styleConfig["paddings"]["tight"])
        GenericButton(buttonFrameLeft, styleConfig, "Label-Import", onLoadLabels).pack(side="left", padx=styleConfig["paddings"]["tight"])
        GenericButton(buttonFrameLeft, styleConfig, "Label-Export", onSaveLabels).pack(side="left", padx=styleConfig["paddings"]["tight"])

        # Helper to add options to frame
        def addOption(frame, index : str, text : str, default : str | bool | None, entryWidth : int = 5):
            if default is None: # Dummy Option / 
                label = tk.Label(frame, text=text, font=styleConfig["font"]["h4"])
                label.pack(anchor="w", padx=styleConfig["paddings"]["slim"], pady=styleConfig["paddings"]["slim"])
                return

            var = self.options[index] if index in self.options else None

            if type(default) == str:
                if var is None:
                    var = tk.StringVar(value=default)
                    self.options[index] = var

                row = tk.Frame(frame)
                row.pack(anchor="w", padx=styleConfig["paddings"]["tight"], pady=styleConfig["paddings"]["tight"])
                entry = tk.Entry(row, width=entryWidth, textvariable=var, validate="key", validatecommand=(entryValidation, "%P"))
                entry.pack(side="left", padx=styleConfig["paddings"]["tight"])
                label = tk.Label(row, text=text, font=styleConfig["font"]["text"])
                label.pack(side="left", padx=styleConfig["paddings"]["slim"])
            elif type(default) == bool:
                if var is None:
                    var = tk.BooleanVar(value=default)
                    self.options[index] = var

                cb = tk.Checkbutton(frame, text=text, variable=var, anchor="w", font=styleConfig["font"]["text"])
                cb.pack(fill=tk.X, padx=styleConfig["paddings"]["tight"], pady=styleConfig["paddings"]["tight"])
        
        config = get_config()

        addOption(scrollFrame, "models_dummy", "Zu verwendende Modelle:", None)
        addOption(scrollFrame, "model_use_randomforest", "Random-Forest", "randomforest" in config["models"]["value"])
        addOption(scrollFrame, "model_use_logreg", "Logistische Regression", "logreg" in config["models"]["value"])

        addOption(scrollFrame, "feature_sets_dummy", "Zu verwendende Feature-Algorithmen:", None)
        addOption(scrollFrame, "features_use_featuretools", "FeatureTools", config["feature_set"]["value"] in ["both", "featuretools"])
        addOption(scrollFrame, "features_use_tsfresh", "TS-Fresh", config["feature_set"]["value"] in ["both", "tsfresh"])

        addOption(scrollFrame, "windowing_dummy", "Fenster-Parameter", None)
        addOption(scrollFrame, "window_sec", str(config["window_sec"]["desc"]), str(config["window_sec"]["value"]))
        addOption(scrollFrame, "step_sec", str(config["step_sec"]["desc"]), str(config["step_sec"]["value"]))
        addOption(scrollFrame, "min_points", str(config["min_points"]["desc"]), str(config["min_points"]["value"]))
        addOption(scrollFrame, "max_points", str(config["max_points"]["desc"]), str(config["max_points"]["value"]))

        addOption(scrollFrame, "validation_dummy", "Validierung", None)
        addOption(scrollFrame, "cv_splits", str(config["cv_splits"]["desc"]), str(config["cv_splits"]["value"]))

        nextButton = GenericButton(self, styleConfig, text="Trainieren", command=onTrain, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])

        self.grid_columnconfigure(0, weight=1, uniform="group1")
        self.grid_columnconfigure(1, weight=1, uniform="group1")
        self.grid_rowconfigure(0, weight=1)
        rightFrame.grid_propagate(False)
        leftFrame.grid_propagate(False)

    def updateTables(self):
        self.trainingFileTable.refresh()

    def validateNumber(self, newValue : str):
        if newValue == "":
            return True
        
        return newValue.isdigit() and not newValue.startswith("0") and int(newValue) > 0 and int(newValue) < 10000