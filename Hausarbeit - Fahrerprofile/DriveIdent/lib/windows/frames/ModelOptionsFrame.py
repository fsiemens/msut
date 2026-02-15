import tkinter as tk
from tkinter import ttk
from typing import Callable

class ModelOptionsFrame(tk.Frame):
    def __init__(self, parent, styleConfig : dict, options : dict[str, tk.Variable], selectedModels : list[str], onNext : Callable, models : list[str] = ["none"]):
        super().__init__(parent)
        self.options = options
        self.selectedModels = selectedModels

        entryValidation = self.register(self.validateNumber)

        leftFrame = tk.LabelFrame(self, text="Select Models", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        rightFrame = tk.LabelFrame(self, text="Options", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")
        
        self.modelListbox = tk.Listbox(leftFrame, selectmode=tk.MULTIPLE, height=10)
        self.modelListbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.modelListbox.bind('<<ListboxSelect>>', self.updateSelectedModels)
        
        scrollbar = tk.Scrollbar(leftFrame, orient=tk.VERTICAL, command=self.modelListbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.modelListbox.config(yscrollcommand=scrollbar.set)

        nextButton = tk.Button(self, text="Next", command=onNext, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])

        for model in models:
            self.modelListbox.insert(tk.END, model)
            if model in self.selectedModels:
                self.modelListbox.select_set(tk.END)
        
        # Helper to add options to frame
        def addOption(frame : tk.LabelFrame, index : str, text : str, default : str | bool, entryWidth : int = 5):
            var = self.options[index] if index in self.options else None

            if type(default) == str:
                if var is None:
                    var = tk.StringVar(value=default)
                    self.options[index] = var

                row = tk.Frame(frame)
                row.pack(anchor="w", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["tight"])
                entry = tk.Entry(row, width=entryWidth, textvariable=var, validate="key", validatecommand=(entryValidation, "%P"))
                entry.pack(side="left", padx=styleConfig["paddings"]["tight"])
                label = tk.Label(row, text=text)
                label.pack(side="left", padx=styleConfig["paddings"]["slim"])
            elif type(default) == bool:
                if var is None:
                    var = tk.BooleanVar(value=default)
                    self.options[index] = var

                cb = tk.Checkbutton(frame, text=text, variable=var, anchor="w")
                cb.pack(fill=tk.X, padx=styleConfig["paddings"]["tight"], pady=styleConfig["paddings"]["tight"])
        
        addOption(rightFrame, "extract-from-raw", "Extract Features from raw data", True)
        addOption(rightFrame, "skip-featuretools", "Skip FeatureTools", False)
        addOption(rightFrame, "ft-max-obs", "Max FeatureTools window size", "500")
        addOption(rightFrame, "ft-depth", "Max FeatureTools DFS Depth", "1")
        addOption(rightFrame, "with-merged", "Merge Data", True)
        addOption(rightFrame, "with-selected", "Auto-select best K Features", True)
        addOption(rightFrame, "k-features", "K: Amount of Features to select", "5")
        addOption(rightFrame, "loo", "Use Leave-One-Out-Strategy instead of K-Fold", False)
        addOption(rightFrame, "force", "Ignore cached Data", False)
        addOption(rightFrame, "n-splits", "Amount of Folds for StratifiedGroupKFold", "5")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)
        rightFrame.grid_propagate(False)
        leftFrame.grid_propagate(False)

    def updateSelectedModels(self, event=None):
        selection = self.modelListbox.curselection()
        self.selectedModels.clear()
        self.selectedModels.extend([self.modelListbox.get(i) for i in selection])

    def validateNumber(self, newValue : str):
        if newValue == "":
            return True
        
        return newValue.isdigit() and not newValue.startswith("0") and int(newValue) > 0 and int(newValue) < 10000