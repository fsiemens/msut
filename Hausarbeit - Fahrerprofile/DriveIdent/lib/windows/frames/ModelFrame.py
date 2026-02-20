import tkinter as tk
from tkinter import ttk
import pandas as pd
from pathlib import Path
from typing import Callable

from DriveIdent.lib.components.ImageGallery import ImageGallery

class ModelFrame(tk.Frame):
    def __init__(self, parent, styleConfig : dict, onNext : Callable, modelAccuracyData : pd.DataFrame = pd.DataFrame(columns=["Model", "Precision"]), images : list[str] = []):
        super().__init__(parent)
        self.modelAccurarcyData = modelAccuracyData
        print(self.modelAccurarcyData)
        self.images = images

        leftFrame = tk.LabelFrame(self, text="Genauigkeit", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        for _, row in modelAccuracyData.iterrows():
            tk.Label(leftFrame, text=f"{row['Model']}-Modell:", font=styleConfig["font"]["h4"], anchor="w").pack(fill="x", pady=styleConfig["paddings"]["slim"])
            tk.Label(leftFrame, text=f"{row['Precision']*100:.2f}% Genauigkeit", font=styleConfig["font"]["text"], anchor="w").pack(fill="x")

        rightFrame = tk.LabelFrame(self, text="Diagramme", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        gallery = ImageGallery(rightFrame, styleConfig, images)
        gallery.pack(fill="both", expand=True)

        nextButton = tk.Button(self, text="Weiter", command=onNext, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        rightFrame.grid_propagate(False)
        leftFrame.grid_propagate(False)
