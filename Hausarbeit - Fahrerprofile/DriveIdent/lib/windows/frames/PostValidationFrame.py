import tkinter as tk
from typing import Callable

class PostValidationFrame(tk.Frame):

    def __init__(self, parent, styleConfig : dict, onCancel : Callable, onProceed : Callable, faultyFiles : list[str] = []):
        super().__init__(parent)

        self.grid_rowconfigure(0, weight=3)   
        self.grid_rowconfigure(1, weight=1)   # Label
        self.grid_rowconfigure(2, weight=1)   # Buttons
        self.grid_rowconfigure(3, weight=3)   

        self.grid_columnconfigure(0, weight=1)

        title = tk.Label(
            self,
            text=f"Found {len(faultyFiles)} faulty file(s)\nProceed without them?",
            font=styleConfig["font"]["h1"],
        )
        title.grid(row=1, column=0, pady=styleConfig["paddings"]["default"])

        buttonFrame = tk.Frame(self)
        buttonFrame.grid(row=2, column=0)

        cancelButton = tk.Button(buttonFrame, text="Cancel", width=12, command=onCancel)
        cancelButton.grid(row=0, column=0, padx=styleConfig["paddings"]["default"])

        proceedButton = tk.Button(buttonFrame, text="Proceed", width=12, command=onProceed)
        proceedButton.grid(row=0, column=1, padx=styleConfig["paddings"]["default"])
