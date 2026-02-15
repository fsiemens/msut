import tkinter as tk
from tkinter import ttk
from typing import Callable

class ValidationFrame(tk.Frame):

    def __init__(self, parent, styleConfig : dict, maxProgress : int, onCancel : Callable):
        super().__init__(parent)
        title = tk.Label(self, text="Validating files...", font=styleConfig["font"]["h1"])
        title.pack(pady=styleConfig["paddings"]["default"])

        self.progress = ttk.Progressbar(
            self,
            orient="horizontal",
            length=300,
            mode="determinate",
            maximum=maxProgress
        )
        self.progress.pack(pady=styleConfig["paddings"]["default"])

        self.messageLabel = tk.Label(self, text="Starting up..", font=styleConfig["font"]["text"], wraplength=400)
        self.messageLabel.pack(pady=styleConfig["paddings"]["default"])

        cancelButton = tk.Button(self, text="Cancel", width=12, command=onCancel)
        cancelButton.pack(pady=styleConfig["paddings"]["default"], side="bottom")