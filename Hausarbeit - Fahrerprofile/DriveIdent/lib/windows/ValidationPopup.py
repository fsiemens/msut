import tkinter as tk
from tkinter import ttk
from typing import cast, Callable, Literal
from .frames.ValidationFrame import ValidationFrame
from .frames.PostValidationFrame import PostValidationFrame

class ValidationPopup(tk.Frame):
    
    def __init__(self, parent, styleConfig : dict, target : Literal["train", "test"], files : list[str], onPopupClose : Callable[[bool, Literal["train", "test"], list[str], list[str]], None]):
        popup = tk.Toplevel(parent)
        popup.title("Popup")
        popup.geometry("500x600")
        popup.resizable(False, False)
        popup.attributes('-topmost', True)
        popup.protocol("WM_DELETE_WINDOW", self.cancel)

        popup.transient(parent)
        popup.grab_set()          # makes window modal

        self.styleConfig = styleConfig
        self.target = target
        self.files = files
        self.faultyFiles = []
        self.popup = popup
        self.closeAction = onPopupClose
        self.frame = ValidationFrame(popup, self.styleConfig, len(files), onCancel=self.cancel) 
        self.frame.pack(pady=styleConfig["paddings"]["default"], expand=True)

    def updateProgress(self, value : int, message : str):
        if not isinstance(self.frame, ValidationFrame):
            return

        self.frame.messageLabel["text"] = message
        self.frame.progress["value"] = value

    def proceed(self):
        self.popup.destroy()
        self.closeAction(True, cast(Literal["train", "test"], self.target), self.files, self.faultyFiles)

    def cancel(self):
        self.popup.destroy()
        self.closeAction(False, cast(Literal["train", "test"], self.target), self.files, self.faultyFiles)

    def showPostValidationFrame(self):
        if self.frame is not None:
            self.frame.destroy()

        self.frame = PostValidationFrame(self.popup, self.styleConfig, onCancel=self.cancel, onProceed=self.proceed, faultyFiles=self.faultyFiles)
        self.frame.pack(expand=True, fill="both")

    def setFaultyFiles(self, faultyFiles : list[str]):
        self.faultyFiles = faultyFiles