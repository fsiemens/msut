import tkinter as tk
from typing import cast, Callable, Literal
from .frames.ProgressFrame import ProgressFrame

class ProgressPopup(tk.Frame):
    
    def __init__(self, parent, styleConfig : dict, title : str):
        popup = tk.Toplevel(parent)
        popup.title("Popup")
        popup.geometry("500x600")
        popup.resizable(False, False)
        popup.attributes('-topmost', True)
        popup.protocol("WM_DELETE_WINDOW", self.close)

        popup.transient(parent)
        popup.grab_set()          # makes window modal

        self.styleConfig = styleConfig
        self.popup = popup

        self.frame = ProgressFrame(popup, styleConfig, title, 100, onCancel=self.close)
        self.frame.pack(pady=styleConfig["paddings"]["default"], expand=True)

    def updateProgress(self, phase, total, completed_list, in_progress_list, message, remaining, percent):
        if not self.popup.winfo_exists():
            return

        self.frame.messageLabel.config(text=message)
        self.frame.progress.config(value=int(percent))
        #self.popup.after(0, lambda: self.frame.messageLabel.config(text=message))
        #self.popup.after(0, lambda: self.frame.progress.config(value=int(percent)))

        if phase == "done":
            self.close()

    def close(self):
        self.popup.destroy()