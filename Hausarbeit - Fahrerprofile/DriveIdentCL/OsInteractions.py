import tkinter as tk
from tkinter import filedialog
import os
import platform
import pandas as pd

def clearTerminal():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")



def askForFiles(windowTitle : str, types : list[tuple[str, str]], multiple : bool = True) -> str | list[str]:
    # Hauptfenster verstecken (wir brauchen nur den Dialog)
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if multiple:
        # Dateiauswahl-Dialog, mehrere Dateien möglich
        filePaths = filedialog.askopenfilenames(
            title=windowTitle,
            filetypes=types,
        )
        filePaths = list(filePaths)
        root.destroy()
        return filePaths

    else:
        filePath = filedialog.askopenfilename(
            title=windowTitle,
            filetypes=types,
        )
        root.destroy()
        return filePath



def saveFile(data: pd.DataFrame):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    filePath = filedialog.asksaveasfilename(
        title="Speicher die Datei als...",
        defaultextension=".lbl",
        filetypes=[("Label-Dateien", "*.lbl")]
    )

    root.destroy()

    if filePath:
        data.to_csv(filePath, index=False)
        print(f"Datei gespeichert unter: {filePath}")
    else:
        print("Speichern abgebrochen.")