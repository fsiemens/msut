import tkinter as tk
from tkinter import filedialog
import pandas as pd

def saveLabelFileOS(data : pd.DataFrame):
    '''
    Allows the User to store the Labels as a Label-File (.lbl) at a location selected through the Operating System 
    
    :param data: DataFrame to save
    :type data: pd.DataFrame
    '''

    saveDataFrameToCsvOS(data, defaultExtension=".lbl", types=[("Label-Dateien", "*.lbl")])

def saveDataFrameToCsvOS(data: pd.DataFrame, windowTitle : str = "Datei speichern als...", defaultExtension : str = ".csv",  types : list[tuple[str, str]] = [("CSV-Datei", "*.csv")], verbose : bool = False):
    '''
    Allows the User to store the DataFrame to a File selected through the Operating System
    
    :param data: DataFrame to save
    :type data: pd.DataFrame
    :param windowTitle: Name / Title of the selection dialog, which is displayed in the header of the window.
    :type windowTitle: str
    :param defaultExtension: The file extension out of types which is selected by default
    :type defaultExtension: str
    :param types: List of tuples containing valid file extension (second) and its display name (first). I.e.: ("CSV-Files", ".csv"). Use "\\*" to allow every file extension. Default: [("CSV-Datei", "\\*.csv")]
    :type types: list[tuple[str, str]]
    :param verbose: Enables logging
    :type verbose: bool
    '''
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    filePath = filedialog.asksaveasfilename(
        title=windowTitle,
        defaultextension=defaultExtension,
        filetypes=types
    )

    root.destroy()

    if filePath:
        data.to_csv(filePath, index=False)
        if verbose:
            print(f"Datei gespeichert unter: {filePath}")
        return
    
    if verbose:
        print("Speichern abgebrochen.")

def saveDataFrameAsCsvPath(data : pd.DataFrame, filepath : str, verbose : bool = False):
    '''
    Saves a DataFrame to a CSV-File
    
    :param data: DataFrame to save
    :type data: pd.DataFrame
    :param filepath: Fully qualified file name at which the CSV is to be saved
    :type filepath: str
    :param verbose: Enables logging
    :type verbose: bool
    '''

    try:
        data.to_csv(filepath)
        return True
    except:
        if verbose:
            print("Speichern fehlgeschlagen.")
        return False