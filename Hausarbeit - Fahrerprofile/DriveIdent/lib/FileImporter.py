import os
import glob
import tkinter as tk
from tkinter import filedialog

def findFilesInFolder(directory : str, pattern : str, verbose : bool = True) -> list[str]:
    '''
    Attempts to find all files matching the specified pattern in the target directory.
    
    :param directory: Target Directory; Relative Paths might be interpreted based on the location of this library
    :type directory: str
    :param pattern: Pattern that the file name must follow in order to be recognized. Allows for wildcards (*), i.e: 'recording_*.csv'
    :type pattern: str
    :param verbose: Enables logging. Default: True
    :type verbose: bool
    :return: List of file paths
    :rtype: list[str]
    '''

    filter = os.path.join(directory, pattern)   # Anfügen des Patterns an den Ordner-Pfad
    files = sorted(glob.glob(filter))           # Suchen nach passenden Dateien

    if verbose and len(files) == 0:
        print("Keine Dateien gefunden!")
        return []

    if verbose:
        print(f"Gefundene Dateien: {len(files)}")
    return files


def selectFilesFromOS(windowTitle : str = "Dateiauswahl", types : list[tuple[str, str]] = [("Alle Dateitypen", "*")], single : bool = False) -> list[str]:
    '''
    Allows the User to select one or multiple files from their Operating System
    
    :param windowTitle: Name / Title of the selection dialog, which is displayed in the header of the window.
    :type windowTitle: str
    :param types: List of tuples containing valid file extension (second) and its display name (first). I.e.: ("CSV-Files", ".csv"). Use "\\*" to allow every file extension. Default: [("Alle Dateitypen", "\\*")]
    :type types: list[tuple[str, str]]
    :param single: If set to True, prohibits the user from selecting more than one file. Default: False (allows multiple files to be selected).
    :type single: bool
    :return: List of file paths. Might be empty.
    :rtype: list[str]
    '''

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if not single:
        # Dateiauswahl-Dialog für mehrere Dateien
        filePaths = filedialog.askopenfilenames(
            title=windowTitle,
            filetypes=types,
        )
        filePaths = list(filePaths)
        root.destroy()
        return filePaths

    filePath = filedialog.askopenfilename(
        title=windowTitle,
        filetypes=types,
    )
    root.destroy()
    return [filePath]

