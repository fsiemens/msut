from DriveIdent.lib.windows.MainWindow import MainWindow
from pathlib import Path
import shutil
import sys

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    "font": {
        "text": ("Calibri", 12),
        "buttonText": ("Calibri", 10),
        "progressButton": ("Calibri", 10, "bold"),
        "h1": ("Calibri", 20, "bold"),
        "h2": ("Calibri", 16),
        "h3": ("Calibri", 14),
        "h4": ("Calibri", 12, "bold")
    },
    "colors": {
        "bg": "gray94",                          # Default: gray94
        "text": "black",                        # Default: black
        "h1" : "black",                         # Default: black
        "h2" : "black",                         # Default: black
        "h3" : "black",                         # Default: black
        "buttonBg" : "gray94",                  # Default: gray94
        "buttonFg" : "black",                   # Default: black
        "buttonSelect": "lightgray",            # Default: lightgray
        "progress" : "limegreen",               # Default: limegreen
        "stepperDone" : "limegreen",            # Default: limegreen
        "stepperActive" : "dodgerblue"          # Default: dodgerblue
    },
    "buttonHeight" : 1,                         # Default: 1
    "buttonWidth": 12,                          # Default: 12
    "buttonRelief": "raised",                   # Default: raised
    "paddings": {
        "default": 20,                          # Default: 20
        "tight": 5,                             # Default: 5
        "slim" : 10,                            # Default: 10
        "wide" : 30                             # Default: 30
    },
    "paths": {
        "artifacts" : BASE_DIR / "artifacts",
        "plots" : BASE_DIR / "artifacts/plots",
        "accuracyData": BASE_DIR / "artifacts/ergebnis.csv"
    }
}

def main():
    folder = Path(CONFIG["paths"]["artifacts"])

    if folder.exists():
        for item in folder.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    root = MainWindow(CONFIG)
    root.mainloop()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()