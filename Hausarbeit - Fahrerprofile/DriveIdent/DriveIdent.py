import pandas as pd
from lib.windows.MainWindow import MainWindow

styleConfig = {
    "font": {
        "text": ("Calibri", 12),
        "buttonText": ("Calibri", 10),
        "progressButton": ("Calibri", 10, "bold"),
        "h1": ("Calibri", 20, "bold"),
        "h2": ("Calibri", 16),
        "h3": ("Calibri", 14)
    },
    "colors": {
        "bg": "white",                          # Default: white
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
    "buttonHeight" : 1,
    "buttonWidth": 12, 
    "buttonRelief": "raised",                   # Default: raised
    "paddings": {
        "default": 20,                          # Default: 20
        "tight": 5,                             # Default: 5
        "slim" : 10,                            # Default: 10
        "wide" : 30                             # Default: 30
    },
}

root = MainWindow(styleConfig)
root.mainloop()
