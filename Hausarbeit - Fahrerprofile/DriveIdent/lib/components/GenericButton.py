import tkinter as tk
from typing import Callable

class GenericButton(tk.Button):
    
    def __init__(self, parent, styleConfig : dict, text : str, command : Callable, width : int = -1, height : int = -1):
        super().__init__(
            parent, 
            text=text, 
            width=styleConfig["buttonWidth"] if width < 0 else width,
            height=styleConfig["buttonHeight"] if height < 0 else height, 
            command=command,
            #relief=styleConfig["buttonRelief"], 
            #bg=styleConfig["colors"]["buttonBg"], 
            #fg=styleConfig["colors"]["buttonFg"],
            #font=styleConfig["font"]["buttonText"]
        )