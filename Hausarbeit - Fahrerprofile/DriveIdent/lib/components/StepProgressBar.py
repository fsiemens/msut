import tkinter as tk
from tkinter import ttk
from typing import Callable

class StepProgress(tk.Frame):
    '''
    A stepped progress bar with labeled steps
    '''

    def __init__(self, parent, styleConfig : dict, labels : list[tuple[str, Callable]], stepAccessValidationFct : Callable[[int], bool] = lambda step : True):
        '''
        Creates the Stepped Progress Bar. The amount of steps is dependent on the amount of labels.
        
        :param self: Self
        :param parent: A tkinter object to use as the parent for this object
        :param labels: Labels for the steps. Must not be empty.
        :type labels: list[str]
        :param stepAccessValidationFct: This Lambda Function provides an integer - the index of the step - to determine if the step can be accessed by clicking on it. If this function returns true, the step can be accessed. 
        :type stepAccessValidationFct: Callable[[int], bool]
        '''
        super().__init__(parent)

        if len(labels) <= 0:
            raise ValueError("Labels must not be empty")

        self.styleConfig = styleConfig
        self.steps = steps = len(labels)
        self.current = tk.IntVar(value=1)
        self.selected = tk.IntVar(value=1)
        self.accessFct = stepAccessValidationFct
        self.actions = [lbl[1] for lbl in labels]

        # Progressbar
        self.progress = ttk.Progressbar(
            self,
            orient="horizontal",
            length=steps*200 + 75,
            mode="determinate",
            maximum=steps - 1
        )
        self.progress.grid(row=0, column=0, columnspan=steps, pady=15)

        # Step Buttons
        self.buttons = []
        for i in range(steps):
            step = i + 1

            rb = tk.Radiobutton(
                self,
                text=labels[step - 1][0],
                variable=self.selected,
                value=step,
                width=13,
                indicatoron=False,
                command=self.onButtonSelect,
                font=styleConfig["font"]["progressButton"],
                fg=styleConfig["colors"]["buttonFg"],
                bg=styleConfig["colors"]["buttonBg"],
                selectcolor=styleConfig["colors"]["stepperActive"],
                relief=styleConfig["buttonRelief"]
            )

            rb.grid(row=0, column=i, padx=75)
            self.buttons.append(rb)

        self.updateProgress()

    def onButtonSelect(self):
        current = self.current.get()
        selected = self.selected.get()

        if self.accessFct(selected):
            self.current.set(selected)
            self.actions[selected -1]()
        else:
            self.selected.set(current)

        self.updateProgress()

    def updateProgress(self):
        current = self.current.get()

        self.progress["value"] = current - 1

        for i, btn in enumerate(self.buttons):
            step = i + 1

            if step < current:
                btn.config(bg=self.styleConfig["colors"]["stepperDone"], selectcolor=self.styleConfig["colors"]["stepperDone"])
            elif step == current:
                btn.config(bg=self.styleConfig["colors"]["stepperActive"], selectcolor=self.styleConfig["colors"]["stepperActive"])
            else:
                btn.config(bg=self.styleConfig["colors"]["buttonBg"], selectcolor=self.styleConfig["colors"]["buttonSelect"])

    def getCurrentStep(self):
        return self.current.get()
    
    def setCurrentStep(self, value : int):
        self.current.set(value)
        self.updateProgress()

    def next(self):
        if self.getCurrentStep() < self.steps:
            self.setCurrentStep(self.getCurrentStep() + 1)

    def back(self):
        if self.getCurrentStep() > 1:
            self.setCurrentStep(self.getCurrentStep() - 1)