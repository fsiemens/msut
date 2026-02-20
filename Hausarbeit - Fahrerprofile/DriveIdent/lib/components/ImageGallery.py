import tkinter as tk
from PIL import Image, ImageTk
from .GenericButton import GenericButton


class ImageGallery(tk.Frame):

    def __init__(self, parent, styleConfig, images: list[str]):
        super().__init__(parent)

        self.images = images
        self.index = 0

        # Originalbild speichern
        self.original_img = None
        self.tk_img = None

        # Bildanzeige
        self.imageLabel = tk.Label(self, bd=0)
        self.imageLabel.pack(fill="both", expand=True)

        # Buttons unten
        button_frame = tk.Frame(self)
        button_frame.pack(pady=styleConfig["paddings"]["tight"])

        self.prevButton = GenericButton(
            button_frame, styleConfig,
            text="Zurück",
            command=self.prevImage
        )
        self.prevButton.pack(side="left", padx=5)

        self.nextButton = GenericButton(
            button_frame, styleConfig,
            text="Weiter",
            command=self.nextImage
        )
        self.nextButton.pack(side="left", padx=5)

        # Resize-Event binden
        self.imageLabel.bind("<Configure>", self.onResize)

        # Erstes Bild laden
        self.loadImage()

    # -----------------------------------------
    def loadImage(self):
        """Originalbild laden"""
        path = self.images[self.index]
        self.original_img = Image.open(path)
        self.resizeAndShow()
        self.prevButton.config(state="normal" if self.index > 0 else "disabled")
        self.nextButton.config(state="normal" if self.index < len(self.images)-1 else "disabled")

    # -----------------------------------------
    def resizeAndShow(self):
        """Bild an aktuelle Label-Größe anpassen"""

        if self.original_img is None:
            return

        w = self.imageLabel.winfo_width()
        h = self.imageLabel.winfo_height()

        if w < 10 or h < 10:
            return  # noch nicht gerendert

        # Seitenverhältnis behalten
        img = self.original_img.copy()
        img.thumbnail((w, h))

        self.tk_img = ImageTk.PhotoImage(img)
        self.imageLabel.config(image=self.tk_img)

    # -----------------------------------------
    def onResize(self, event):
        """Wird bei Größenänderung ausgelöst"""
        self.resizeAndShow()

    # -----------------------------------------
    def nextImage(self):
        if self.index < len(self.images) - 1:
            self.index += 1
            self.loadImage()

    def prevImage(self):
        if self.index > 0:
            self.index -= 1
            self.loadImage()
