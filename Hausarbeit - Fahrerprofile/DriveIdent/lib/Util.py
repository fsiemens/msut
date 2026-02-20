import pandas as pd
from pathlib import Path

def getModelAccuracyData(path : str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except:
        print("Ergebnis.csv not found")
        return pd.DataFrame()

def getPlotPaths(path : str):
    return [str(p) for p in Path(path).rglob("*.png")]