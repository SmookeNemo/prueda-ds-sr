"""Preparación específica del dataset (term_deposit). La usan train.py y predict.py,
así los datos nuevos reciben exactamente las mismas transformaciones."""
import numpy as np
import pandas as pd

TARGET = "subscribed"
MAPEO_TARGET = {"yes": 1, "no": 0}

# duration: solo se conoce DESPUÉS de la llamada -> leakage, se excluye del modelo
COLS_EXCLUIR = ["duration"]


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=COLS_EXCLUIR, errors="ignore").copy()
    if "pdays" in df.columns:
        # 999 = nunca contactado antes: es un código, no una cantidad de días
        df["contactado_antes"] = (df["pdays"] != 999).astype(int)
        df["pdays"] = df["pdays"].replace(999, np.nan)
    return df


def preparar_target(y: pd.Series) -> pd.Series:
    return y.map(MAPEO_TARGET).astype(int)
