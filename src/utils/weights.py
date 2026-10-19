"""Balanceo por pesos. Se calcula UNA vez con y_train y se pasa a todos los modelos
con `pipeline.fit(X, y, model__sample_weight=w)`.

Peso de la clase c:  w_c = n / (k * n_c)
OJO: en XGBoost usa sample_weight O scale_pos_weight, nunca los dos.
Los pesos solo se usan para entrenar; las métricas se calculan sin ponderar.
"""
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def pesos_por_clase(y) -> dict:
    clases = np.unique(y)
    pesos = compute_class_weight(class_weight="balanced", classes=clases, y=y)
    return dict(zip(clases.tolist(), pesos.round(4).tolist()))


def calcular_pesos(y) -> np.ndarray:
    """Vector sample_weight (una entrada por fila de train)."""
    w = compute_sample_weight(class_weight="balanced", y=y)
    print(f"Pesos por clase: {pesos_por_clase(y)}")
    return w


def scale_pos_weight(y) -> float:
    """Alternativa propia de XGBoost: n_negativos / n_positivos (no combinar con sample_weight)."""
    y = pd.Series(y)
    return float((y == 0).sum() / (y == 1).sum())
