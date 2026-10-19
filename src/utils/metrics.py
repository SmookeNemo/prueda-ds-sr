"""Métricas completas, evaluación train/val/test con umbral 0.5 y óptimo, gaps de sobreajuste
y gráficos (ROC, matriz de confusión)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve)


def ks_statistic(y_true, y_prob) -> float:
    """KS = máxima separación entre las distribuciones acumuladas de ambas clases."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _div(a, b):
    return float(a / b) if b else 0.0


def metricas_completas(y_true, y_prob, umbral: float = 0.5) -> dict:
    """Todas las métricas de la matriz de confusión + las de la curva."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "umbral": umbral,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": _div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "npv": _div(tn, tn + fn),
        "recall": recall,                       # sensibilidad / TPR
        "especificidad": _div(tn, tn + fp),     # TNR
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "f1": _div(2 * precision * recall, precision + recall),
        # No dependen del umbral
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def evaluar_modelo(modelo, conjuntos: dict, umbral_opt: float, nombre: str = "") -> pd.DataFrame:
    """Evalúa en cada conjunto ({'train': (X, y), 'val': (X, y), 'test': (X, y)})
    con umbral 0.5 y con el óptimo. Devuelve una fila por (conjunto, umbral)."""
    filas = []
    for set_name, (X, y) in conjuntos.items():
        p = modelo.predict_proba(X)[:, 1]
        for tipo, u in [("0.5", 0.5), ("optimo", umbral_opt)]:
            m = metricas_completas(y, p, u)
            filas.append({"modelo": nombre, "conjunto": set_name, "tipo_umbral": tipo, **m})
    return pd.DataFrame(filas)


def calcular_gaps(df_eval: pd.DataFrame) -> dict:
    """Brechas de sobreajuste (AUC y F1 con umbral óptimo): train-val y val-test."""
    opt = df_eval[df_eval["tipo_umbral"] == "optimo"].set_index("conjunto")
    gaps = {}
    for met in ["auc", "f1", "ks"]:
        if {"train", "val"} <= set(opt.index):
            gaps[f"gap_{met}_train_val"] = opt.loc["train", met] - opt.loc["val", met]
        if {"val", "test"} <= set(opt.index):
            gaps[f"gap_{met}_val_test"] = opt.loc["val", met] - opt.loc["test", met]
    return gaps


def diagnostico_sobreajuste(gaps: dict, tolerancia: float = 0.05) -> str:
    g = gaps.get("gap_auc_train_val", 0)
    if g > 2 * tolerancia:
        return "Sobreajuste grave"
    if g > tolerancia:
        return "Sobreajuste moderado"
    return "OK"


# ----------------------------------------------------------------- Gráficos
def graficar_roc(curvas: dict, titulo: str = "Curva ROC"):
    """curvas: {'nombre': (y_true, y_prob), ...}. Varias curvas en el mismo gráfico."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for nombre, (y, p) in curvas.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray")
    ax.set_xlabel("FPR (1 - especificidad)")
    ax.set_ylabel("TPR (recall)")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def graficar_matriz_confusion(y_true, y_prob, umbral: float, titulo: str = ""):
    y_pred = (np.asarray(y_prob) >= umbral).astype(int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]), display_labels=[0, 1]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(titulo or f"Matriz de confusión (umbral={umbral:.2f})")
    fig.tight_layout()
    return fig
