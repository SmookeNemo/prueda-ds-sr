"""Búsqueda del umbral óptimo de corte. SIEMPRE se elige en validation, nunca en test.
Cada modelo tiene su propio umbral."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve


def _pred(y_prob, u):
    return (np.asarray(y_prob) >= u).astype(int)


def buscar_umbral(y_true, y_prob, metrica=f1_score, paso: float = 0.01):
    """Itera umbrales de `paso` a 1-paso y devuelve el que maximiza `metrica`.
    metrica: cualquier función f(y_true, y_pred), p. ej. f1_score, recall_score,
             o functools.partial(fbeta_score, beta=2).
    Devuelve (umbral_opt, score_opt, umbrales, scores)."""
    umbrales = np.round(np.arange(paso, 1, paso), 4)
    scores = []
    for u in umbrales:
        try:
            scores.append(metrica(y_true, _pred(y_prob, u), zero_division=0))
        except TypeError:  # métricas sin zero_division
            scores.append(metrica(y_true, _pred(y_prob, u)))
    scores = np.array(scores)
    i = int(np.argmax(scores))
    return float(umbrales[i]), float(scores[i]), umbrales, scores


def umbral_youden(y_true, y_prob) -> float:
    """Umbral que maximiza TPR - FPR (Youden J). Se reporta como referencia."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return float(min(thr[np.argmax(tpr - fpr)], 1.0))


def umbral_con_restriccion(y_true, y_prob, metrica_objetivo=precision_score,
                           metrica_restriccion=recall_score, minimo: float = 0.8, paso: float = 0.01):
    """Maximiza metrica_objetivo sujeto a metrica_restriccion >= minimo
    (p. ej. mejor precisión con recall de al menos 80%)."""
    mejor_u, mejor_s = None, -np.inf
    for u in np.round(np.arange(paso, 1, paso), 4):
        p = _pred(y_prob, u)
        if metrica_restriccion(y_true, p, zero_division=0) >= minimo:
            s = metrica_objetivo(y_true, p, zero_division=0)
            if s > mejor_s:
                mejor_u, mejor_s = float(u), float(s)
    if mejor_u is None:
        print("Ningún umbral cumple la restricción")
    return mejor_u, mejor_s


def graficar_umbral(y_true, y_prob, umbral_opt: float = None, paso: float = 0.01, titulo: str = ""):
    """Precision, recall y F1 en función del umbral, marcando el óptimo y el 0.5."""
    umbrales = np.round(np.arange(paso, 1, paso), 4)
    prec = [precision_score(y_true, _pred(y_prob, u), zero_division=0) for u in umbrales]
    rec = [recall_score(y_true, _pred(y_prob, u), zero_division=0) for u in umbrales]
    f1 = [f1_score(y_true, _pred(y_prob, u), zero_division=0) for u in umbrales]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(umbrales, prec, label="Precision")
    ax.plot(umbrales, rec, label="Recall")
    ax.plot(umbrales, f1, label="F1", lw=2.5)
    ax.axvline(0.5, ls=":", c="gray", label="Umbral 0.5")
    if umbral_opt is not None:
        ax.axvline(umbral_opt, ls="--", c="red", label=f"Óptimo = {umbral_opt:.2f}")
    ax.set_xlabel("Umbral")
    ax.set_ylabel("Métrica (validation)")
    ax.set_title(titulo or "Métricas vs umbral")
    ax.legend()
    fig.tight_layout()
    return fig
