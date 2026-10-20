"""Tratamiento de atípicos con IQR (capping / winsorización).

Los límites se calculan SOLO con train (en fit) y se aplican igual a validation y test.
IQRCapper es un transformer de sklearn, así que va dentro del Pipeline y queda
guardado con el modelo en MLflow.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Recorta cada columna a [Q1 - factor*IQR, Q3 + factor*IQR]."""

    def __init__(self, factor: float = 1.5, columnas=None):
        self.factor = factor
        self.columnas = columnas   # None = todas; lista de posiciones (int) = solo esas

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        lower = (q1 - self.factor * iqr).to_numpy(dtype=float, copy=True)
        upper = (q3 + self.factor * iqr).to_numpy(dtype=float, copy=True)
        # IQR = 0 (variables con muchos ceros o un valor dominante, p. ej. pdays=999):
        # el capping las volvería constantes, así que esas columnas no se recortan.
        sin_rango = (iqr == 0).to_numpy(copy=True)
        if self.columnas is not None:   # las columnas no elegidas no se recortan
            elegidas = np.zeros(X_df.shape[1], dtype=bool)
            elegidas[list(self.columnas)] = True
            sin_rango |= ~elegidas
        lower[sin_rango], upper[sin_rango] = -np.inf, np.inf
        self.cols_sin_capping_ = list(X_df.columns[sin_rango])
        self.lower_, self.upper_ = lower, upper
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def resumen_atipicos(df: pd.DataFrame, num_cols: list, factor: float = 1.5) -> pd.DataFrame:
    """Tabla con límites IQR y cantidad / % de atípicos por variable (para el EDA)."""
    filas = []
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n_bajo, n_alto = (s < lo).sum(), (s > hi).sum()
        filas.append({"variable": c, "lim_inf": lo, "lim_sup": hi, "n_bajo": n_bajo,
                      "n_alto": n_alto, "pct_atipicos": round(100 * (n_bajo + n_alto) / max(len(s), 1), 2),
                      "nota": "IQR=0: no se aplica capping" if iqr == 0 else ""})
    return pd.DataFrame(filas).sort_values("pct_atipicos", ascending=False).reset_index(drop=True)


def eliminar_atipicos_iqr(X: pd.DataFrame, y: pd.Series, cols: list, factor: float = 1.5):
    """Elimina las FILAS con algún valor fuera de [Q1 - factor*IQR, Q3 + factor*IQR].
    Usar SOLO sobre train (después del split). Validation y test no se tocan.
    Columnas con IQR = 0 se omiten (se eliminaría todo lo distinto al valor dominante).
    Los nulos no cuentan como atípicos. Devuelve (X_filtrado, y_filtrado, tabla_resumen)."""
    mascara = pd.Series(True, index=X.index)
    filas = []
    for c in cols:
        s = X[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            filas.append({"variable": c, "filas_atipicas": 0, "nota": "IQR=0: omitida"})
            continue
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        fuera = s.notna() & ((s < lo) | (s > hi))
        filas.append({"variable": c, "filas_atipicas": int(fuera.sum()), "nota": ""})
        mascara &= ~fuera
    resumen = pd.DataFrame(filas).sort_values("filas_atipicas", ascending=False).reset_index(drop=True)
    n_antes, n_despues = len(X), int(mascara.sum())
    print(f"Filas eliminadas: {n_antes - n_despues} de {n_antes} ({100 * (n_antes - n_despues) / n_antes:.2f}%)")
    print(f"Tasa de evento: {y.mean():.4f} -> {y[mascara].mean():.4f}")
    return X[mascara], y[mascara], resumen
