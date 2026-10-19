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

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - self.factor * iqr).to_numpy()
        self.upper_ = (q3 + self.factor * iqr).to_numpy()
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
                      "n_alto": n_alto, "pct_atipicos": round(100 * (n_bajo + n_alto) / max(len(s), 1), 2)})
    return pd.DataFrame(filas).sort_values("pct_atipicos", ascending=False).reset_index(drop=True)
