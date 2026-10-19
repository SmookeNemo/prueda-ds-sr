"""Funciones rápidas para el Análisis Exploratorio de Datos (EDA)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def resumen_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Tipo, nulos, % nulos, valores únicos y un ejemplo por columna."""
    resumen = pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "n_nulos": df.isna().sum(),
        "pct_nulos": (df.isna().mean() * 100).round(2),
        "n_unicos": df.nunique(),
        "ejemplo": df.iloc[0] if len(df) else None,
    })
    print(f"Shape: {df.shape} | Duplicados: {df.duplicated().sum()}")
    return resumen.sort_values("pct_nulos", ascending=False)


def distribucion_target(y: pd.Series) -> pd.DataFrame:
    """Conteo y % por clase, más el ratio de desbalance."""
    tabla = pd.DataFrame({"n": y.value_counts(), "pct": (y.value_counts(normalize=True) * 100).round(2)})
    ratio = tabla["n"].max() / tabla["n"].min()
    print(f"Ratio de desbalance (mayoritaria/minoritaria): {ratio:.2f} : 1")
    return tabla


def tipos_variables(df: pd.DataFrame, target: str, max_unicos_cat: int = 15, excluir=None):
    """Separa columnas numéricas y categóricas (numéricas con pocos únicos se tratan como categóricas)."""
    excluir = set(excluir or []) | {target}
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c in excluir:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > max_unicos_cat:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    print(f"Numéricas ({len(num_cols)}): {num_cols}")
    print(f"Categóricas ({len(cat_cols)}): {cat_cols}")
    return num_cols, cat_cols


def detectar_columnas_problema(df: pd.DataFrame, umbral_nulos: float = 0.6) -> dict:
    """Columnas constantes, tipo ID (todo único) y con demasiados nulos."""
    n = len(df)
    res = {
        "constantes": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
        "tipo_id": [c for c in df.columns if df[c].nunique() == n],
        "muchos_nulos": [c for c in df.columns if df[c].isna().mean() > umbral_nulos],
    }
    for k, v in res.items():
        print(f"{k}: {v}")
    return res


def graficar_numericas_vs_target(df: pd.DataFrame, num_cols: list, target: str, ncols: int = 3):
    """Boxplot de cada variable numérica separada por clase del target."""
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, num_cols):
        sns.boxplot(data=df, x=target, y=c, ax=ax)
        ax.set_title(c)
    for ax in axes[len(num_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def tasa_target_por_categoria(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """Para una categórica: n por categoría y tasa del evento (target=1)."""
    return (df.groupby(col, dropna=False)[target]
              .agg(n="count", tasa_evento="mean")
              .sort_values("tasa_evento", ascending=False))


def graficar_correlacion(df: pd.DataFrame, num_cols: list, figsize=(10, 8)):
    """Heatmap de correlación de Spearman entre numéricas."""
    corr = df[num_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=len(num_cols) <= 15, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlación (Spearman)")
    fig.tight_layout()
    return fig


def muestreo_estratificado(df: pd.DataFrame, target: str, n: int = None, frac: float = None,
                           random_state: int = 42) -> pd.DataFrame:
    """Muestra estratificada por el target (útil si el dataset es muy grande para la hora)."""
    if n is None and frac is None:
        raise ValueError("Indica n o frac")
    tamano = n if n is not None else frac
    if (n is not None and n >= len(df)) or (frac is not None and frac >= 1):
        return df.copy()
    muestra, _ = train_test_split(df, train_size=tamano, stratify=df[target], random_state=random_state)
    print(f"Muestra: {len(muestra)} filas | tasa evento original {df[target].mean():.4f} "
          f"vs muestra {muestra[target].mean():.4f}")
    return muestra
