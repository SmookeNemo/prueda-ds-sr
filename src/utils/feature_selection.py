"""Selección de variables: Information Value, correlación, VIF y feature importance.

Todo se calcula SOLO con train.
- Set árboles (XGBoost, RF): IV + importancia. La multicolinealidad no afecta la predicción.
- Set logit: IV + filtro de correlación + VIF iterativo.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ----------------------------------------------------------------- Information Value
def calcular_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10):
    """IV de una variable. Numéricas se agrupan por cuantiles; nulos van en su propio grupo.
    Devuelve (iv, tabla_woe)."""
    x = df[feature]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > bins:
        grupos = pd.qcut(x, q=bins, duplicates="drop").astype(str)
    else:
        grupos = x.astype(str)
    grupos = grupos.where(x.notna(), "NULO")

    t = pd.DataFrame({"grupo": grupos, "y": df[target]}).groupby("grupo")["y"].agg(n="count", eventos="sum")
    t["no_eventos"] = t["n"] - t["eventos"]
    # +0.5 evita divisiones por cero / log(0) en grupos sin eventos
    t["dist_eventos"] = (t["eventos"] + 0.5) / (t["eventos"].sum() + 0.5)
    t["dist_no_eventos"] = (t["no_eventos"] + 0.5) / (t["no_eventos"].sum() + 0.5)
    t["woe"] = np.log(t["dist_no_eventos"] / t["dist_eventos"])
    t["iv"] = (t["dist_no_eventos"] - t["dist_eventos"]) * t["woe"]
    return t["iv"].sum(), t


def _poder_iv(iv: float) -> str:
    if iv < 0.02:
        return "Inútil"
    if iv < 0.1:
        return "Débil"
    if iv < 0.3:
        return "Medio"
    if iv < 0.5:
        return "Fuerte"
    return "Sospechoso (revisar leakage)"


def tabla_iv(df: pd.DataFrame, features: list, target: str, bins: int = 10) -> pd.DataFrame:
    """IV de todas las variables, ordenado de mayor a menor."""
    filas = [{"variable": f, "iv": calcular_iv(df, f, target, bins)[0]} for f in features]
    t = pd.DataFrame(filas).sort_values("iv", ascending=False).reset_index(drop=True)
    t["poder"] = t["iv"].apply(_poder_iv)
    return t


def filtrar_por_iv(t_iv: pd.DataFrame, min_iv: float = 0.02, max_iv: float = 0.5):
    """Devuelve (variables que pasan, variables sospechosas por IV alto)."""
    pasan = t_iv.loc[t_iv["iv"] >= min_iv, "variable"].tolist()
    sospechosas = t_iv.loc[t_iv["iv"] > max_iv, "variable"].tolist()
    print(f"Pasan IV >= {min_iv}: {len(pasan)} de {len(t_iv)}")
    if sospechosas:
        print(f"OJO, IV > {max_iv} (posible leakage): {sospechosas}")
    return pasan, sospechosas


# ----------------------------------------------------------------- Correlación (set logit)
def filtrar_correlacion(df: pd.DataFrame, num_features: list, umbral: float = 0.8, prioridad: dict = None):
    """Para cada par con |corr| > umbral elimina la variable de menor prioridad (por defecto el IV).
    Devuelve (variables que quedan, lista de pares eliminados)."""
    prioridad = prioridad or {}
    corr = df[num_features].corr(method="spearman").abs()
    pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_features)
             for b in num_features[i + 1:] if corr.loc[a, b] > umbral]
    pares.sort(key=lambda p: -p[2])
    eliminadas, log = set(), []
    for a, b, r in pares:
        if a in eliminadas or b in eliminadas:
            continue
        quitar = a if prioridad.get(a, 0) < prioridad.get(b, 0) else b
        eliminadas.add(quitar)
        log.append({"var_1": a, "var_2": b, "corr": round(r, 3), "eliminada": quitar})
    quedan = [f for f in num_features if f not in eliminadas]
    print(f"Correlación > {umbral}: eliminadas {len(eliminadas)} -> {sorted(eliminadas)}")
    return quedan, pd.DataFrame(log)


# ----------------------------------------------------------------- VIF (set logit)
def calcular_vif(df: pd.DataFrame, num_features: list) -> pd.DataFrame:
    """VIF de cada variable numérica (los nulos se rellenan con la mediana solo para el cálculo)."""
    X = df[num_features].astype(float)
    X = add_constant(X.fillna(X.median()), has_constant="add")
    vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return (pd.DataFrame({"variable": num_features, "vif": vifs})
              .sort_values("vif", ascending=False).reset_index(drop=True))


def vif_iterativo(df: pd.DataFrame, num_features: list, umbral: float = 5.0):
    """Quita una a una la variable de mayor VIF hasta que todas queden <= umbral.
    Devuelve (variables que quedan, tabla VIF final, variables eliminadas)."""
    feats, eliminadas = list(num_features), []
    while len(feats) > 1:
        t = calcular_vif(df, feats)
        if t["vif"].iloc[0] <= umbral:
            break
        peor = t["variable"].iloc[0]
        eliminadas.append((peor, round(t["vif"].iloc[0], 2)))
        feats.remove(peor)
    print(f"VIF > {umbral}: eliminadas {eliminadas}")
    return feats, calcular_vif(df, feats) if len(feats) > 1 else pd.DataFrame(), eliminadas


# ----------------------------------------------------------------- Feature importance (set árboles)
def importancia_xgb(X: pd.DataFrame, y, sample_weight=None, random_state: int = 42) -> pd.DataFrame:
    """Importancia (gain) con un XGBoost rápido. Acepta categóricas sin codificar."""
    from xgboost import XGBClassifier
    Xc = X.copy()
    for c in Xc.columns:
        if not pd.api.types.is_numeric_dtype(Xc[c]):
            Xc[c] = Xc[c].astype("category")
    modelo = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                           enable_categorical=True, random_state=random_state, n_jobs=-1, verbosity=0)
    modelo.fit(Xc, y, sample_weight=sample_weight)
    gain = modelo.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({"variable": X.columns, "importancia": [gain.get(c, 0.0) for c in X.columns]})
    imp["importancia"] = imp["importancia"] / imp["importancia"].sum()
    imp = imp.sort_values("importancia", ascending=False).reset_index(drop=True)
    imp["acumulado"] = imp["importancia"].cumsum()
    return imp


def seleccionar_por_importancia(imp: pd.DataFrame, acumulado: float = 0.95, min_importancia: float = 0.0):
    """Variables que explican el `acumulado` de la importancia total."""
    corte = imp.loc[imp["acumulado"] <= acumulado]
    # incluye la variable que cruza el umbral
    n = min(len(corte) + 1, len(imp))
    sel = imp.head(n)
    sel = sel.loc[sel["importancia"] > min_importancia, "variable"].tolist()
    print(f"Importancia: {len(sel)} variables explican ~{acumulado:.0%}")
    return sel
