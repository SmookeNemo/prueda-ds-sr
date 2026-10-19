"""Split, pipelines de preprocesamiento, challenge de modelos y tuneo de hiperparámetros."""
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .metrics import calcular_gaps, diagnostico_sobreajuste, evaluar_modelo
from .outliers import IQRCapper
from .threshold import buscar_umbral, umbral_youden


# ----------------------------------------------------------------- Split
def split_train_val_test(X, y, val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
    """Split estratificado 70/15/15 por defecto. Hacerlo ANTES de cualquier transformación."""
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=random_state)
    for n, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{n:5s}: {len(yy):>7} filas | tasa evento {np.mean(yy):.4f}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------------------------------------- Preprocesamiento
def construir_preprocesador(num_cols: list, cat_cols: list, escalar: bool = True,
                            capping: bool = True, factor_iqr: float = 1.5) -> ColumnTransformer:
    """Numéricas: imputación mediana -> capping IQR -> escalado (opcional).
    Categóricas: imputación moda -> One-Hot. Columnas no listadas se descartan."""
    pasos_num = [("imputar", SimpleImputer(strategy="median"))]
    if capping:
        pasos_num.append(("iqr", IQRCapper(factor=factor_iqr)))
    if escalar:
        pasos_num.append(("escalar", StandardScaler()))
    pasos_cat = [("imputar", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(pasos_num), list(num_cols)))
    if cat_cols:
        transformers.append(("cat", Pipeline(pasos_cat), list(cat_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def construir_pipeline(modelo, num_cols: list, cat_cols: list, escalar: bool = True,
                       capping: bool = True) -> Pipeline:
    """Pipeline = preprocesamiento + modelo. El paso del modelo se llama 'model'
    (por eso los pesos se pasan como model__sample_weight)."""
    return Pipeline([("prep", construir_preprocesador(num_cols, cat_cols, escalar, capping)),
                     ("model", modelo)])


def modelos_base(random_state: int = 42) -> dict:
    """Modelos candidatos con parámetros razonables. Los pesos se pasan en fit, no aquí."""
    return {
        "logit": LogisticRegression(max_iter=2000, solver="lbfgs"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                                n_jobs=-1, random_state=random_state),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                                 n_jobs=-1, random_state=random_state, verbosity=0),
    }


# ----------------------------------------------------------------- Challenge
def entrenar_y_evaluar(nombre: str, pipeline: Pipeline, data: dict, w_train, metrica_umbral=None) -> dict:
    """Entrena con pesos, busca umbral en validation y evalúa en train/val/test.
    data = {'train': (X, y), 'val': (X, y), 'test': (X, y)}."""
    from sklearn.metrics import f1_score
    metrica_umbral = metrica_umbral or f1_score
    X_tr, y_tr = data["train"]
    pipeline.fit(X_tr, y_tr, model__sample_weight=w_train)

    X_val, y_val = data["val"]
    p_val = pipeline.predict_proba(X_val)[:, 1]
    umbral_opt, score_opt, _, _ = buscar_umbral(y_val, p_val, metrica=metrica_umbral)

    df_eval = evaluar_modelo(pipeline, data, umbral_opt, nombre)
    gaps = calcular_gaps(df_eval)
    return {"nombre": nombre, "pipeline": pipeline, "umbral": umbral_opt,
            "umbral_youden": umbral_youden(y_val, p_val), "metricas": df_eval, "gaps": gaps,
            "diagnostico": diagnostico_sobreajuste(gaps)}


def tabla_comparativa(resultados: dict) -> pd.DataFrame:
    """Una fila por modelo: métricas de validation con umbral óptimo + gaps + diagnóstico."""
    filas = []
    for nombre, r in resultados.items():
        m = r["metricas"]
        val_opt = m[(m["conjunto"] == "val") & (m["tipo_umbral"] == "optimo")].iloc[0]
        val_05 = m[(m["conjunto"] == "val") & (m["tipo_umbral"] == "0.5")].iloc[0]
        filas.append({
            "modelo": nombre, "umbral": r["umbral"],
            "auc_val": val_opt["auc"], "gini_val": val_opt["gini"], "ks_val": val_opt["ks"],
            "pr_auc_val": val_opt["pr_auc"],
            "f1_val_0.5": val_05["f1"], "f1_val_opt": val_opt["f1"],
            "recall_val_opt": val_opt["recall"], "precision_val_opt": val_opt["precision"],
            **{k: v for k, v in r["gaps"].items() if k.startswith(("gap_auc", "gap_f1"))},
            "diagnostico": r["diagnostico"],
        })
    return pd.DataFrame(filas).sort_values("auc_val", ascending=False).reset_index(drop=True)


def elegir_campeon(tabla: pd.DataFrame, metrica: str = "auc_val", max_gap: float = 0.05,
                   empate: float = 0.005, preferir_simple: str = "logit") -> str:
    """1) descarta modelos con gap AUC train-val > max_gap; 2) mejor `metrica` en validation;
    3) si el modelo simple está a menos de `empate` del mejor, se prefiere el simple."""
    validos = tabla[tabla["gap_auc_train_val"] <= max_gap]
    descartados = tabla.loc[tabla["gap_auc_train_val"] > max_gap, "modelo"].tolist()
    if descartados:
        print(f"Descartados por sobreajuste (gap AUC > {max_gap}): {descartados}")
    if validos.empty:
        print("Todos superan el gap máximo; se elige el de menor gap.")
        return tabla.sort_values("gap_auc_train_val").iloc[0]["modelo"]
    validos = validos.sort_values(metrica, ascending=False)
    mejor = validos.iloc[0]
    simple = validos[validos["modelo"] == preferir_simple]
    if mejor["modelo"] != preferir_simple and not simple.empty and mejor[metrica] - simple.iloc[0][metrica] <= empate:
        print(f"Empate técnico: se prefiere {preferir_simple} por simplicidad.")
        return preferir_simple
    return mejor["modelo"]


# ----------------------------------------------------------------- Tuneo
def espacios_busqueda() -> dict:
    """Distribuciones para RandomizedSearchCV (prefijo model__ por el Pipeline)."""
    return {
        "logit": {"model__C": loguniform(1e-3, 1e2)},
        "random_forest": {"model__n_estimators": randint(200, 600),
                          "model__max_depth": randint(3, 15),
                          "model__min_samples_leaf": randint(5, 100),
                          "model__max_features": ["sqrt", "log2", 0.5]},
        "xgboost": {"model__n_estimators": randint(150, 700),
                    "model__max_depth": randint(2, 8),
                    "model__learning_rate": loguniform(0.01, 0.3),
                    "model__subsample": uniform(0.6, 0.4),
                    "model__colsample_bytree": uniform(0.6, 0.4),
                    "model__min_child_weight": randint(1, 20),
                    "model__reg_lambda": loguniform(1e-2, 10)},
    }


def tunear_modelo(pipeline: Pipeline, param_dist: dict, X_train, y_train, w_train,
                  n_iter: int = 20, cv: int = 3, scoring: str = "roc_auc", random_state: int = 42):
    """RandomizedSearchCV con CV estratificada; los pesos se reparten por fold automáticamente."""
    busqueda = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=n_iter, scoring=scoring,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
        random_state=random_state, n_jobs=-1, refit=True, verbose=0)
    busqueda.fit(X_train, y_train, model__sample_weight=w_train)
    print(f"Mejor {scoring} CV: {busqueda.best_score_:.4f}")
    print(f"Mejores parámetros: {busqueda.best_params_}")
    return busqueda.best_estimator_, busqueda.best_params_, busqueda.best_score_
