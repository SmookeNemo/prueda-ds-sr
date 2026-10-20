"""Entrenamiento supervisado end-to-end. Ejecutar desde la raíz:  python -m src.train
Cambia solo la sección CONFIG para el dataset de la prueba."""
import json

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.features import TARGET, preparar_features, preparar_target
from src.utils import *

# ============================================================ CONFIG
RUTA_DATOS = "data/term_deposit.csv"
EXPERIMENTO = "prueba_supervisado"
NOMBRE_MODELO = "modelo_clasificacion"
RANDOM_STATE = 42
TUNEAR_TOP = 2                     # cuántos modelos del challenge se tunean
N_ITER_TUNEO = 10
# Atípicos (IQR 1.5). Cada variable numérica cae en uno de tres grupos:
ATIPICOS_ELIMINAR = []             # elimina las FILAS de train con atípicos en estas variables
ATIPICOS_CAPPING = None            # None = todas las demás numéricas | [] = ninguna | lista = solo esas
                                   # las numéricas que no estén en ninguna lista quedan sin tratamiento
GUARDAR_MODELOS_CHALLENGE = True   # guarda el modelo en cada run (permite registrarlo después)


def main(df: pd.DataFrame = None):
    # ---------------------------------------------------- 1. Datos
    df = df if df is not None else pd.read_csv(RUTA_DATOS)
    df = df.drop_duplicates().reset_index(drop=True)
    X = preparar_features(df.drop(columns=[TARGET]))
    y = preparar_target(df[TARGET])

    # ---------------------------------------------------- 2. Split (antes de transformar)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, random_state=RANDOM_STATE)
    num_cols, cat_cols = tipos_variables(pd.concat([X_train, y_train], axis=1), TARGET)

    # ---------------------------------------------------- 2b. Atípicos
    # Eliminar: quita filas SOLO de train (val/test no se tocan: en producción no se borran clientes).
    # Va antes de la selección y de los pesos, porque cambia X_train e y_train.
    # Capping: se aplica dentro del Pipeline (límites de train, mismos en val/test/producción).
    eliminar = [c for c in ATIPICOS_ELIMINAR if c in num_cols]
    if ATIPICOS_CAPPING is None:
        capping = [c for c in num_cols if c not in eliminar]
    else:
        capping = [c for c in ATIPICOS_CAPPING if c in num_cols]
    repetidas = set(eliminar) & set(capping)
    if repetidas:
        raise ValueError(f"Variables en ELIMINAR y CAPPING a la vez: {sorted(repetidas)}")
    sin_tratar = [c for c in num_cols if c not in eliminar and c not in capping]
    print(f"Atípicos -> eliminar filas: {eliminar} | capping: {capping} | sin tratamiento: {sin_tratar}")
    if eliminar:
        X_train, y_train, resumen_elim = eliminar_atipicos_iqr(X_train, y_train, eliminar)
        print(resumen_elim)

    # ---------------------------------------------------- 3. Selección de variables (solo train)
    train_df = pd.concat([X_train, y_train], axis=1)
    t_iv = tabla_iv(train_df, num_cols + cat_cols, TARGET)
    print(t_iv)
    pasan_iv, _ = filtrar_por_iv(t_iv)
    iv_dict = dict(zip(t_iv["variable"], t_iv["iv"]))

    # Set árboles: IV + importancia
    w_train = calcular_pesos(y_train)
    imp = importancia_xgb(X_train[pasan_iv], y_train, sample_weight=w_train, random_state=RANDOM_STATE)
    feats_arboles = seleccionar_por_importancia(imp, acumulado=0.95)

    # Set logit: IV + correlación + VIF (sobre numéricas); categóricas que pasan IV se mantienen
    num_iv = [c for c in pasan_iv if c in num_cols]
    num_corr, _ = filtrar_correlacion(X_train, num_iv, umbral=0.8, prioridad=iv_dict)
    num_vif, tabla_vif, _ = vif_iterativo(X_train, num_corr, umbral=5)
    feats_logit = num_vif + [c for c in pasan_iv if c in cat_cols]

    json.dump({"arboles": feats_arboles, "logit": feats_logit}, open("features.json", "w"), indent=2)

    def split_cols(feats):
        return [c for c in feats if c in num_cols], [c for c in feats if c in cat_cols]

    # ---------------------------------------------------- 4. Challenge de modelos (con pesos)
    configurar_mlflow(EXPERIMENTO)
    data = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
    base = modelos_base(RANDOM_STATE)
    config = {  # nombre: (features, escalar)
        "logit": (feats_logit, True),
        "random_forest": (feats_arboles, False),
        "xgboost": (feats_arboles, False),
    }

    resultados = {}
    for nombre, (feats, escalar) in config.items():
        nc, cc = split_cols(feats)
        pipe = construir_pipeline(base[nombre], nc, cc, escalar=escalar, capping=capping)
        r = entrenar_y_evaluar(nombre, pipe, data, w_train)
        r["features"] = feats
        p_val = r["pipeline"].predict_proba(X_val)[:, 1]
        figs = {"roc_val": graficar_roc({nombre: (y_val, p_val)}),
                "umbral_val": graficar_umbral(y_val, p_val, r["umbral"], titulo=nombre),
                "cm_val_opt": graficar_matriz_confusion(y_val, p_val, r["umbral"])}
        loguear_run(r, figs, etiqueta="challenge",
                    X_ejemplo=X_train if GUARDAR_MODELOS_CHALLENGE else None)
        resultados[nombre] = r

    tabla = tabla_comparativa(resultados)
    print("\n=== CHALLENGE ===\n", tabla.round(4).to_string())

    # ---------------------------------------------------- 5. Tuneo de los mejores
    espacios = espacios_busqueda()
    for nombre in tabla["modelo"].head(TUNEAR_TOP):
        feats, escalar = config[nombre]
        nc, cc = split_cols(feats)
        pipe = construir_pipeline(base[nombre], nc, cc, escalar=escalar, capping=capping)
        mejor_pipe, mejores_params, _ = tunear_modelo(pipe, espacios[nombre], X_train, y_train, w_train,
                                                      n_iter=N_ITER_TUNEO, random_state=RANDOM_STATE)
        r = entrenar_y_evaluar(f"{nombre}_tuned", mejor_pipe, data, w_train)
        r["features"] = feats
        loguear_run(r, etiqueta="tuneo", params_extra={k.replace("model__", "best_"): v
                                                       for k, v in mejores_params.items()},
                    X_ejemplo=X_train if GUARDAR_MODELOS_CHALLENGE else None)
        resultados[f"{nombre}_tuned"] = r

    tabla = tabla_comparativa(resultados)
    print("\n=== CHALLENGE + TUNEO ===\n", tabla.round(4).to_string())
    tabla.to_csv("tabla_comparativa.csv", index=False)

    # ---------------------------------------------------- 6. Campeón: test se mira aquí
    # Para registrar otro modelo después: python -m src.registrar --modelo <nombre>
    campeon = elegir_campeon(tabla, metrica="auc_val", max_gap=0.05)
    r = resultados[campeon]
    print(f"\nCAMPEÓN: {campeon} | umbral={r['umbral']} | {r['diagnostico']}")
    print(r["metricas"].round(4).to_string())

    p_test = r["pipeline"].predict_proba(X_test)[:, 1]
    figs = {"roc_train_val_test": graficar_roc({
                k: (yy, r["pipeline"].predict_proba(XX)[:, 1]) for k, (XX, yy) in data.items()}),
            "cm_test_opt": graficar_matriz_confusion(y_test, p_test, r["umbral"], "Test - umbral óptimo"),
            "cm_test_05": graficar_matriz_confusion(y_test, p_test, 0.5, "Test - umbral 0.5")}
    registrar_campeon(r, X_train, NOMBRE_MODELO, features=r["features"], figuras=figs)
    return resultados, tabla


if __name__ == "__main__":
    main()
