"""MLflow: tracking de todos los modelos del challenge y registro SOLO del campeón.

- Cada modelo evaluado se loguea como un run (params, métricas, umbral, gráficos y el modelo),
  pero NO se registra.
- El campeón se registra en el Model Registry con alias 'champion' y el umbral como tag.
- Registro manual: registrar_desde_run(run_id) registra cualquier run guardado.
- predict.py carga 'models:/<nombre>@champion' y lee el umbral desde el tag.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

ALIAS = "champion"


def configurar_mlflow(experimento: str, tracking_uri: str = "sqlite:///mlflow.db"):
    """SQLite permite usar el Model Registry en local. UI: mlflow ui --backend-store-uri sqlite:///mlflow.db"""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experimento)
    print(f"MLflow -> {tracking_uri} | experimento: {experimento}")


def _params_simples(pipeline) -> dict:
    modelo = pipeline.named_steps.get("model", pipeline)
    return {k: v for k, v in modelo.get_params().items()
            if isinstance(v, (int, float, str, bool)) and v is not None}


def _metricas_planas(df_eval: pd.DataFrame) -> dict:
    """val_opt_auc, test_0.5_f1, ... (una métrica por conjunto x umbral x métrica)."""
    out = {}
    excluir = {"modelo", "conjunto", "tipo_umbral", "umbral"}
    for _, fila in df_eval.iterrows():
        pref = f"{fila['conjunto']}_{'opt' if fila['tipo_umbral'] == 'optimo' else '0.5'}"
        for col, val in fila.items():
            if col not in excluir:
                out[f"{pref}_{col}"] = float(val)
    return out


def _log_model(pipe, X_ejemplo: pd.DataFrame, registered_model_name: str = None, code_paths: list = None):
    """Guarda el pipeline en el run activo. cloudpickle porque MLflow 3.x usa 'skops' por defecto
    y rechaza clases propias (IQRCapper). Enteros -> float para que la firma acepte nulos."""
    ejemplo = X_ejemplo.head(5).copy()
    for c in ejemplo.select_dtypes("integer").columns:
        ejemplo[c] = ejemplo[c].astype("float64")
    firma = infer_signature(ejemplo, pipe.predict_proba(ejemplo)[:, 1])
    kwargs = dict(sk_model=pipe, signature=firma, input_example=ejemplo,
                  registered_model_name=registered_model_name, code_paths=code_paths,
                  serialization_format="cloudpickle")
    try:
        return mlflow.sklearn.log_model(name="model", **kwargs)          # MLflow 3.x
    except TypeError:
        return mlflow.sklearn.log_model(artifact_path="model", **kwargs)  # MLflow 2.x


def _etiquetar_version(nombre_modelo: str, version, umbral: float, algoritmo: str,
                       features: list = None, alias: str = ALIAS):
    client = MlflowClient()
    client.set_registered_model_alias(nombre_modelo, alias, version)
    client.set_model_version_tag(nombre_modelo, version, "umbral", str(umbral))
    client.set_model_version_tag(nombre_modelo, version, "algoritmo", algoritmo)
    if features:
        client.set_model_version_tag(nombre_modelo, version, "features", ",".join(features))
    print(f"Registrado '{nombre_modelo}' v{version} ({algoritmo}) con alias '@{alias}' | umbral={umbral}")


def loguear_run(resultado: dict, figuras: dict = None, etiqueta: str = "challenge", params_extra: dict = None,
                X_ejemplo: pd.DataFrame = None):
    """Loguea un modelo del challenge SIN registrarlo. resultado = salida de entrenar_y_evaluar().
    Si se pasa X_ejemplo, también guarda el modelo en el run, para poder registrarlo después a mano."""
    with mlflow.start_run(run_name=f"{etiqueta}_{resultado['nombre']}") as run:
        mlflow.set_tag("etapa", etiqueta)
        mlflow.set_tag("modelo", resultado["nombre"])
        mlflow.log_params(_params_simples(resultado["pipeline"]))
        if params_extra:
            mlflow.log_params(params_extra)
        mlflow.log_param("umbral_optimo", resultado["umbral"])
        mlflow.log_metric("umbral_youden", resultado["umbral_youden"])
        mlflow.log_metrics(_metricas_planas(resultado["metricas"]))
        mlflow.log_metrics({k: float(v) for k, v in resultado["gaps"].items()})
        mlflow.set_tag("diagnostico", resultado["diagnostico"])
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"graficos/{nombre}.png")
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = os.path.join(tmp, f"metricas_{resultado['nombre']}.csv")
            resultado["metricas"].to_csv(ruta, index=False)
            mlflow.log_artifact(ruta, artifact_path="tablas")
        if resultado.get("features"):
            mlflow.set_tag("features", ",".join(resultado["features"]))
        if X_ejemplo is not None:
            _log_model(resultado["pipeline"], X_ejemplo)   # guardado, NO registrado
        return run.info.run_id


def registrar_campeon(resultado: dict, X_ejemplo: pd.DataFrame, nombre_modelo: str,
                      features: list = None, figuras: dict = None, code_paths: list = None) -> int:
    """Loguea y REGISTRA el campeón, le asigna el alias 'champion' y guarda el umbral como tag.
    code_paths=['src'] empaqueta el código propio (IQRCapper) junto al modelo."""
    pipe = resultado["pipeline"]
    with mlflow.start_run(run_name=f"campeon_{resultado['nombre']}"):
        mlflow.set_tag("etapa", "campeon")
        mlflow.set_tag("modelo", resultado["nombre"])
        mlflow.log_params(_params_simples(pipe))
        mlflow.log_param("umbral_optimo", resultado["umbral"])
        mlflow.log_metrics(_metricas_planas(resultado["metricas"]))
        mlflow.log_metrics({k: float(v) for k, v in resultado["gaps"].items()})
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"graficos/{nombre}.png")
            plt.close(fig)
        info = _log_model(pipe, X_ejemplo, registered_model_name=nombre_modelo, code_paths=code_paths)

    version = info.registered_model_version
    _etiquetar_version(nombre_modelo, version, resultado["umbral"], resultado["nombre"], features)
    return int(version)


# ----------------------------------------------------------------- Registro manual
def listar_runs(experimento: str) -> pd.DataFrame:
    """Runs del experimento con las métricas clave, para elegir cuál registrar."""
    runs = mlflow.search_runs(experiment_names=[experimento], order_by=["metrics.val_opt_auc DESC"])
    if runs.empty:
        print("No hay runs en el experimento")
        return runs
    cols = {"run_id": "run_id", "tags.etapa": "etapa", "tags.modelo": "modelo",
            "params.umbral_optimo": "umbral", "metrics.val_opt_auc": "auc_val",
            "metrics.val_opt_f1": "f1_val_opt", "metrics.test_opt_auc": "auc_test",
            "metrics.gap_auc_train_val": "gap_auc_train_val"}
    tabla = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    return tabla.round(4).reset_index(drop=True)


def registrar_desde_run(run_id: str, nombre_modelo: str, alias: str = ALIAS) -> int:
    """Registra el modelo guardado en un run específico y le asigna el alias.
    El umbral y las features se toman del propio run."""
    run = MlflowClient().get_run(run_id)
    umbral = float(run.data.params.get("umbral_optimo", 0.5))
    algoritmo = run.data.tags.get("modelo", "desconocido")
    features = run.data.tags.get("features")
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", nombre_modelo)
    except Exception as e:
        raise RuntimeError(f"El run {run_id} no tiene modelo guardado (corre train.py con "
                           f"GUARDAR_MODELOS_CHALLENGE=True). Detalle: {e}")
    _etiquetar_version(nombre_modelo, mv.version, umbral, algoritmo,
                       features.split(",") if features else None, alias)
    return int(mv.version)


def cargar_campeon(nombre_modelo: str, alias: str = ALIAS):
    """Devuelve (modelo sklearn, umbral, features o None)."""
    client = MlflowClient()
    mv = client.get_model_version_by_alias(nombre_modelo, alias)
    modelo = mlflow.sklearn.load_model(f"models:/{nombre_modelo}@{alias}")
    umbral = float(mv.tags.get("umbral", 0.5))
    features = mv.tags.get("features")
    print(f"Cargado '{nombre_modelo}' v{mv.version} (@{alias}) | umbral={umbral}")
    return modelo, umbral, features.split(",") if features else None


def predecir_con_campeon(df: pd.DataFrame, nombre_modelo: str, alias: str = ALIAS) -> pd.DataFrame:
    """Flujo de ejecución: carga el campeón, predice probabilidad y aplica el umbral óptimo."""
    modelo, umbral, _ = cargar_campeon(nombre_modelo, alias)
    out = df.copy()
    out["probabilidad"] = modelo.predict_proba(df)[:, 1]
    out["prediccion"] = (out["probabilidad"] >= umbral).astype(int)
    out["umbral_usado"] = umbral
    return out
