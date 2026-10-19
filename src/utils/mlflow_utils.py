"""MLflow: tracking de todos los modelos del challenge y registro SOLO del campeón.

- Cada modelo evaluado se loguea como un run (params, métricas, umbral, gráficos). No se registra.
- El campeón se registra en el Model Registry con alias 'champion' y el umbral como tag.
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


def loguear_run(resultado: dict, figuras: dict = None, etiqueta: str = "challenge", params_extra: dict = None):
    """Loguea un modelo del challenge SIN registrarlo. resultado = salida de entrenar_y_evaluar()."""
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
        return run.info.run_id


def registrar_campeon(resultado: dict, X_ejemplo: pd.DataFrame, nombre_modelo: str,
                      features: list = None, figuras: dict = None, code_paths: list = None) -> int:
    """Loguea y REGISTRA el campeón, le asigna el alias 'champion' y guarda el umbral como tag.
    code_paths=['src'] empaqueta el código propio (IQRCapper) junto al modelo."""
    pipe = resultado["pipeline"]
    ejemplo = X_ejemplo.head(5)
    firma = infer_signature(ejemplo, pipe.predict_proba(ejemplo)[:, 1])

    with mlflow.start_run(run_name=f"campeon_{resultado['nombre']}") as run:
        mlflow.set_tag("etapa", "campeon")
        mlflow.set_tag("modelo", resultado["nombre"])
        mlflow.log_params(_params_simples(pipe))
        mlflow.log_param("umbral_optimo", resultado["umbral"])
        mlflow.log_metrics(_metricas_planas(resultado["metricas"]))
        mlflow.log_metrics({k: float(v) for k, v in resultado["gaps"].items()})
        for nombre, fig in (figuras or {}).items():
            mlflow.log_figure(fig, f"graficos/{nombre}.png")
            plt.close(fig)
        # cloudpickle: MLflow 3.x usa 'skops' por defecto y rechaza clases propias (IQRCapper)
        kwargs = dict(sk_model=pipe, signature=firma, input_example=ejemplo,
                      registered_model_name=nombre_modelo, code_paths=code_paths,
                      serialization_format="cloudpickle")
        try:
            info = mlflow.sklearn.log_model(name="model", **kwargs)          # MLflow 3.x
        except TypeError:
            info = mlflow.sklearn.log_model(artifact_path="model", **kwargs)  # MLflow 2.x

    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(nombre_modelo, ALIAS, version)
    client.set_model_version_tag(nombre_modelo, version, "umbral", str(resultado["umbral"]))
    client.set_model_version_tag(nombre_modelo, version, "algoritmo", resultado["nombre"])
    if features is not None:
        client.set_model_version_tag(nombre_modelo, version, "features", ",".join(features))
    print(f"Registrado '{nombre_modelo}' v{version} con alias '@{ALIAS}' | umbral={resultado['umbral']}")
    return int(version)


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
