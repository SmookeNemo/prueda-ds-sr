"""Flujo de ejecución con el modelo campeón.
Uso:  python -m src.predict data/nuevos.csv outputs/predicciones.csv"""
import os
import sys

import pandas as pd

from src.features import TARGET, preparar_features
from src.utils import configurar_mlflow, predecir_con_campeon

NOMBRE_MODELO = "modelo_clasificacion"
EXPERIMENTO = "prueba_supervisado"


def main(entrada: str, salida: str):
    configurar_mlflow(EXPERIMENTO)
    df = pd.read_csv(entrada).drop(columns=[TARGET], errors="ignore")
    pred = predecir_con_campeon(preparar_features(df), NOMBRE_MODELO)
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    pred.to_csv(salida, index=False)
    print(f"{len(pred)} predicciones guardadas en {salida} | tasa predicha: {pred['prediccion'].mean():.4f}")


if __name__ == "__main__":
    entrada = sys.argv[1] if len(sys.argv) > 1 else "data/nuevos.csv"
    salida = sys.argv[2] if len(sys.argv) > 2 else "outputs/predicciones.csv"
    main(entrada, salida)
