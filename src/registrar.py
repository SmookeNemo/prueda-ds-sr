"""Registro manual de un modelo específico en el Model Registry.

Uso (desde la raíz del proyecto):
    python -m src.registrar                    # lista los runs con sus métricas
    python -m src.registrar --modelo logit     # registra el run más reciente de ese modelo
    python -m src.registrar --run-id <id>      # registra un run concreto
"""
import argparse

import mlflow

from src.utils import ALIAS, configurar_mlflow, listar_runs, registrar_desde_run

EXPERIMENTO = "prueba_supervisado"
NOMBRE_MODELO = "modelo_clasificacion"


def main():
    parser = argparse.ArgumentParser(description="Registrar un modelo específico en MLflow")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--run-id", help="run_id a registrar")
    grupo.add_argument("--modelo", help="nombre del modelo (tag 'modelo'), p. ej. logit o xgboost_tuned")
    parser.add_argument("--alias", default=ALIAS, help="alias a asignar (por defecto: champion)")
    args = parser.parse_args()

    configurar_mlflow(EXPERIMENTO)
    if not args.run_id and not args.modelo:
        print(listar_runs(EXPERIMENTO).to_string())
        return

    run_id = args.run_id
    if args.modelo:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENTO],
                                  filter_string=f"tags.modelo = '{args.modelo}' and tags.etapa != 'campeon'",
                                  order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            raise SystemExit(f"No hay runs con modelo='{args.modelo}'")
        run_id = runs.iloc[0]["run_id"]
    registrar_desde_run(run_id, NOMBRE_MODELO, alias=args.alias)


if __name__ == "__main__":
    main()
