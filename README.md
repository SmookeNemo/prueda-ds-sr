# Evaluación técnica – Ciencia de Datos

Modelo supervisado (clasificación binaria con desbalance tratado por pesos) y modelo no supervisado.

## Cómo ejecutar
```bash
pip install -r requirements.txt
python -m src.train                                   # challenge + tuneo + registro del campeón
python -m src.predict data/nuevos.csv outputs/predicciones.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db     # ver runs y modelo registrado
```

## Estructura
```
src/
├── train.py              # flujo supervisado completo (editar solo CONFIG)
├── predict.py            # flujo de ejecución con models:/<nombre>@champion
└── utils/
    ├── eda.py            # resumen, target, tipos, gráficos, muestreo estratificado
    ├── outliers.py       # IQRCapper (transformer) y resumen de atípicos
    ├── feature_selection.py  # IV, correlación, VIF iterativo, importancia XGBoost
    ├── weights.py        # sample_weight balanceado
    ├── threshold.py      # búsqueda de umbral por iteración, Youden, con restricción
    ├── metrics.py        # matriz de confusión completa, AUC, Gini, KS, PR-AUC, gaps, gráficos
    ├── modeling.py       # split, pipelines, challenge, tabla comparativa, campeón, tuneo
    └── mlflow_utils.py   # log de runs, registro del campeón, carga y predicción
```

## Decisiones metodológicas
- Split estratificado 70/15/15 **antes** de cualquier transformación (sin leakage).
- Atípicos: capping IQR (1.5) con límites calculados en train, dentro del Pipeline.
- Selección: IV en train (IV < 0.02 fuera, IV > 0.5 revisar leakage).
  - Árboles: IV + importancia acumulada 95 %.
  - Logit: IV + correlación > 0.8 + VIF > 5 (la multicolinealidad solo afecta al logit).
- Balanceo: `sample_weight` balanceado, mismo vector para todos los modelos.
- Umbral: búsqueda por iteración en validation (maximiza F1); Youden como referencia.
- Métricas en train/val/test con umbral 0.5 y óptimo; gaps de AUC y F1 para sobreajuste.
- Campeón: mejor AUC en validation entre modelos con gap AUC train-val ≤ 0.05; en empate, el más simple.
- Solo el campeón se registra en MLflow (alias `champion`, umbral como tag).

## Resultados
_(completar con la tabla comparativa y el campeón)_
