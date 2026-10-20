"""Utilidades para la evaluación técnica. Uso:  from src.utils import *"""
from .eda import (resumen_datos, distribucion_target, tipos_variables, detectar_columnas_problema,
                  graficar_numericas_vs_target, tasa_target_por_categoria, graficar_correlacion,
                  muestreo_estratificado)
from .outliers import IQRCapper, resumen_atipicos, eliminar_atipicos_iqr
from .feature_selection import (calcular_iv, tabla_iv, filtrar_por_iv, filtrar_correlacion, calcular_vif,
                                vif_iterativo, importancia_xgb, seleccionar_por_importancia)
from .weights import calcular_pesos, pesos_por_clase, scale_pos_weight
from .threshold import buscar_umbral, umbral_youden, umbral_con_restriccion, graficar_umbral
from .metrics import (metricas_completas, ks_statistic, evaluar_modelo, calcular_gaps,
                      diagnostico_sobreajuste, graficar_roc, graficar_matriz_confusion)
from .modeling import (split_train_val_test, construir_preprocesador, construir_pipeline, modelos_base,
                       entrenar_y_evaluar, tabla_comparativa, elegir_campeon, espacios_busqueda,
                       tunear_modelo)
from .mlflow_utils import (ALIAS, configurar_mlflow, loguear_run, registrar_campeon, cargar_campeon,
                           predecir_con_campeon, listar_runs, registrar_desde_run)
