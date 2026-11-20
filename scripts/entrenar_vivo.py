#!/usr/bin/env python3
"""Entrena y evalúa el Random Forest que usa la demo en vivo.

Los datos provienen de tres capturas con la cámara real:
- `data/vivo/base_ambas_manos.csv`: 570 cuadros, seis clases y dos manos.
- `data/vivo/correccion_izquierda.csv`: 119 cuadros de pulgar y puño izquierdos.
- `data/vivo/sesion_independiente.csv`: 190 cuadros para validación externa.

El script reproduce las métricas de `models/metricas_rf_vivo.json`, entrena el
modelo final y lo serializa en `models/modelo_rf_vivo.joblib`. No modifica
`data/dataset.csv`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.geometria import NOMBRES_CARACTERISTICAS, caracteristicas_geometricas
from scripts.utils import (
    CLASES_GESTOS,
    cargar_dataset,
    construir_matriz_x,
    limpiar_dataset,
)

COLUMNAS = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
)
SEMILLA = 42
MODELO = ROOT / "models" / "modelo_rf_vivo.joblib"
METRICAS = ROOT / "models" / "metricas_rf_vivo.json"
DATASET_ACADEMICO = ROOT / "data" / "dataset.csv"


def crear_pipeline() -> Pipeline:
    """Pipeline geométrico + escalado + Random Forest calibrado para la demo."""
    return Pipeline(
        [
            ("geometria", FunctionTransformer(caracteristicas_geometricas)),
            ("escalador", StandardScaler()),
            (
                "clasificador",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=3,
                    random_state=SEMILLA,
                    n_jobs=1,
                ),
            ),
        ]
    )


def cargar_csv(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Devuelve landmarks, etiquetas y manos (None si el CSV no trae mano)."""
    df = pd.read_csv(ruta)
    manos = df["mano"].to_numpy() if "mano" in df.columns else None
    return df[COLUMNAS].to_numpy(np.float32), df["label"].to_numpy(), manos


def ruta_legible(ruta: Path) -> str:
    """Ruta relativa al repo cuando sea posible, para métricas portables."""
    try:
        return str(ruta.resolve().relative_to(ROOT))
    except ValueError:
        return str(ruta)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Entrena el Random Forest de la demo con capturas en vivo."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "data" / "vivo" / "base_ambas_manos.csv",
        help="CSV base con ambas manos y seis clases",
    )
    parser.add_argument(
        "--correccion",
        type=Path,
        default=ROOT / "data" / "vivo" / "correccion_izquierda.csv",
        help="CSV de corrección para pulgar y puño izquierdos",
    )
    parser.add_argument(
        "--sesion",
        type=Path,
        default=ROOT / "data" / "vivo" / "sesion_independiente.csv",
        help="CSV de la sesión independiente de validación",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=MODELO,
        help="Ruta de salida del modelo .joblib",
    )
    parser.add_argument(
        "--metricas",
        type=Path,
        default=METRICAS,
        help="Ruta de salida de las métricas .json",
    )
    args = parser.parse_args()

    X_base, y_base, mano_base = cargar_csv(args.base)
    X_foco, y_foco, mano_foco = cargar_csv(args.correccion)
    if mano_base is None or mano_foco is None:
        raise ValueError("Los CSV base y de corrección deben incluir la columna 'mano'.")

    X = np.vstack([X_base, X_foco])
    y = np.concatenate([y_base, y_foco])
    manos = np.concatenate([mano_base, mano_foco])

    resultado: dict = {
        "modelo": "RandomForestClassifier",
        "configuracion": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_leaf": 3,
            "n_jobs": 1,
            "random_state": SEMILLA,
        },
        "entrenamiento": {
            "base_570": ruta_legible(args.base),
            "correccion_izquierda": ruta_legible(args.correccion),
            "sesion_independiente": ruta_legible(args.sesion),
            "muestras_totales": int(len(y)),
            "muestras_base": int(len(y_base)),
            "muestras_correccion": int(len(y_foco)),
            "por_mano": {
                mano: int((manos == mano).sum())
                for mano in ("izquierda", "derecha")
            },
            "por_clase": {
                clase: int((y == clase).sum()) for clase in CLASES_GESTOS
            },
        },
    }

    print("=== VALIDACIÓN ENTRE MANOS (BASE 570) ===")
    resultado["validacion_entre_manos_base_570"] = {}
    for mano_test in ("izquierda", "derecha"):
        train = mano_base != mano_test
        test = mano_base == mano_test
        modelo = crear_pipeline()
        modelo.fit(X_base[train], y_base[train])
        pred = modelo.predict(X_base[test])
        mascara_accion = y_base[test] != "neutro"
        acc = float(accuracy_score(y_base[test], pred))
        acc_accion = float(
            accuracy_score(y_base[test][mascara_accion], pred[mascara_accion])
        )
        resultado["validacion_entre_manos_base_570"][mano_test] = {
            "accuracy": round(acc, 4),
            "accuracy_solo_acciones": round(acc_accion, 4),
            "matriz_confusion": confusion_matrix(
                y_base[test], pred, labels=CLASES_GESTOS
            ).tolist(),
        }
        print(f"  test {mano_test}: acc={acc:.4f} acciones={acc_accion:.4f}")

    print("\n=== VALIDACIÓN TEMPORAL 70/30 POR BLOQUE (689) ===")
    indices_train: list[int] = []
    indices_test: list[int] = []
    bloques = pd.DataFrame({"mano": manos, "label": y})
    for _, sub in bloques.groupby(["mano", "label"], sort=False):
        corte = int(len(sub) * 0.7)
        indices_train.extend(sub.index[:corte])
        indices_test.extend(sub.index[corte:])
    modelo_temporal = crear_pipeline()
    modelo_temporal.fit(X[indices_train], y[indices_train])
    pred_temporal = modelo_temporal.predict(X[indices_test])
    acc_temporal = float(accuracy_score(y[indices_test], pred_temporal))
    resultado["validacion_temporal_689"] = {
        "accuracy": round(acc_temporal, 4),
        "muestras_test": int(len(indices_test)),
        "matriz_confusion": confusion_matrix(
            y[indices_test], pred_temporal, labels=CLASES_GESTOS
        ).tolist(),
    }
    print(f"  acc={acc_temporal:.4f} (n={len(indices_test)})")

    print("\n=== HOLDOUT DEL 30% DE LA CORRECCIÓN ===")
    indices_train_foco: list[int] = []
    indices_test_foco: list[int] = []
    correccion = pd.DataFrame({"label": y_foco})
    for _, sub in correccion.groupby("label", sort=False):
        corte = int(len(sub) * 0.7)
        indices_train_foco.extend(sub.index[:corte])
        indices_test_foco.extend(sub.index[corte:])
    X_holdout = np.vstack([X_base, X_foco[indices_train_foco]])
    y_holdout = np.concatenate([y_base, y_foco[indices_train_foco]])
    modelo_holdout = crear_pipeline()
    modelo_holdout.fit(X_holdout, y_holdout)
    pred_holdout = modelo_holdout.predict(X_foco[indices_test_foco])
    acc_holdout = float(
        accuracy_score(y_foco[indices_test_foco], pred_holdout)
    )
    resultado["holdout_correccion_30"] = {
        "accuracy": round(acc_holdout, 4),
        "muestras_test": int(len(indices_test_foco)),
        "matriz_confusion": confusion_matrix(
            y_foco[indices_test_foco],
            pred_holdout,
            labels=["pulgar", "puno"],
        ).tolist(),
    }
    print(f"  acc={acc_holdout:.4f} (n={len(indices_test_foco)})")

    print("\n=== SESIÓN INDEPENDIENTE ===")
    X_sesion, y_sesion, _ = cargar_csv(args.sesion)
    modelo_sesion = crear_pipeline()
    modelo_sesion.fit(X, y)
    pred_sesion = modelo_sesion.predict(X_sesion)
    acc_sesion = float(accuracy_score(y_sesion, pred_sesion))
    resultado["sesion_190"] = {
        "accuracy": round(acc_sesion, 4),
        "muestras": int(len(y_sesion)),
        "matriz_confusion": confusion_matrix(
            y_sesion, pred_sesion, labels=CLASES_GESTOS[:5]
        ).tolist(),
    }
    print(f"  acc={acc_sesion:.4f} (n={len(y_sesion)})")

    print("\n=== DATASET ACADÉMICO (REFERENCIA DE DOMINIO) ===")
    dataset = limpiar_dataset(cargar_dataset(DATASET_ACADEMICO))
    X_academico = construir_matriz_x(dataset)
    y_academico = dataset["label"].to_numpy()
    modelo_academico = crear_pipeline()
    modelo_academico.fit(X, y)
    pred_academico = modelo_academico.predict(X_academico)
    acc_academico = float(accuracy_score(y_academico, pred_academico))
    resultado["dataset_academico"] = {
        "accuracy": round(acc_academico, 4),
        "muestras": int(len(y_academico)),
        "nota": (
            "El modelo de demo se entrena con la sesión en vivo; el dataset "
            "académico se conserva para el modelo base y el reporte."
        ),
    }
    print(f"  acc={acc_academico:.4f} (n={len(y_academico)})")

    print("\n=== MODELO FINAL Y LATENCIA ===")
    modelo_final = crear_pipeline()
    modelo_final.fit(X, y)
    for _ in range(5):
        modelo_final.predict(X[:1])
    latencias = []
    for _ in range(100):
        inicio = time.perf_counter()
        modelo_final.predict(X[:1])
        latencias.append((time.perf_counter() - inicio) * 1000.0)
    resultado["latencia_ms"] = {
        "mediana": round(statistics.median(latencias), 3),
        "media": round(statistics.mean(latencias), 3),
        "max": round(max(latencias), 3),
        "nota": "Predicción de una muestra en CPU, sin MediaPipe.",
    }
    print(
        f"  mediana={resultado['latencia_ms']['mediana']} ms "
        f"max={resultado['latencia_ms']['max']} ms"
    )

    importancias = modelo_final.named_steps["clasificador"].feature_importances_
    orden = np.argsort(importancias)[::-1][:10]
    resultado["importancia_features"] = [
        {
            "feature": NOMBRES_CARACTERISTICAS[i],
            "importancia": round(float(importancias[i]), 4),
        }
        for i in orden
    ]

    if args.metricas.exists():
        previas = json.loads(args.metricas.read_text(encoding="utf-8"))
        if "validacion_en_vivo" in previas:
            resultado["validacion_en_vivo"] = previas["validacion_en_vivo"]

    args.modelo.parent.mkdir(parents=True, exist_ok=True)
    args.metricas.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, args.modelo)
    args.metricas.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nModelo en {args.modelo}")
    print(f"Métricas en {args.metricas}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
