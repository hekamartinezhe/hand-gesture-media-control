#!/usr/bin/env python3
"""Entrena un SVM RBF sobre características geométricas de la mano.

El modelo resultante conserva la interfaz del demo: recibe un vector de 63
landmarks normalizados y devuelve la clase predicha. La transformación
geométrica se encapsula en el Pipeline para que `demo.py` no cambie.

Uso:
    python scripts/entrenar_geometrico.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.geometria import NOMBRES_CARACTERISTICAS, caracteristicas_geometricas
from scripts.utils import (
    CLASES_GESTOS,
    SEMILLA,
    cargar_dataset,
    construir_matriz_x,
    limpiar_dataset,
)

RUTA_DATASET = ROOT_DIR / "data" / "dataset.csv"
RUTA_MODELO = ROOT_DIR / "models" / "modelo_geom.joblib"
RUTA_METRICAS = ROOT_DIR / "models" / "metricas_geom.json"


def crear_modelo() -> Pipeline:
    """Crea el Pipeline de geometría + escalado + SVM RBF."""
    return Pipeline(
        [
            ("geometria", FunctionTransformer(caracteristicas_geometricas)),
            ("escalador", StandardScaler()),
            (
                "clasificador",
                SVC(kernel="rbf", C=10.0, gamma="scale", random_state=SEMILLA),
            ),
        ]
    )


def rotar(X: np.ndarray, grados_z: float, grados_y: float) -> np.ndarray:
    """Rota cada muestra alrededor de los ejes z e y (en grados)."""
    puntos = np.stack([X[:, :21], X[:, 21:42], X[:, 42:63]], axis=2)
    cz, sz = np.cos(np.deg2rad(grados_z)), np.sin(np.deg2rad(grados_z))
    cy, sy = np.cos(np.deg2rad(grados_y)), np.sin(np.deg2rad(grados_y))
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rotados = np.einsum("nij,jk->nik", puntos, (rz @ ry).T)
    return np.concatenate(
        [rotados[:, :, 0], rotados[:, :, 1], rotados[:, :, 2]], axis=1
    ).astype(np.float32)


def espejar(X: np.ndarray) -> np.ndarray:
    """Espeja el eje x de cada muestra (simula la mano contraria)."""
    copia = X.copy()
    copia[:, :21] *= -1.0
    return copia


def evaluar(modelo: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Calcula accuracy, F1 macro y reporte por clase."""
    prediccion = modelo.predict(X)
    return {
        "accuracy": float(accuracy_score(y, prediccion)),
        "f1_macro": float(f1_score(y, prediccion, average="macro")),
        "reporte": classification_report(
            y, prediccion, labels=CLASES_GESTOS, output_dict=True, zero_division=0
        ),
    }


def main() -> int:
    df = limpiar_dataset(cargar_dataset(RUTA_DATASET))
    X = construir_matriz_x(df)
    y = df["label"].to_numpy()

    # Split temporal por clase: entrena con el 70% inicial de cada toma y
    # evalúa con el 30% final. Evita que cuadros casi idénticos queden en
    # ambos lados y da una estimación más honesta que el split aleatorio.
    indices_train: list[int] = []
    indices_test: list[int] = []
    for clase in CLASES_GESTOS:
        indices = np.where(y == clase)[0]
        corte = int(len(indices) * 0.7)
        indices_train.extend(indices[:corte])
        indices_test.extend(indices[corte:])

    X_train, y_train = X[indices_train], y[indices_train]
    X_test, y_test = X[indices_test], y[indices_test]

    modelo = crear_modelo()
    modelo.fit(X_train, y_train)
    metricas_test = evaluar(modelo, X_test, y_test)

    rng = np.random.default_rng(SEMILLA)
    variantes = {
        "original": X_test,
        "espejado": espejar(X_test),
        "rotado_z30": rotar(X_test, 30.0, 0.0),
        "rotado_y20": rotar(X_test, 0.0, 20.0),
        "jitter": X_test + rng.normal(0.0, 0.006, X_test.shape).astype(np.float32),
    }
    robustez = {
        nombre: float(accuracy_score(y_test, modelo.predict(datos)))
        for nombre, datos in variantes.items()
    }

    print(f"Características: {len(NOMBRES_CARACTERISTICAS)}")
    print(f"Split temporal: train={len(X_train)} test={len(X_test)}")
    print(
        f"Test: accuracy={metricas_test['accuracy']:.4f} "
        f"f1_macro={metricas_test['f1_macro']:.4f}"
    )
    for nombre, exactitud in robustez.items():
        print(f"  {nombre:12s} accuracy={exactitud:.4f}")

    # Reentrena con todas las muestras para el artefacto desplegable.
    modelo_final = crear_modelo()
    modelo_final.fit(X, y)
    joblib.dump(modelo_final, RUTA_MODELO)

    metricas = {
        "modelo": "svm_rbf_geometrico",
        "fecha": "2026-11-16",
        "features": len(NOMBRES_CARACTERISTICAS),
        "descripcion_features": (
            "Ratios de rectitud, ángulos de articulaciones y distancias "
            "normalizadas por el tamaño de la palma; calculadas en 2D y 3D."
        ),
        "entrenamiento_final": int(len(X)),
        "split_temporal": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            **metricas_test,
        },
        "robustez": robustez,
        "suavizado_demo": "EMA con alfa=0.4 sobre los 21 landmarks",
        "notas": (
            "Las características son invariantes a traslación, escala, "
            "lateralidad y rotación. El split temporal evita la fuga por "
            "cuadros consecutivos casi idénticos."
        ),
    }
    RUTA_METRICAS.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nModelo guardado en {RUTA_MODELO}")
    print(f"Métricas guardadas en {RUTA_METRICAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
