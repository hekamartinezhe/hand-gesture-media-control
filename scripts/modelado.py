"""Entrenamiento y evaluación de clasificadores clásicos para gestos.

La partición de datos y las métricas viven fuera del notebook para que el
experimento sea reproducible por línea de comandos o desde otro script.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.utils import SEMILLA

VECINOS_KNN: tuple[int, ...] = (3, 5, 7, 9, 11)


def _con_escalado(estimador: Any) -> Pipeline:
    """Envuelve un clasificador con estandarización, necesaria para SVM/KNN/MLP."""
    return Pipeline(
        [
            ("escalador", StandardScaler()),
            ("clasificador", estimador),
        ]
    )


def construir_candidatos() -> dict[str, Any]:
    """Devuelve los clasificadores base; KNN se ajusta aparte sobre validación."""
    return {
        "svm_rbf": _con_escalado(
            SVC(kernel="rbf", C=10.0, gamma="scale", random_state=SEMILLA)
        ),
        "svm_lineal": _con_escalado(
            SVC(kernel="linear", C=1.0, random_state=SEMILLA)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=SEMILLA
        ),
        "mlp": _con_escalado(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=SEMILLA,
            )
        ),
    }


def _latencia_ms(modelo: Any, muestra: np.ndarray, repeticiones: int = 100) -> float:
    """Mide el costo medio de predecir una sola muestra (sin contar el warm-up)."""
    modelo.predict(muestra)
    inicio = time.perf_counter()
    for _ in range(repeticiones):
        modelo.predict(muestra)
    return (time.perf_counter() - inicio) * 1000.0 / repeticiones


def evaluar(modelo: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Calcula accuracy, F1 macro y latencia de una muestra."""
    predicciones = modelo.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predicciones)),
        "f1_macro": float(f1_score(y, predicciones, average="macro")),
        "latencia_ms": float(_latencia_ms(modelo, X[:1])),
    }


def entrenar_y_comparar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Entrena los candidatos y elige KNN usando solo el conjunto de validación.

    Returns:
        (resultados, modelos): métricas por modelo y modelos ya entrenados.
    """
    resultados: dict[str, dict[str, Any]] = {}
    modelos: dict[str, Any] = {}

    for nombre, modelo in construir_candidatos().items():
        inicio = time.perf_counter()
        modelo.fit(X_train, y_train)
        duracion = time.perf_counter() - inicio
        resultados[nombre] = {
            **evaluar(modelo, X_val, y_val),
            "tiempo_entrenamiento_s": float(duracion),
        }
        modelos[nombre] = modelo

    # KNN: se prueban varios vecinos y se conserva el mejor F1 macro en validación.
    inicio = time.perf_counter()
    mejor_k = VECINOS_KNN[0]
    mejor_f1 = -1.0
    mejor_modelo: Pipeline | None = None
    for k in VECINOS_KNN:
        candidato = _con_escalado(KNeighborsClassifier(n_neighbors=k))
        candidato.fit(X_train, y_train)
        f1 = float(f1_score(y_val, candidato.predict(X_val), average="macro"))
        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_k = k
            mejor_modelo = candidato

    assert mejor_modelo is not None
    resultados["knn"] = {
        **evaluar(mejor_modelo, X_val, y_val),
        "tiempo_entrenamiento_s": float(time.perf_counter() - inicio),
        "n_neighbors": mejor_k,
        "vecinos_evaluados": list(VECINOS_KNN),
    }
    modelos["knn"] = mejor_modelo
    return resultados, modelos


def seleccionar_mejor(resultados: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Ordena por F1 macro, luego accuracy y finalmente menor latencia."""
    nombre = max(
        resultados,
        key=lambda n: (
            resultados[n]["f1_macro"],
            resultados[n]["accuracy"],
            -resultados[n]["latencia_ms"],
        ),
    )
    return {"nombre": nombre, **resultados[nombre]}


def entrenar_knn_final(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_neighbors: int,
) -> Pipeline:
    """Reentrena KNN con train+val una vez fijado el hiperparámetro."""
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    modelo = _con_escalado(KNeighborsClassifier(n_neighbors=n_neighbors))
    modelo.fit(X, y)
    return modelo
