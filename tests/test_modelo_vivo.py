"""Tests del modelo Random Forest adaptado a la cámara en vivo."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.filtro import FiltroDisparo, PestilloGesto
from scripts.utils import (
    CLASES_GESTOS,
    cargar_dataset,
    construir_matriz_x,
    limpiar_dataset,
    normalizar,
)

ROOT = Path(__file__).resolve().parent.parent
MODELO = ROOT / "models" / "modelo_rf_vivo.joblib"
DATASET = ROOT / "data" / "dataset.csv"
SESION_INDEPENDIENTE = ROOT / "data" / "vivo" / "sesion_independiente.csv"


@pytest.fixture(scope="module")
def modelo():
    """Carga el pipeline de la demo una sola vez por módulo."""
    if not MODELO.exists():
        pytest.skip(f"No existe {MODELO}; entrena el modelo antes de probar.")
    return joblib.load(MODELO)


@pytest.fixture(scope="module")
def muestra_real() -> np.ndarray:
    """Toma landmarks reales del dataset académico como entrada válida."""
    dataset = limpiar_dataset(cargar_dataset(DATASET))
    return construir_matriz_x(dataset)[:5]


def test_modelo_carga_y_predice_clases_validas(modelo, muestra_real) -> None:
    """La predicción siempre pertenece a las seis clases del proyecto."""
    predicciones = modelo.predict(muestra_real)

    assert len(predicciones) == len(muestra_real)
    assert set(predicciones).issubset(set(CLASES_GESTOS))


def test_modelo_acepta_una_sola_muestra(modelo, muestra_real) -> None:
    """El bucle del demo predice muestra por muestra, no por lotes."""
    prediccion = modelo.predict(muestra_real[:1])

    assert prediccion.shape == (1,)
    assert str(prediccion[0]) in CLASES_GESTOS


def test_normalizar_no_cambia_la_prediccion(modelo, muestra_real) -> None:
    """La demo normaliza antes de predecir; el modelo no debe notar el cambio."""
    vector = muestra_real[0]
    puntos = np.stack(
        [vector[:21], vector[21:42], vector[42:63]], axis=1
    ).astype(np.float32)
    normalizado = normalizar(puntos, "Right").reshape(1, -1)

    sin_normalizar = modelo.predict(vector.reshape(1, -1))[0]
    con_normalizar = modelo.predict(normalizado)[0]

    assert sin_normalizar == con_normalizar


def test_latencia_por_cuadro_es_util_para_tiempo_real(modelo, muestra_real) -> None:
    """La inferencia de una muestra debe quedar muy por debajo del cuadro."""
    for _ in range(5):
        modelo.predict(muestra_real[:1])

    latencias = []
    for _ in range(50):
        inicio = time.perf_counter()
        modelo.predict(muestra_real[:1])
        latencias.append((time.perf_counter() - inicio) * 1000.0)

    mediana = statistics.median(latencias)
    assert mediana < 20.0, f"Latencia mediana demasiado alta: {mediana:.2f} ms"


def test_replay_sesion_independiente_dispara_una_accion_por_bloque(modelo) -> None:
    """El flujo final (1 cuadro + pestillo) dispara una vez por presentación."""
    if not SESION_INDEPENDIENTE.exists():
        pytest.skip(f"No existe {SESION_INDEPENDIENTE}")

    sesion = pd.read_csv(SESION_INDEPENDIENTE)
    filtro = FiltroDisparo(frames_confirmacion=1, cooldown_s=0.5)
    pestillo = PestilloGesto()
    acciones: list[str] = []
    aciertos = 0
    ahora = 0.0

    for _, fila in sesion.iterrows():
        puntos = np.array(
            [
                fila[[f"x{i}" for i in range(21)]],
                fila[[f"y{i}" for i in range(21)]],
                fila[[f"z{i}" for i in range(21)]],
            ],
            dtype=np.float32,
        ).T
        vector = normalizar(puntos, str(fila["handedness"]))
        predicho = str(modelo.predict(vector.reshape(1, -1))[0])
        if predicho == fila["label"]:
            aciertos += 1

        ahora += 1 / 15
        if pestillo.actualizar(predicho):
            disparo = filtro.actualizar(predicho, ahora)
        else:
            filtro.reiniciar()
            disparo = None
        if disparo is not None:
            pestillo.bloquear(disparo)
            acciones.append(disparo)

    assert len(sesion) == 190
    assert aciertos == 190
    assert acciones == ["palma", "senalar", "paz", "pulgar", "puno"]
