"""Tests de las características geométricas y del suavizado de landmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.geometria import (
    NOMBRES_CARACTERISTICAS,
    SuavizadorLandmarks,
    caracteristicas_geometricas,
    clasificar_geometrico,
)
from scripts.utils import CLASES_GESTOS, cargar_dataset, limpiar_dataset


def _vector_desde_puntos(puntos: np.ndarray) -> np.ndarray:
    """Convierte (21, 3) al orden del dataset: x0..x20, y0..y20, z0..z20."""
    return np.concatenate(
        [puntos[:, 0], puntos[:, 1], puntos[:, 2]]
    ).astype(np.float32)


def _mano_sintetica() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=(21, 3)).astype(np.float32)


def test_caracteristicas_tienen_longitud_esperada() -> None:
    """La transformación produce 64 valores finitos por muestra."""
    puntos = _mano_sintetica()
    vector = caracteristicas_geometricas(_vector_desde_puntos(puntos))

    assert vector.shape == (len(NOMBRES_CARACTERISTICAS),)
    assert vector.shape == (64,)
    assert np.isfinite(vector).all()


def test_caracteristicas_invariancia_espejado() -> None:
    """Espejar la mano no debe cambiar las características geométricas."""
    puntos = _mano_sintetica()
    espejado = puntos.copy()
    espejado[:, 0] *= -1.0

    original = caracteristicas_geometricas(_vector_desde_puntos(puntos))
    reflejado = caracteristicas_geometricas(_vector_desde_puntos(espejado))

    assert np.allclose(original, reflejado, atol=1e-5)


def test_caracteristicas_invariancia_rotacion() -> None:
    """Rotar la mano alrededor de z no debe cambiar las características."""
    puntos = _mano_sintetica()
    angulo = np.deg2rad(30.0)
    rotacion = np.array(
        [
            [np.cos(angulo), -np.sin(angulo), 0.0],
            [np.sin(angulo), np.cos(angulo), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotada = puntos @ rotacion.T

    original = caracteristicas_geometricas(_vector_desde_puntos(puntos))
    girada = caracteristicas_geometricas(_vector_desde_puntos(rotada))

    assert np.allclose(original, girada, atol=1e-5)


def test_caracteristicas_rechazan_longitud_invalida() -> None:
    """Una entrada con menos de 63 coordenadas debe fallar de forma clara."""
    with pytest.raises(ValueError):
        caracteristicas_geometricas(np.zeros(62, dtype=np.float32))


def test_suavizador_promedia_y_reinicia() -> None:
    """El EMA arranca con el primer cuadro y vuelve a arrancar tras reiniciar."""
    suavizador = SuavizadorLandmarks(alfa=0.5)
    primero = np.zeros((21, 3), dtype=np.float32)
    segundo = np.ones((21, 3), dtype=np.float32)

    assert np.allclose(suavizador.suavizar(primero), primero)
    assert np.allclose(suavizador.suavizar(segundo), 0.5 * segundo)

    suavizador.reiniciar()
    assert np.allclose(suavizador.suavizar(segundo), segundo)


def test_suavizador_rechaza_alfa_invalido() -> None:
    """El factor de suavizado debe estar en (0, 1]."""
    with pytest.raises(ValueError):
        SuavizadorLandmarks(alfa=0.0)
    with pytest.raises(ValueError):
        SuavizadorLandmarks(alfa=1.5)


def test_clasificador_geometrico_reconoce_una_muestra_por_clase() -> None:
    """La regla geométrica asigna la etiqueta correcta a poses claras."""
    ruta = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
    df = limpiar_dataset(cargar_dataset(ruta))
    columnas = (
        [f"x{i}" for i in range(21)]
        + [f"y{i}" for i in range(21)]
        + [f"z{i}" for i in range(21)]
    )

    for clase in CLASES_GESTOS:
        fila = df[df["label"] == clase].iloc[0][columnas].to_numpy(np.float32)
        puntos = fila.reshape(3, 21).T
        assert clasificar_geometrico(puntos) == clase


def test_clasificador_geometrico_acepta_anular_semi_extendido() -> None:
    """`paz` no debe exigir que el anular esté completamente doblado.

    Reproduce el caso de una mano izquierda donde el anular queda casi
    recto pero su punta está más cerca de la muñeca que la del medio.
    Antes del criterio `ratio_anular`, esta pose caía en `neutro`.
    """
    puntos = np.zeros((21, 3), dtype=np.float32)
    puntos[0] = (0.0, 0.0, 0.0)

    # Pulgar recogido.
    puntos[1] = (-0.15, -0.05, 0.0)
    puntos[2] = (-0.25, 0.00, 0.0)
    puntos[3] = (-0.30, 0.10, 0.0)
    puntos[4] = (0.00, 0.20, 0.0)

    # Índice y medio extendidos.
    puntos[5] = (0.25, 0.90, 0.0)
    puntos[6] = (0.28, 1.20, 0.0)
    puntos[7] = (0.30, 1.45, 0.0)
    puntos[8] = (0.32, 1.70, 0.0)
    puntos[9] = (0.00, 1.00, 0.0)
    puntos[10] = (0.00, 1.40, 0.0)
    puntos[11] = (0.00, 1.70, 0.0)
    puntos[12] = (0.00, 2.00, 0.0)

    # Anular casi recto, pero con la punta al 83% del alcance del medio.
    puntos[13] = (-0.25, 0.95, 0.0)
    puntos[14] = (-0.25, 1.25, 0.0)
    puntos[15] = (-0.25, 1.45, 0.0)
    puntos[16] = (-0.25, 1.65, 0.0)

    # Meñique doblado hacia la palma.
    puntos[17] = (-0.45, 0.85, 0.0)
    puntos[18] = (-0.45, 0.90, 0.0)
    puntos[19] = (-0.42, 0.85, 0.0)
    puntos[20] = (-0.40, 0.80, 0.0)

    assert clasificar_geometrico(puntos) == "paz"
