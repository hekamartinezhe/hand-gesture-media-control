"""Tests unitarios para la función matemática normalizar() de landmarks."""

import numpy as np
import pytest

from scripts.utils import COLUMNAS_CSV, COLUMNAS_FEATURES, normalizar


@pytest.fixture
def mano_sintetica_derecha() -> np.ndarray:
    """Genera una mano sintética plausible de 21 puntos con nudillo medio extendido."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(low=0.2, high=0.8, size=(21, 3)).astype(np.float32)
    # Muñeca en (0.5, 0.8, 0.0)
    pts[0] = [0.5, 0.8, 0.0]
    # Nudillo medio (landmark 9) en posición distinta para escala no nula
    pts[9] = [0.5, 0.4, 0.05]
    return pts


def test_esquema_columnas_completo() -> None:
    """Verifica que el esquema total sea de 67 columnas."""
    assert len(COLUMNAS_FEATURES) == 63
    assert len(COLUMNAS_CSV) == 67
    assert COLUMNAS_FEATURES[:3] == ["x0", "x1", "x2"]
    assert COLUMNAS_FEATURES[21:24] == ["y0", "y1", "y2"]
    assert COLUMNAS_FEATURES[42:45] == ["z0", "z1", "z2"]
    assert COLUMNAS_CSV[-4:] == ["label", "handedness", "session_id", "timestamp"]


def test_forma_y_tipo_salida(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que el vector resultante sea unidimensional de 63 floats (float32)."""
    res_2d = normalizar(mano_sintetica_derecha, handedness="Right")
    assert res_2d.shape == (63,)
    assert res_2d.dtype == np.float32

    # Prueba con entrada aplanada (63,)
    res_1d = normalizar(mano_sintetica_derecha.flatten(), handedness="Right")
    assert np.allclose(res_2d, res_1d)


def test_muneca_siempre_en_origen(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que el landmark 0 siempre tenga coordenadas (0, 0, 0) tras normalizar."""
    res = normalizar(mano_sintetica_derecha)
    # En vector ordenado: x0 = res[0], y0 = res[21], z0 = res[42]
    assert np.isclose(res[0], 0.0)
    assert np.isclose(res[21], 0.0)
    assert np.isclose(res[42], 0.0)


def test_nudillo_medio_norma_unitaria(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que la distancia de la muñeca al nudillo medio (punto 9) sea exactamente 1.0."""
    res = normalizar(mano_sintetica_derecha)
    x9 = res[9]
    y9 = res[21 + 9]
    z9 = res[42 + 9]
    norma_nudillo_9 = np.sqrt(x9**2 + y9**2 + z9**2)
    assert np.isclose(norma_nudillo_9, 1.0, atol=1e-5)


def test_invarianza_de_traslacion(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que desplazar la mano en el espacio 3D no altera el vector normalizado."""
    desplazamiento = np.array([12.5, -45.2, 8.7], dtype=np.float32)
    mano_desplazada = mano_sintetica_derecha + desplazamiento

    res_original = normalizar(mano_sintetica_derecha)
    res_desplazada = normalizar(mano_desplazada)

    assert np.allclose(res_original, res_desplazada, atol=1e-5)


def test_invarianza_de_escala(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que escalar la mano (acercarla o alejarla) produce la misma representación."""
    factor_escala = 3.75
    mano_escalada = (mano_sintetica_derecha - mano_sintetica_derecha[0]) * factor_escala + mano_sintetica_derecha[0]

    res_original = normalizar(mano_sintetica_derecha)
    res_escalada = normalizar(mano_escalada)

    assert np.allclose(res_original, res_escalada, atol=1e-5)


def test_invarianza_de_lateralidad_espejado(mano_sintetica_derecha: np.ndarray) -> None:
    """Verifica que una mano izquierda reflejada espacialmente genera el mismo vector que la derecha."""
    # Simular mano izquierda espejando el eje x respecto a la muñeca
    mano_izquierda = mano_sintetica_derecha.copy()
    offset_x = mano_izquierda[0, 0]
    mano_izquierda[:, 0] = -(mano_izquierda[:, 0] - offset_x) + offset_x

    res_derecha = normalizar(mano_sintetica_derecha, handedness="Right")
    res_izquierda = normalizar(mano_izquierda, handedness="Left")

    assert np.allclose(res_derecha, res_izquierda, atol=1e-5)


def test_tolerancia_division_por_cero() -> None:
    """Verifica que una mano con puntos colapsados en el origen no lance ZeroDivisionError."""
    puntos_colapsados = np.zeros((21, 3), dtype=np.float32)
    res = normalizar(puntos_colapsados)
    assert res.shape == (63,)
    assert np.all(res == 0.0)


def test_tamano_invalido_lanza_excepcion() -> None:
    """Verifica que pasar una cantidad incorrecta de puntos lanza ValueError."""
    with pytest.raises(ValueError, match="Se esperaban 63 valores"):
        normalizar(np.zeros((10, 3)))
