"""Características geométricas de la mano y suavizado temporal.

Las características se calculan a partir de ratios de distancias y ángulos
entre articulaciones. Son invariantes a traslación, escala, lateralidad y
rotación, por lo que no dependen de la mano ni de la orientación del usuario.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Articulaciones de cada dedo: MCP, PIP, DIP y TIP.
DEDOS: Dict[str, Tuple[int, int, int, int]] = {
    "indice": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anular": (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}

NOMBRES_CARACTERISTICAS: list[str] = []
for _dim in ("2d", "3d"):
    for _dedo in DEDOS:
        NOMBRES_CARACTERISTICAS.extend(
            [
                f"rectitud_{_dedo}_{_dim}",
                f"angulo_pip_{_dedo}_{_dim}",
                f"angulo_dip_{_dedo}_{_dim}",
                f"alcance_{_dedo}_{_dim}",
                f"separacion_{_dedo}_{_dim}",
            ]
        )
    NOMBRES_CARACTERISTICAS.extend(
        [
            f"angulo_pulgar_mcp_{_dim}",
            f"angulo_pulgar_ip_{_dim}",
            f"pulgar_indice_{_dim}",
            f"pulgar_menique_{_dim}",
            f"pulgar_punta_indice_{_dim}",
            f"pulgar_muneca_{_dim}",
            f"alcance_indice_{_dim}",
            f"alcance_medio_{_dim}",
            f"alcance_anular_{_dim}",
            f"alcance_menique_{_dim}",
            f"ancho_palma_{_dim}",
            f"separacion_puntas_{_dim}",
        ]
    )


def _distancia(a: np.ndarray, b: np.ndarray, dims: int) -> float:
    """Distancia euclidiana usando solo las dimensiones indicadas."""
    return float(np.linalg.norm((a - b)[:dims]))


def _angulo(
    puntos: np.ndarray, a: int, b: int, c: int, dims: int
) -> float:
    """Ángulo en el vértice `b` formado por los puntos a-b-c, en grados."""
    v1 = (puntos[a] - puntos[b])[:dims]
    v2 = (puntos[c] - puntos[b])[:dims]
    denominador = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denominador < 1e-9:
        return 180.0
    coseno = float(np.dot(v1, v2) / denominador)
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def _rectitud(
    puntos: np.ndarray, a: int, b: int, c: int, d: int, dims: int
) -> float:
    """Ratio entre la distancia MCP-TIP y la longitud de los tres segmentos.

    Vale ~1.0 para un dedo extendido y ~0.5 para un dedo flexionado.
    """
    segmentos = (
        _distancia(puntos[a], puntos[b], dims)
        + _distancia(puntos[b], puntos[c], dims)
        + _distancia(puntos[c], puntos[d], dims)
    )
    return _distancia(puntos[a], puntos[d], dims) / max(segmentos, 1e-9)


def _caracteristicas_una(puntos: np.ndarray) -> np.ndarray:
    """Calcula las 64 características de una mano de forma (21, 3)."""
    caracteristicas: list[float] = []

    for dims in (2, 3):
        escala = max(_distancia(puntos[0], puntos[9], dims), 1e-9)

        for nombre, (a, b, c, d) in DEDOS.items():
            caracteristicas.extend(
                [
                    _rectitud(puntos, a, b, c, d, dims),
                    _angulo(puntos, a, b, d, dims) / 180.0,
                    _angulo(puntos, b, c, d, dims) / 180.0,
                    _distancia(puntos[0], puntos[d], dims) / escala,
                    _distancia(
                        puntos[d], puntos[12 if nombre == "indice" else 8], dims
                    )
                    / escala,
                ]
            )

        caracteristicas.extend(
            [
                _angulo(puntos, 2, 3, 4, dims) / 180.0,
                _angulo(puntos, 1, 2, 3, dims) / 180.0,
                _distancia(puntos[4], puntos[5], dims) / escala,
                _distancia(puntos[4], puntos[17], dims) / escala,
                _distancia(puntos[4], puntos[8], dims) / escala,
                _distancia(puntos[0], puntos[4], dims) / escala,
                _distancia(puntos[0], puntos[8], dims) / escala,
                _distancia(puntos[0], puntos[12], dims) / escala,
                _distancia(puntos[0], puntos[16], dims) / escala,
                _distancia(puntos[0], puntos[20], dims) / escala,
                _distancia(puntos[5], puntos[17], dims) / escala,
                (
                    _distancia(puntos[8], puntos[12], dims)
                    + _distancia(puntos[12], puntos[16], dims)
                    + _distancia(puntos[16], puntos[20], dims)
                )
                / escala,
            ]
        )

    return np.asarray(caracteristicas, dtype=np.float32)


def _vector_a_puntos(vector: np.ndarray) -> np.ndarray:
    """Convierte (x0..x20, y0..y20, z0..z20) en una matriz (21, 3)."""
    return np.stack(
        [vector[:21], vector[21:42], vector[42:63]], axis=1
    ).astype(np.float32)


def caracteristicas_geometricas(vector: np.ndarray) -> np.ndarray:
    """Transforma landmarks (63,) o (n, 63) en características geométricas.

    Args:
        vector: Coordenadas x0..x20, y0..y20, z0..z20.

    Returns:
        np.ndarray: Vector de 64 características o matriz (n, 64).

    Raises:
        ValueError: Si la entrada no tiene 63 coordenadas por muestra.
    """
    arreglo = np.asarray(vector, dtype=np.float32)

    if arreglo.ndim == 1:
        if arreglo.size != 63:
            raise ValueError(
                f"Se esperaban 63 coordenadas, pero se recibieron {arreglo.size}."
            )
        return _caracteristicas_una(_vector_a_puntos(arreglo))

    if arreglo.ndim == 2:
        if arreglo.shape[1] != 63:
            raise ValueError(
                f"Se esperaban 63 columnas, pero se recibieron {arreglo.shape[1]}."
            )
        return np.stack(
            [_caracteristicas_una(_vector_a_puntos(fila)) for fila in arreglo]
        ).astype(np.float32)

    raise ValueError("La entrada debe ser un vector (63,) o una matriz (n, 63).")


# Umbrales del verificador geométrico. Se calibraron con una sesión de captura
# real y se contrastaron contra el dataset; son deliberadamente conservadores.
UMBRAL_DEDO_EXTENDIDO = 0.85
UMBRAL_DEDO_FLEXIONADO = 0.80
UMBRAL_PULGAR_EXTENDIDO = 0.85
UMBRAL_ANGULO_PULGAR = 0.85
UMBRAL_DIFERENCIA_PAZ = 0.18
UMBRAL_RATIO_ANULAR_PAZ = 0.90


def clasificar_geometrico(puntos: np.ndarray) -> str:
    """Clasifica un gesto usando la forma de la mano, sin depender del modelo ML.

    La regla es invariante a escala, traslación, lateralidad y rotación porque
    usa ratios de rectitud y distancias normalizadas por el tamaño de la palma.
    Se usa como verificación del modelo: si el ML se desvía por cambio de
    dominio, la geometría recupera el gesto que realmente está formando la mano.

    Args:
        puntos: Landmarks de forma (21, 3) o vector de 63 coordenadas.

    Returns:
        str: Una de las clases ``palma``, ``senalar``, ``paz``, ``pulgar``,
        ``puno`` o ``neutro``.
    """
    arreglo = np.asarray(puntos, dtype=np.float32)
    if arreglo.ndim == 1:
        if arreglo.size != 63:
            raise ValueError(
                f"Se esperaban 63 coordenadas, pero se recibieron {arreglo.size}."
            )
        mano = _vector_a_puntos(arreglo)
    elif arreglo.shape == (21, 3):
        mano = arreglo
    else:
        raise ValueError("Se esperaban landmarks de forma (21, 3) o (63,).")

    rectitudes = {
        nombre: _rectitud(mano, *dedo, 2) for nombre, dedo in DEDOS.items()
    }
    indice = rectitudes["indice"]
    medio = rectitudes["medio"]
    anular = rectitudes["anular"]
    menique = rectitudes["menique"]
    dedos = [indice, medio, anular, menique]

    escala = max(_distancia(mano[0], mano[9], 2), 1e-9)
    pulgar = _distancia(mano[4], mano[5], 2) / escala
    angulo_pulgar = _angulo(mano, 2, 3, 4, 2) / 180.0
    alcance_anular = _distancia(mano[0], mano[16], 2) / escala
    alcance_medio = _distancia(mano[0], mano[12], 2) / escala
    ratio_anular = alcance_anular / max(alcance_medio, 1e-9)

    if indice > UMBRAL_DEDO_EXTENDIDO and medio < UMBRAL_DEDO_FLEXIONADO:
        return "senalar"

    if (
        indice > UMBRAL_DEDO_EXTENDIDO
        and medio > UMBRAL_DEDO_EXTENDIDO
        and (
            medio - anular > UMBRAL_DIFERENCIA_PAZ
            or anular < UMBRAL_DEDO_FLEXIONADO
            or ratio_anular < UMBRAL_RATIO_ANULAR_PAZ
        )
    ):
        return "paz"

    if min(dedos) > UMBRAL_DEDO_EXTENDIDO:
        return "palma" if pulgar < UMBRAL_PULGAR_EXTENDIDO else "neutro"

    if (
        max(dedos) < UMBRAL_DEDO_EXTENDIDO
        and pulgar > UMBRAL_PULGAR_EXTENDIDO
        and angulo_pulgar > UMBRAL_ANGULO_PULGAR
    ):
        return "pulgar"

    if max(dedos) < UMBRAL_DEDO_EXTENDIDO:
        return "puno"

    return "neutro"


class SuavizadorLandmarks:
    """Filtro exponencial simple para reducir el temblor de MediaPipe.

    Mantiene un estado por coordenada y lo actualiza con:
        estado = alfa * nuevo + (1 - alfa) * estado

    Un `alfa` menor suaviza más pero introduce más retardo. Para gestos
    estáticos, valores entre 0.3 y 0.5 eliminan el temblor visible sin
    impedir el cambio de gesto.
    """

    def __init__(self, alfa: float = 0.4) -> None:
        if not 0.0 < alfa <= 1.0:
            raise ValueError("El factor de suavizado debe estar en (0, 1].")
        self.alfa = float(alfa)
        self._estado: np.ndarray | None = None

    def suavizar(self, puntos: np.ndarray) -> np.ndarray:
        """Devuelve los landmarks suavizados de forma (21, 3)."""
        actual = np.asarray(puntos, dtype=np.float32)
        if actual.shape != (21, 3):
            raise ValueError(
                f"Se esperaban landmarks de forma (21, 3), no {actual.shape}."
            )

        if self._estado is None:
            self._estado = actual.copy()
        else:
            self._estado = self.alfa * actual + (1.0 - self.alfa) * self._estado

        return self._estado.copy()

    def reiniciar(self) -> None:
        """Olvida el estado anterior; se usa cuando se pierde la mano."""
        self._estado = None
