"""Utilidades matemáticas, de normalización y control multimedia multiplataforma.

Centraliza las funciones puras para que sean reutilizables tanto en Linux
(vía playerctl / MPRIS) como en Windows (vía Win32 virtual media keys).
"""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Esquema de características y metadatos del CSV
COLUMNAS_FEATURES: List[str] = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
)
COLUMNAS_METADATA: List[str] = [
    "label",
    "handedness",
    "session_id",
    "timestamp",
]
COLUMNAS_CSV: List[str] = COLUMNAS_FEATURES + COLUMNAS_METADATA

# Mapeo de acciones admitidas a comandos de playerctl (Linux)
COMANDOS_LINUX: Dict[str, List[str]] = {
    "play_pause": ["playerctl", "play-pause"],
    "next": ["playerctl", "next"],
    "previous": ["playerctl", "previous"],
    "vol_up": ["playerctl", "volume", "0.1+"],
    "vol_down": ["playerctl", "volume", "0.1-"],
}

# Mapeo de acciones admitidas a códigos Virtual Key de Windows (Win32 API)
# VK_MEDIA_PLAY_PAUSE = 0xB3, VK_MEDIA_NEXT_TRACK = 0xB0,
# VK_MEDIA_PREV_TRACK = 0xB1, VK_VOLUME_UP = 0xAF, VK_VOLUME_DOWN = 0xAE
CODIGOS_VK_WINDOWS: Dict[str, int] = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "vol_up": 0xAF,
    "vol_down": 0xAE,
}

# Clases de gestos mapeadas a acciones multimedia
GESTO_A_ACCION: Dict[str, Optional[str]] = {
    "palma": "play_pause",
    "senalar": "next",
    "paz": "previous",
    "pulgar": "vol_up",
    "puno": "vol_down",
    "neutro": None,
}

# Clases válidas del dataset y semilla única de reproducibilidad
CLASES_GESTOS: List[str] = ["palma", "senalar", "paz", "pulgar", "puno", "neutro"]
SEMILLA: int = 42

# Una coordenada normalizada (muñeca en el origen, nudillo medio a distancia 1)
# rara vez supera 3; valores mayores delatan landmarks corruptos.
MAX_COORDENADA_NORMALIZADA: float = 10.0


def ejecutar_comando_multimedia(
    accion: str, sistema: Optional[str] = None
) -> bool:
    """Despacha la acción multimedia según el sistema operativo detectado.

    Args:
        accion: Nombre de la acción ('play_pause', 'next', 'previous', 'vol_up', 'vol_down').
        sistema: Sobrescritura opcional del sistema operativo para testing ('Linux', 'Windows').

    Returns:
        bool: True si el comando fue emitido exitosamente, False en caso contrario.
    """
    if accion not in COMANDOS_LINUX:
        return False

    os_actual = sistema or platform.system()

    if os_actual == "Linux":
        comando = COMANDOS_LINUX[accion]
        try:
            resultado = subprocess.run(
                comando,
                check=False,
                capture_output=True,
                text=True,
            )
            return resultado.returncode == 0
        except FileNotFoundError:
            # playerctl no está instalado en el sistema
            return False

    elif os_actual == "Windows":
        vk = CODIGOS_VK_WINDOWS.get(accion)
        if vk is None:
            return False

        try:
            import ctypes

            # KEYEVENTF_KEYUP = 2
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
            ctypes.windll.user32.keybd_event(vk, 0, 2, 0)
            return True
        except (AttributeError, OSError):
            return False

    return False


def despachar_gesto(gesto: str) -> bool:
    """Traduce un gesto detectado a su acción y ejecuta el comando multimedia.

    Args:
        gesto: Nombre de la clase del gesto ('palma', 'senalar', etc.).

    Returns:
        bool: True si se ejecutó una acción, False si era 'neutro' o inválido.
    """
    accion = GESTO_A_ACCION.get(gesto)
    if not accion:
        return False
    return ejecutar_comando_multimedia(accion)


def normalizar(landmarks: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Normaliza un conjunto de 21 landmarks 3D para lograr invarianza espacial.

    Aplica la normalización geométrica:
    1. Traslación al origen restando las coordenadas de la muñeca (landmark 0).
    2. Escalamiento métrico dividiendo entre la distancia euclidiana muñeca-nudillo medio (landmark 9).
    3. Espejado horizontal en el eje x si la mano detectada es izquierda ('Left' o 'Izquierda').
    4. Vectorización a un array 1D de 63 elementos ordenados como (x0..x20, y0..y20, z0..z20).

    Args:
        landmarks: Array de coordenadas de forma (21, 3) o (63,).
        handedness: Lateralidad de la mano ('Right', 'Left', 'Derecha', 'Izquierda').

    Returns:
        np.ndarray: Vector unidimensional de 63 floats (float32).

    Raises:
        ValueError: Si la cantidad total de coordenadas no es exactamente 63.
    """
    arr = np.asarray(landmarks, dtype=np.float32)

    if arr.size != 63:
        raise ValueError(
            f"Se esperaban 63 valores de coordenadas (21 landmarks x 3 ejes), pero se recibieron {arr.size}."
        )

    pts = arr.reshape((21, 3)).copy()

    # 1. Traslación al origen restando la muñeca (landmark 0)
    muneca = pts[0].copy()
    pts -= muneca

    # 2. Escalamiento métrico: distancia muñeca (0,0,0) -> nudillo medio (landmark 9)
    escala = float(np.linalg.norm(pts[9]))
    epsilon = 1e-6
    if escala > epsilon:
        pts /= escala

    # 3. Invarianza de lateralidad: espejar eje x si es mano izquierda
    if str(handedness).strip().lower() in ("left", "izquierda"):
        pts[:, 0] = -pts[:, 0]

    # 4. Vectorización ordenada: (x0..x20, y0..y20, z0..z20)
    vector = np.concatenate([pts[:, 0], pts[:, 1], pts[:, 2]]).astype(np.float32)
    return vector


def _validar_columnas(df: pd.DataFrame) -> None:
    """Lanza ValueError si al DataFrame le falta alguna columna del esquema."""
    faltantes = [col for col in COLUMNAS_CSV if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias del esquema: {faltantes}")


def cargar_dataset(ruta: Path) -> pd.DataFrame:
    """Carga el CSV crudo y verifica que respete el esquema de 67 columnas.

    Args:
        ruta: Ruta al archivo CSV del dataset.

    Returns:
        pd.DataFrame: Dataset sin transformaciones.

    Raises:
        FileNotFoundError: Si la ruta no existe.
        ValueError: Si faltan columnas del esquema esperado.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el dataset: {ruta}")

    df = pd.read_csv(ruta)
    _validar_columnas(df)
    return df


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra filas nulas, landmarks anómalos, etiquetas inválidas y duplicados.

    Los datos crudos en `data/` nunca se modifican: la función retorna una copia.

    Args:
        df: Dataset con el esquema completo de 67 columnas.

    Returns:
        pd.DataFrame: Copia limpia, con índice reiniciado.

    Raises:
        ValueError: Si faltan columnas del esquema.
    """
    _validar_columnas(df)

    limpio = df.copy()
    for col in COLUMNAS_FEATURES:
        limpio[col] = pd.to_numeric(limpio[col], errors="coerce")

    coordenadas = limpio[COLUMNAS_FEATURES].to_numpy(dtype=np.float64)
    finitas = np.isfinite(coordenadas).all(axis=1)
    no_degeneradas = np.abs(coordenadas).sum(axis=1) > 0.0
    dentro_de_rango = np.abs(coordenadas).max(axis=1) <= MAX_COORDENADA_NORMALIZADA
    etiquetas_validas = limpio["label"].isin(CLASES_GESTOS).to_numpy()
    manos_validas = limpio["handedness"].isin(["Left", "Right"]).to_numpy()

    mascara = (
        finitas
        & no_degeneradas
        & dentro_de_rango
        & etiquetas_validas
        & manos_validas
    )
    limpio = limpio.loc[mascara].drop_duplicates(
        subset=COLUMNAS_FEATURES, keep="first"
    )
    return limpio.reset_index(drop=True)


def construir_matriz_x(df: pd.DataFrame) -> np.ndarray:
    """Construye la matriz de características (n, 63) con lateralidad canónica.

    Reutiliza `normalizar()` para espejar el eje x de las manos izquierdas:
    así ambas manos comparten el mismo espacio de representación.

    Args:
        df: Dataset con el esquema completo de 67 columnas.

    Returns:
        np.ndarray: Matriz float32 de forma (n, 63).

    Raises:
        ValueError: Si faltan columnas del esquema.
    """
    _validar_columnas(df)

    if df.empty:
        return np.empty((0, len(COLUMNAS_FEATURES)), dtype=np.float32)

    filas = df[COLUMNAS_FEATURES].to_numpy(dtype=np.float32)
    manos = df["handedness"].astype(str).to_numpy()
    return np.stack(
        [normalizar(fila, mano) for fila, mano in zip(filas, manos)]
    ).astype(np.float32)


def dividir_estratificado(
    df: pd.DataFrame, random_state: int = SEMILLA
) -> Dict[str, np.ndarray]:
    """Divide el dataset en 70% train, 15% validación y 15% prueba.

    La estratificación preserva la proporción de las 6 clases en cada partición,
    usando `random_state=42` para reproducibilidad.

    Args:
        df: Dataset limpio con el esquema completo.
        random_state: Semilla de la partición.

    Returns:
        Dict con las llaves X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = construir_matriz_x(df)
    y = df["label"].to_numpy()

    # Primer corte 70/30; el 30% restante se reparte en partes iguales 15/15.
    X_train, X_resto, y_train, y_resto = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_resto,
        y_resto,
        test_size=0.50,
        stratify=y_resto,
        random_state=random_state,
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

