"""Utilidades matemáticas, de normalización y control multimedia multiplataforma.

Centraliza las funciones puras para que sean reutilizables tanto en Linux
(vía playerctl / MPRIS) como en Windows (vía Win32 virtual media keys).
"""

import platform
import subprocess
import sys
from typing import Dict, List, Optional

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
