"""Utilidades compartidas de MediaPipe Hands y dibujo de landmarks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
_ULTIMO_TIMESTAMP_MS = -1


COLOR_PUNTO = (0, 255, 100)
COLOR_LINEA = (255, 255, 255)
CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def construir_detector(num_manos: int = 1) -> vision.HandLandmarker:
    """Crea el detector de manos en CPU a partir del modelo local."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra {MODEL_PATH}. Ejecuta: python scripts/test_mediapipe.py"
        )

    base_opts = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_manos,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detectar_mano(
    detector: vision.HandLandmarker, frame: np.ndarray
) -> tuple[Optional[list[Any]], str]:
    """Detecta una mano sobre el frame BGR sin voltear.

    Usa el modo VIDEO de MediaPipe: el seguimiento entre cuadros evita repetir
    la detección completa y reduce la latencia en CPU.

    Returns:
        (landmarks, handedness): landmarks es None si no hay detección.
    """
    global _ULTIMO_TIMESTAMP_MS

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = max(
        int(time.monotonic() * 1000), _ULTIMO_TIMESTAMP_MS + 1
    )
    _ULTIMO_TIMESTAMP_MS = timestamp_ms
    resultado = detector.detect_for_video(mp_img, timestamp_ms)

    if not resultado.hand_landmarks:
        return None, "Right"

    landmarks = resultado.hand_landmarks[0]
    handedness = "Right"
    if resultado.handedness:
        handedness = resultado.handedness[0][0].display_name
    return landmarks, handedness


def dibujar_esqueleto(
    frame: np.ndarray,
    landmarks: list[Any],
    espejo: bool = False,
) -> None:
    """Dibuja puntos y conexiones; `espejo` compensa el flip horizontal del display."""
    alto, ancho = frame.shape[:2]
    puntos = [
        (
            int((1 - lm.x) * ancho) if espejo else int(lm.x * ancho),
            int(lm.y * alto),
        )
        for lm in landmarks
    ]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], COLOR_LINEA, 1)
    for px, py in puntos:
        cv2.circle(frame, (px, py), 5, COLOR_PUNTO, -1)
