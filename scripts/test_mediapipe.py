#!/usr/bin/env python3
"""Prueba sintética de inicialización de MediaPipe Hands.

Verifica que el entorno de ejecución, TensorFlow Lite y MediaPipe Tasks
carguen correctamente en CPU sin fallas de enlaces ni símbolos dinámicos.
"""

from pathlib import Path
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# URL oficial del modelo Hand Landmarker bundle de Google MediaPipe
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"


def asegurar_modelo() -> Path:
    """Verifica que el archivo .task exista o lo descarga si está ausente."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo Hand Landmarker a {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Descarga completada.")
    return MODEL_PATH


def main() -> None:
    model_path = asegurar_modelo()

    # Configuración de detector para CPU
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Inicializando HandLandmarker en CPU...")
    detector = vision.HandLandmarker.create_from_options(options)

    # Crear cuadro sintético en negro (640x480, BGR estándar de OpenCV)
    frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Ejecutar inferencia de prueba
    resultado = detector.detect(mp_image)

    # Validar que devolvió una estructura válida (esperamos 0 manos en cuadro negro)
    assert hasattr(resultado, "hand_landmarks"), "El resultado no contiene 'hand_landmarks'"
    print(f"Inferencia ejecutada con éxito. Manos detectadas: {len(resultado.hand_landmarks)}")
    print("MEDIAPIPE_OK")


if __name__ == "__main__":
    main()
