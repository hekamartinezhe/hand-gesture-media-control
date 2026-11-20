#!/usr/bin/env python3
"""Verificación visual del feed de cámara con landmarks de MediaPipe dibujados en tiempo real.

Uso:
    python scripts/test_cam.py [--cam 0]

Presiona 'q' para salir.
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"

# Colores de los landmarks en BGR
COLOR_PUNTO = (0, 255, 0)
COLOR_LINEA = (255, 255, 255)
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),          # Pulgar
    (0,5),(5,6),(6,7),(7,8),           # Índice
    (0,9),(9,10),(10,11),(11,12),      # Medio
    (0,13),(13,14),(14,15),(15,16),    # Anular
    (0,17),(17,18),(18,19),(19,20),    # Meñique
    (5,9),(9,13),(13,17),              # Palma
]


def construir_detector() -> vision.HandLandmarker:
    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encuentra {MODEL_PATH}.")
        print("Ejecuta primero: python scripts/test_mediapipe.py")
        sys.exit(1)
    base_opts = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def dibujar_landmarks(frame: np.ndarray, landmarks_norm: list, w: int, h: int, espejo: bool = False) -> None:
    """Dibuja el esqueleto de la mano sobre el frame."""
    puntos = [(int((1 - lm.x) * w) if espejo else int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm]
    for a, b in CONEXIONES:
        cv2.line(frame, puntos[a], puntos[b], COLOR_LINEA, 1)
    for px, py in puntos:
        cv2.circle(frame, (px, py), 4, COLOR_PUNTO, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de cámara con detección de mano en tiempo real.")
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara (default: 0)")
    args = parser.parse_args()

    detector = construir_detector()
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.cam}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Cámara /dev/video{args.cam} abierta. Presiona 'q' para salir.")
    fps_hist: list[float] = []
    tick = cv2.getTickFrequency()
    t_prev = cv2.getTickCount()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame no recibido, reintentando...")
            continue

        # MediaPipe detecta sobre frame SIN flip → handedness correcto
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect(mp_img)

        # Flip solo para display (efecto espejo)
        frame = cv2.flip(frame, 1)

        t_now = cv2.getTickCount()
        fps = tick / (t_now - t_prev)
        t_prev = t_now
        fps_hist.append(fps)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        fps_prom = sum(fps_hist) / len(fps_hist)

        if resultado.hand_landmarks:
            lms = resultado.hand_landmarks[0]
            dibujar_landmarks(frame, lms, w, h, espejo=True)
            nombre_mano = resultado.handedness[0][0].display_name if resultado.handedness else "?"
            cv2.putText(frame, f"Mano: {nombre_mano}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
            cv2.putText(frame, "MANO DETECTADA", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Sin mano", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        cv2.putText(frame, f"FPS: {fps_prom:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, "Presiona 'q' para salir", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Test Camara — MediaPipe Hands", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFPS promedio: {sum(fps_hist)/len(fps_hist):.1f}")


if __name__ == "__main__":
    main()
