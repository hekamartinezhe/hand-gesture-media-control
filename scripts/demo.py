#!/usr/bin/env python3
"""Demo en tiempo real: MediaPipe + modelo ML + MPRIS.

Uso:
    python scripts/demo.py
    python scripts/demo.py --cooldown 0.5
    python scripts/demo.py --dry-run --timeout 5
    python scripts/demo.py --cam 1

Por defecto la acción la decide el modelo entrenado y se ejecuta en cuanto el
gesto cambia. Para repetir el mismo gesto hay que bajarlo o cambiarlo y volver
a formarlo. La regla geométrica queda disponible con `--usar-regla` como
baseline interpretable. Presiona 'q' en la ventana para salir.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.filtro import FiltroDisparo, PestilloGesto
from scripts.geometria import SuavizadorLandmarks, clasificar_geometrico
from scripts.utils import despachar_gesto, normalizar
from scripts.vision import construir_detector, detectar_mano, dibujar_esqueleto

MODELO_PATH = ROOT_DIR / "models" / "modelo_rf_vivo.joblib"

DESCRIPCION_ACCION = {
    "palma": "Play / Pausa",
    "senalar": "Siguiente cancion",
    "paz": "Cancion anterior",
    "pulgar": "Volumen +10%",
    "puno": "Volumen -10%",
    "neutro": "Sin acción",
}

COLOR_TEXTO = (207, 200, 200)
COLOR_OK = (100, 220, 0)
COLOR_ALERTA = (60, 60, 255)
COLOR_BLOQUEO = (0, 200, 255)


def cargar_modelo(ruta: Path) -> object:
    """Carga el Pipeline serializado y falla con un mensaje claro si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encuentra {ruta}. Coloca el modelo en models/ "
            "o revisa la ruta con --modelo."
        )
    return joblib.load(ruta)


def dibujar_hud(
    frame: np.ndarray,
    gesto: str | None,
    filtro: FiltroDisparo,
    ahora: float,
    fps: float,
    ultima_accion: str | None,
    dry_run: bool,
    gesto_ml: str | None,
    gesto_geo: str | None,
    usar_geometria: bool,
    mostrar_ml: bool,
) -> None:
    """Dibuja gesto, acción, cooldown y FPS."""
    alto, ancho = frame.shape[:2]

    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (ancho, 150), (22, 20, 21), -1)
    cv2.addWeighted(panel, 0.78, frame, 0.22, 0, frame)

    estado = gesto.upper() if gesto else "SIN MANO"
    color_estado = COLOR_OK if gesto and gesto != "neutro" else COLOR_TEXTO
    cv2.putText(frame, f"GESTO: {estado}", (16, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color_estado, 2)

    accion = DESCRIPCION_ACCION.get(gesto or "", "—")
    cv2.putText(frame, f"Accion: {accion}", (16, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXTO, 1)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (ancho - 120, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1)

    if mostrar_ml:
        cv2.putText(frame, f"ML: {gesto_ml or '—'}", (16, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
        if usar_geometria:
            color_geo = (
                COLOR_OK if gesto_geo and gesto_geo != "neutro" else COLOR_TEXTO
            )
            cv2.putText(frame, f"GEO: {gesto_geo or '—'}", (170, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_geo, 1)

    if ultima_accion:
        cv2.putText(frame, f"Ultima: {ultima_accion}", (16, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXTO, 1)

    restante = filtro.cooldown_restante(ahora)
    if restante > 0:
        cv2.putText(frame, f"COOLDOWN {restante:.1f}s", (ancho - 190, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, COLOR_ALERTA, 2)
    else:
        cv2.putText(frame, "LISTO", (ancho - 90, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OK, 1)

    if dry_run:
        cv2.putText(frame, "DRY-RUN", (ancho - 110, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLOQUEO, 1)

    cv2.putText(
        frame,
        "Cambia de gesto para repetir  |  q: salir",
        (16, alto - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Control de Spotify por gestos con un modelo ML en tiempo real."
    )
    parser.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="Segundos de ejecución; 0 = hasta presionar 'q'",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.5,
        help="Segundos de bloqueo entre acciones (por defecto 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecuta playerctl; solo muestra las acciones detectadas",
    )
    parser.add_argument(
        "--modelo",
        type=Path,
        default=MODELO_PATH,
        help="Ruta al modelo .joblib",
    )
    parser.add_argument(
        "--suavizado",
        type=float,
        default=0.4,
        help="Factor EMA para los landmarks (0, 1]; menor = más suave (por defecto 0.4)",
    )
    parser.add_argument(
        "--usar-regla",
        action="store_true",
        help="Usa la regla geométrica interpretable en lugar del modelo entrenado",
    )
    parser.add_argument(
        "--mostrar-ml",
        action="store_true",
        help="Muestra en el HUD la predicción ML y la verificación geométrica",
    )
    args = parser.parse_args()

    if not 0.0 < args.suavizado <= 1.0:
        parser.error("--suavizado debe estar en el intervalo (0, 1].")

    try:
        modelo = cargar_modelo(args.modelo)
        detector = construir_detector()
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.cam}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    filtro = FiltroDisparo(frames_confirmacion=1, cooldown_s=args.cooldown)
    ejecutor = ThreadPoolExecutor(max_workers=1)

    inicio = time.monotonic()
    ultimo_cuadro = inicio
    fps = 0.0
    gesto: str | None = None
    gesto_ml: str | None = None
    gesto_geo: str | None = None
    ultima_accion: str | None = None
    pestillo = PestilloGesto()
    suavizador = SuavizadorLandmarks(args.suavizado)
    mostrar_ventana = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    print("[INFO] Demo iniciada.")
    print(f"[INFO] Modelo: {args.modelo}")
    modo = "regla geométrica (baseline)" if args.usar_regla else "modelo entrenado"
    print(f"[INFO] Modo: {modo}")
    print(f"[INFO] Suavizado EMA alfa={args.suavizado:.2f}")
    print(f"[INFO] Cooldown entre acciones: {args.cooldown:.2f} s")
    print("[INFO] Modo automático: el gesto se ejecuta al instante. 'q' para salir.")

    try:
        while True:
            ahora = time.monotonic()
            if args.timeout > 0 and ahora - inicio >= args.timeout:
                break

            ok, frame = cap.read()
            if not ok:
                print("[WARN] No se pudo leer la cámara.")
                break

            # Detección sobre el frame sin flip para conservar el handedness real.
            landmarks, handedness = detectar_mano(detector, frame)
            frame = cv2.flip(frame, 1)

            if landmarks is not None:
                puntos = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
                )
                puntos = suavizador.suavizar(puntos)
                landmarks_suaves = [
                    SimpleNamespace(
                        x=float(punto[0]),
                        y=float(punto[1]),
                        z=float(punto[2]),
                    )
                    for punto in puntos
                ]
                dibujar_esqueleto(frame, landmarks_suaves, espejo=True)
                vector = normalizar(puntos, handedness)
                gesto_ml = str(modelo.predict(vector.reshape(1, -1))[0])
                gesto_geo = clasificar_geometrico(puntos)
                gesto = gesto_geo if args.usar_regla else gesto_ml
            else:
                suavizador.reiniciar()
                gesto_ml = None
                gesto_geo = None
                gesto = None

            if pestillo.actualizar(gesto):
                disparo = filtro.actualizar(gesto, ahora)
            else:
                filtro.reiniciar()
                disparo = None

            if disparo is not None:
                ultima_accion = f"{disparo} → {DESCRIPCION_ACCION[disparo]}"
                if args.dry_run:
                    print(f"[DRY-RUN] {ultima_accion}", flush=True)
                else:
                    ejecutor.submit(despachar_gesto, disparo)
                    print(f"[ACCIÓN] {ultima_accion}", flush=True)
                pestillo.bloquear(disparo)

            delta = ahora - ultimo_cuadro
            ultimo_cuadro = ahora
            if delta > 0:
                fps = (1.0 / delta) if fps == 0.0 else 0.9 * fps + 0.1 / delta

            dibujar_hud(
                frame, gesto, filtro, ahora, fps, ultima_accion,
                args.dry_run, gesto_ml, gesto_geo,
                args.usar_regla or args.mostrar_ml, args.mostrar_ml,
            )

            if mostrar_ventana:
                cv2.imshow("Gesto-Control Spotify", frame)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.")
    finally:
        cap.release()
        if mostrar_ventana:
            cv2.destroyAllWindows()
        ejecutor.shutdown(wait=True)

    print("[INFO] Demo finalizada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
