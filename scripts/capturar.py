#!/usr/bin/env python3
"""Script interactivo de captura guiada de landmarks para dataset.csv.

Uso:
    python scripts/capturar.py --clase palma
    python scripts/capturar.py --clase palma --muestras 400 --cam 0
    python scripts/capturar.py --borrar-clase palma
    python scripts/capturar.py --borrar-todo

Clases disponibles: palma, senalar, paz, pulgar, puno, neutro

Presiona 'q' en cualquier momento para cancelar la captura.
"""

import argparse
import csv
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.vision import construir_detector, detectar_mano, dibujar_esqueleto

DATASET_PATH = ROOT_DIR / "data" / "dataset.csv"

CLASES_VALIDAS = ["palma", "senalar", "paz", "pulgar", "puno", "neutro"]

ETIQUETAS_GESTOS = {
    "palma":   "Palma abierta (como deteniendo algo)",
    "senalar": "Señalar (solo índice extendido)",
    "paz":     "Paz o V (índice y medio extendidos)",
    "pulgar":  "Pulgar arriba (thumbs up)",
    "puno":    "Puño cerrado",
    "neutro":  "Mano relajada o posición de descanso",
}

COLUMNAS = (
    [f"x{i}" for i in range(21)]
    + [f"y{i}" for i in range(21)]
    + [f"z{i}" for i in range(21)]
    + ["label", "handedness", "session_id", "timestamp"]
)


def normalizar(pts: np.ndarray) -> np.ndarray:
    """Normaliza los 21 landmarks: traslación a muñeca, escala al nudillo medio."""
    pts = pts.copy()
    pts -= pts[0]
    escala = float(np.linalg.norm(pts[9]))
    if escala > 1e-6:
        pts /= escala
    return pts


def escribir_cabecera() -> None:
    """Escribe la cabecera del CSV, reemplazando el contenido existente."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(COLUMNAS)


def inicializar_csv() -> None:
    """Crea el CSV con cabeceras si no existe."""
    if not DATASET_PATH.exists():
        escribir_cabecera()
        print(f"[INFO] Dataset creado en {DATASET_PATH}")


def borrar_clase(clase: str) -> int:
    """Elimina todas las muestras de una clase. Retorna cuántas filas se borraron."""
    if not DATASET_PATH.exists():
        return 0
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    conservadas = [fila for fila in filas if fila.get("label") != clase]
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()
        writer.writerows(conservadas)
    return len(filas) - len(conservadas)


def borrar_todo() -> int:
    """Elimina todas las muestras conservando la cabecera. Retorna filas borradas."""
    if not DATASET_PATH.exists():
        return 0
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    escribir_cabecera()
    return n


def cuenta_regresiva(cap: cv2.VideoCapture, clase: str, segundos: int = 3) -> bool:
    """Muestra cuenta regresiva sobre el video en vivo. Retorna False si el usuario cancela."""
    desc = ETIQUETAS_GESTOS.get(clase, clase)
    fin = time.time() + segundos
    while time.time() < fin:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        restante = int(fin - time.time()) + 1
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"GESTO: {clase.upper()}", (w//2 - 160, h//2 - 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 100), 2)
        cv2.putText(frame, desc, (w//2 - 220, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Iniciando en  {restante}...", (w//2 - 130, h//2 + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 200, 255), 2)
        cv2.putText(frame, "Presiona 'q' para cancelar", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.imshow("Captura de Dataset", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def capturar_clase(
    clase: str,
    muestras_objetivo: int,
    cap: cv2.VideoCapture,
    detector: vision.HandLandmarker,
) -> int:
    """Bucle principal de captura. Retorna cuántas muestras se grabaron."""
    session_id = str(uuid.uuid4())[:8]
    muestras_capturadas = 0

    inicializar_csv()

    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        while muestras_capturadas < muestras_objetivo:
            ok, frame = cap.read()
            if not ok:
                continue

            # MediaPipe detecta sobre el frame SIN flip → handedness correcto
            lms, handedness_raw = detectar_mano(detector, frame)

            # Flip solo para display (efecto espejo natural al usuario)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            mano_detectada = lms is not None
            estado_color = (0, 220, 100) if mano_detectada else (0, 80, 255)

            if mano_detectada:
                dibujar_esqueleto(frame, lms, espejo=True)

                # Coordenadas crudas 3D
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

                # Normalizar (el flip de handedness se aplicará en la función normalizar de utils.py
                # con el handedness correcto; aquí guardamos coords normalizadas + handedness raw)
                pts_norm = normalizar(pts)

                # El espejado definitivo se hace con handedness_raw en el notebook/training
                # para mantener datos auténticos
                vec = np.concatenate([pts_norm[:, 0], pts_norm[:, 1], pts_norm[:, 2]])

                ts = time.time()
                fila = list(vec) + [clase, handedness_raw, session_id, f"{ts:.4f}"]
                writer.writerow(fila)
                f.flush()
                muestras_capturadas += 1

            # HUD - Información de estado
            pct = muestras_capturadas / muestras_objetivo
            barra_w = w - 40
            cv2.rectangle(frame, (20, h - 50), (20 + barra_w, h - 30), (60, 60, 60), -1)
            cv2.rectangle(frame, (20, h - 50), (20 + int(barra_w * pct), h - 30),
                          estado_color, -1)
            cv2.putText(frame, f"{muestras_capturadas} / {muestras_objetivo}",
                        (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

            cv2.putText(frame, f"CAPTURANDO: {clase.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 100), 2)

            if not mano_detectada:
                cv2.putText(frame, "MANO NO DETECTADA - acomoda la mano", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
            else:
                cv2.putText(frame, "Grabando...", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, estado_color, 2)

            cv2.putText(frame, "Presiona 'q' para cancelar", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
            cv2.imshow("Captura de Dataset", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[WARN] Captura cancelada por el usuario.")
                break

    return muestras_capturadas


def resumen_dataset() -> None:
    """Imprime en consola un resumen de las clases actualmente en el dataset."""
    if not DATASET_PATH.exists():
        return
    conteo: dict[str, int] = {}
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            etiq = fila.get("label", "")
            conteo[etiq] = conteo.get(etiq, 0) + 1
    if conteo:
        print("\n── Estado actual del dataset ──────────────────")
        for cl in CLASES_VALIDAS:
            n = conteo.get(cl, 0)
            barra = "█" * (n // 10) + "░" * max(0, 40 - n // 10)
            marca = " ✓" if n >= 300 else f" ({300 - n} faltan)"
            print(f"  {cl:10s} {barra} {n:>4d}{marca}")
        print("────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Captura interactiva de landmarks de mano para el dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--clase",
        choices=CLASES_VALIDAS,
        help=f"Gesto a capturar (requerido salvo con --borrar-todo):\n"
        + "\n".join(f"  {c}: {d}" for c, d in ETIQUETAS_GESTOS.items()),
    )
    parser.add_argument(
        "--muestras",
        type=int,
        default=400,
        help="Número de muestras a capturar (default: 400)",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Índice de cámara (default: 0)",
    )
    borrado = parser.add_mutually_exclusive_group()
    borrado.add_argument(
        "--borrar-clase",
        choices=CLASES_VALIDAS,
        metavar="CLASE",
        help="Borra todas las muestras de la clase indicada y sale.",
    )
    borrado.add_argument(
        "--borrar-todo",
        action="store_true",
        help="Borra todas las muestras del dataset (conserva la cabecera) y sale.",
    )
    args = parser.parse_args()

    if args.clase and (args.borrar_clase or args.borrar_todo):
        parser.error("No combines --clase con las opciones de borrado.")

    if args.borrar_clase:
        n = borrar_clase(args.borrar_clase)
        print(f"\n✓ {n} muestras de '{args.borrar_clase}' eliminadas de {DATASET_PATH}")
        resumen_dataset()
        return

    if args.borrar_todo:
        n = borrar_todo()
        print(f"\n✓ Dataset vaciado: {n} muestras eliminadas de {DATASET_PATH}")
        resumen_dataset()
        return

    if not args.clase:
        parser.error("--clase es requerido para capturar (o usa --borrar-todo)")

    print(f"\n── Captura de gesto: {args.clase.upper()} ──────────────────────────────────")
    print(f"   Descripción : {ETIQUETAS_GESTOS[args.clase]}")
    print(f"   Muestras    : {args.muestras}")
    print(f"   Cámara      : /dev/video{args.cam}")
    print(f"   Destino     : {DATASET_PATH}")
    print("────────────────────────────────────────────────────────────────")

    resumen_dataset()

    detector = construir_detector()
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir /dev/video{args.cam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cuenta_regresiva(cap, args.clase, segundos=4):
        print("Captura cancelada.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    n = capturar_clase(args.clase, args.muestras, cap, detector)
    cap.release()
    cv2.destroyAllWindows()

    if n >= args.muestras:
        print(f"\n✓ Captura completa: {n} muestras de '{args.clase}' guardadas en {DATASET_PATH}")
    else:
        print(f"\n! Captura parcial: {n} muestras guardadas (objetivo: {args.muestras})")

    resumen_dataset()


if __name__ == "__main__":
    main()
