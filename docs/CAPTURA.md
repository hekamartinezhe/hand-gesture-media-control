# Guía de Captura del Dataset

## Resumen

El dataset se genera con `scripts/capturar.py`. El script abre la cámara, detecta la mano con MediaPipe, normaliza los 21 landmarks 3D y los guarda en `data/dataset.csv`.

Cada clase necesita **mínimo 300 muestras** (recomendado: 400).

---

## Comandos por clase

```bash
cd Proyecto_2do_Parcial
source .venv/bin/activate

python scripts/capturar.py --clase palma   --muestras 400
python scripts/capturar.py --clase senalar --muestras 400
python scripts/capturar.py --clase paz     --muestras 400
python scripts/capturar.py --clase pulgar  --muestras 400
python scripts/capturar.py --clase puno    --muestras 400
python scripts/capturar.py --clase neutro  --muestras 400
```

> Si se cancela antes de llegar al objetivo, el script guarda lo capturado hasta ese momento. Puedes volver a correr el mismo comando para seguir acumulando muestras: el CSV se **agrega** (no sobreescribe).

---

## Flags de `capturar.py`

| Flag | Tipo | Default | Descripción |
|------|------|---------|-------------|
| `--clase` | str | *requerido para capturar* | Gesto a capturar: `palma`, `senalar`, `paz`, `pulgar`, `puno`, `neutro` |
| `--muestras` | int | `400` | Número de muestras objetivo para esta sesión |
| `--cam` | int | `0` | Índice de cámara (`/dev/video0` → `0`, `/dev/video1` → `1`) |
| `--borrar-clase` | str | — | Borra todas las muestras de la clase indicada y sale |
| `--borrar-todo` | flag | — | Borra todas las muestras del dataset y sale (conserva la cabecera) |

**Ejemplos:**

```bash
# Capturar 400 muestras de "paz" con cámara por defecto
python scripts/capturar.py --clase paz

# Capturar solo 100 muestras con cámara secundaria
python scripts/capturar.py --clase puno --muestras 100 --cam 1

# Borrar todas las muestras de "palma" antes de recapturar
python scripts/capturar.py --borrar-clase palma

# Vaciar el dataset completo (deja solo la cabecera)
python scripts/capturar.py --borrar-todo
```

> Los flags de borrado no abren la cámara: solo modifican el CSV y terminan. No se combinan con `--clase`. El borrado es inmediato e irreversible.

---

## Flags de `test_cam.py`

```bash
python scripts/test_cam.py [--cam 0]
```

| Flag | Tipo | Default | Descripción |
|------|------|---------|-------------|
| `--cam` | int | `0` | Índice de cámara a abrir |

Muestra el feed en tiempo real con esqueleto de landmarks y FPS. Presiona `q` para salir. Útil para verificar que la cámara y MediaPipe funcionan antes de capturar.

---

## Gestos disponibles

| Clase | Descripción | Cómo hacerlo |
|-------|-------------|--------------|
| `palma` | Palma abierta, dedos juntos o separados, enfrentando la cámara | Mano extendida hacia el frente |
| `senalar` | Solo el índice extendido, resto cerrado | Apunta directo a la cámara |
| `paz` | Índice y medio extendidos (V de victoria) | Signo de paz hacia la cámara |
| `pulgar` | Pulgar arriba, resto cerrado (thumbs up) | Lateral o frontal |
| `puno` | Todos los dedos cerrados | Puño al frente |
| `neutro` | Mano relajada, posición de descanso natural | Sin gesto definido, dedos sueltos |

---

## Técnica de captura

El gesto debe ser **estático** durante cada ráfaga. El script captura frames independientes, no video.

**Lo que sí varía** (da robustez al clasificador):

- **Distancia:** 40–70 cm de la cámara
- **Ángulo:** inclina la muñeca ~15° en distintas direcciones
- **Rotación leve:** gira la mano sobre su eje sin cambiar el gesto
- **Ambas manos:** alterna izquierda y derecha (~50/50 por clase)

**Lo que NO debes hacer:**

- Mover la mano tan rápido que MediaPipe pierda el track
- Cambiar el gesto a mitad de la captura
- Capturar con contraluz o fondo muy cargado (reduce detección)

---

## Verificación del dataset

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/dataset.csv')
print(df['label'].value_counts().sort_index())
assert all(df['label'].value_counts() >= 300), 'Faltan muestras en alguna clase'
print('DATASET_OK')
"
```

---

## Schema del CSV (`data/dataset.csv`)

67 columnas totales:

| Columnas | Descripción |
|----------|-------------|
| `x0`–`x20` | Coordenada X normalizada de cada landmark |
| `y0`–`y20` | Coordenada Y normalizada de cada landmark |
| `z0`–`z20` | Coordenada Z normalizada de cada landmark |
| `label` | Clase del gesto (`palma`, `senalar`, etc.) |
| `handedness` | Mano detectada por MediaPipe (`Left` / `Right`) |
| `session_id` | UUID corto de la sesión de captura |
| `timestamp` | Unix timestamp del frame |

> La normalización (traslación a muñeca + escala al nudillo medio) se aplica en captura. El flip de coordenadas X para manos izquierdas se aplica en el notebook de entrenamiento.
