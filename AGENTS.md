# Guía Operativa para Agentes — Hand Gesture Media Control

## 1. Proyecto y Stack
Control de Spotify por gestos en tiempo real mediante clasificación de 21 landmarks 3D con ML clásico.
- **Hardware/SO:** ThinkPad T480 (CPU i5), Linux (Wayland/Niri), cámara Android vía scrcpy/ADB o webcam.
- **Stack:** Python 3.10+, MediaPipe Hands, OpenCV, scikit-learn, joblib, playerctl (MPRIS), pytest.

## 2. Mapa del Repositorio
- `specs/`: Fuente de verdad (requisitos, diseño, plan, tareas, decisiones).
- `.agents/`: Reglas obligatorias (`rules/`) y skills especializadas (`skills/`).
- `scripts/`: Código ejecutable (`capturar.py`, `demo.py`, `utils.py`).
- `data/`: Datasets propios en CSV (inmutables salvo en tareas de captura).
- `notebooks/`: EDA, entrenamiento, métricas y selección de modelo.
- `models/`: Modelos serializados (`.joblib`) y métricas exportadas.
- `reporte/`: Gráficas, figuras y reporte académico.
- `docs/`: Presentaciones (`propuesta.md`) y guías de referencia.

## 3. Comandos Clave
```bash
python3 -m venv .venv && source .venv/bin/activate  # Entorno virtual
pip install -r requirements.txt                    # Dependencias
pytest tests/ -v                                   # Verificación de tests
python scripts/demo.py                             # Demo en tiempo real
presenterm docs/propuesta.md                       # Slides de presentación
```

## 4. Convenciones
- **Idioma:** Español en documentación, comentarios y nombres descriptivos de variables.
- **Código:** Tipado estático simple (PEP 484), funciones puras con `pathlib.Path` (sin rutas absolutas fijas).
- **Estándar:** Clean code, modularidad estricta y zero speculative code.

## 5. Flujo de Trabajo Agéntico
1. Consultar `specs/tasks.md` y tomar **UNA sola tarea** con owner `[AGENTE]` que no esté bloqueada.
2. Leer solo las secciones pertinentes de `specs/requirements.md` y `specs/design.md`.
3. Implementar la solución mínima que cumpla los criterios de aceptación.
4. Ejecutar el comando exacto de verificación listado en la tarea.
5. Marcar `[x]` en `specs/tasks.md`, registrar en `specs/decisions.md` si hubo decisión de diseño.
6. Proponer mensaje de commit conventional (`feat:`, `fix:`, `test:`, `docs:`).

## 6. Qué NO Hacer
- **NO** implementar código de tareas futuras ni adelantar pasos.
- **NO** modificar ni borrar contenido en `data/` sin confirmación previa del usuario.
- **NO** instalar paquetes fuera de `.venv/` ni solicitar permisos `sudo` innecesarios.
- **NO** inventar métricas ni resultados no respaldados por ejecuciones reales.
