# Backlog de Tareas del Proyecto (Tasks)

Este archivo es la fuente de trabajo operativo. Las tareas se ejecutan **una a la vez**, respetando sus dependencias y validando con el comando de verificación antes de marcar el checkbox.

---

## Fase F0: Entorno y Verificación de Hardware

- [x] **T-01: Configuración de Entorno Virtual y Dependencias Base**
  - **Fase:** F0
  - **Owner:** `[AGENTE]`
  - **Dependencias:** Ninguna
  - **Archivos que toca:** `requirements.txt`, `.gitignore`
  - **Criterio de aceptación:** Archivo `requirements.txt` fijado con versiones estables (`opencv-python`, `mediapipe`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pytest`, `matplotlib`, `seaborn`) que instala sin conflictos en `.venv`.
  - **Comando de verificación:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 -c "import cv2, mediapipe, sklearn, pandas, joblib; print('ENV_OK')"`

- [ ] **T-02: Hello World de MediaPipe Hands**
  - **Fase:** F0
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-01
  - **Archivos que toca:** `scripts/test_mediapipe.py`
  - **Criterio de aceptación:** Script sintético que crea una imagen en negro con OpenCV, pasa por MediaPipe Hands y confirma que la API inicializa sin fallas de símbolos ni librerías dinámicas faltantes.
  - **Comando de verificación:** `python3 scripts/test_mediapipe.py`

- [ ] **T-03: Setup y Validación del Feed de Cámara**
  - **Fase:** F0
  - **Owner:** `[HUMANO]`
  - **Dependencias:** T-01, T-02
  - **Archivos que toca:** `scripts/test_cam.py`
  - **Criterio de aceptación:** Verificación física de la cámara (celular vía `scrcpy`/ADB o webcam integrada en `/dev/video*`) mostrando una ventana OpenCV con video fluido a $\ge 20$ FPS y landmarks de la mano sobreimpresos.
  - **Comando de verificación:** `python3 scripts/test_cam.py`

---

## Fase F1: Adquisición del Dataset Propio

- [ ] **T-04: Módulo Matemático de Normalización Geométrica y Tests Unitarios**
  - **Fase:** F1
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-01
  - **Archivos que toca:** `scripts/utils.py`, `tests/test_normalizar.py`
  - **Criterio de aceptación:** Implementación de `normalizar()` cumpliendo traslación al landmark 0, escalamiento con landmark 9 e inversión en eje $x$ para mano izquierda. Cobertura de tests unitarios pasando al 100%.
  - **Comando de verificación:** `pytest tests/test_normalizar.py -v`

- [ ] **T-05: Script Interactivo de Captura de Muestras**
  - **Fase:** F1
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-03, T-04
  - **Archivos que toca:** `scripts/capturar.py`
  - **Criterio de aceptación:** Script de consola/OpenCV que permite seleccionar una de las 6 clases (`palma`, `senalar`, `paz`, `pulgar`, `puno`, `neutro`), muestra un conteo regresivo, captura landmarks en tiempo real y los apéndice con el esquema estricto en `data/dataset.csv`.
  - **Comando de verificación:** `python3 scripts/capturar.py --help`

- [ ] **T-06: Grabación Física del Dataset de Gestos**
  - **Fase:** F1
  - **Owner:** `[HUMANO]`
  - **Dependencias:** T-05
  - **Archivos que toca:** `data/dataset.csv`
  - **Criterio de aceptación:** Obtención de al menos 350 cuadros por clase (para margen de limpieza), variando distancia (40–70 cm), ángulos leves, iluminación y ambas manos (izquierda y derecha).
  - **Comando de verificación:** `python3 -c "import pandas as pd; df = pd.read_csv('data/dataset.csv'); print(df['label'].value_counts()); assert all(df['label'].value_counts() >= 300), 'Faltan muestras'"`

---

## Fase F2: Procesamiento, Modelado y Evaluación

- [ ] **T-07: Limpieza, Validación y Split Estratificado de Datos**
  - **Fase:** F2
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-06
  - **Archivos que toca:** `notebooks/analisis.ipynb`, `scripts/utils.py`
  - **Criterio de aceptación:** Filtrado de filas con valores nulos o landmarks anómalos. División estratificada 70% train, 15% val y 15% test con `random_state=42`. Verificación de balance de clases.
  - **Comando de verificación:** `pytest tests/test_datos.py -v`

- [ ] **T-08: Entrenamiento y Comparación de Clasificadores ML con Visualización PCA 2D**
  - **Fase:** F2
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-07
  - **Archivos que toca:** `notebooks/analisis.ipynb`, `models/metricas.json`
  - **Criterio de aceptación:** Entrenamiento y evaluación comparativa de SVM (RBF/Lineal), KNN, Random Forest y MLP. Proyección PCA 2D de los clusters de gestos guardada en figura. Tabla comparativa de Accuracy y F1 macro.
  - **Comando de verificación:** `python3 -c "import json; m=json.load(open('models/metricas.json')); print(m); assert m['mejor_modelo']['f1_macro'] >= 0.90"`

- [ ] **T-09: Selección, Serialización del Mejor Modelo y Matriz de Confusión**
  - **Fase:** F2
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-08
  - **Archivos que toca:** `models/modelo.joblib`, `models/matriz_confusion.png`, `models/metadata.json`
  - **Criterio de aceptación:** Exportación del mejor modelo en `models/modelo.joblib` verificando que puede ser cargado con `joblib.load()` e inferir sobre un vector sintético de 63 dimensiones en $< 5\text{ ms}$.
  - **Comando de verificación:** `python3 -c "import joblib, numpy as np; m = joblib.load('models/modelo.joblib'); res = m.predict(np.zeros((1, 63))); print('MODELO_VALIDO:', res)"`

---

## Fase F3: Prototipo en Vivo e Integración MPRIS

- [ ] **T-10: Lógica del Filtro de Disparo (Debounce + Cooldown) y Tests Unitarios**
  - **Fase:** F3
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-04
  - **Archivos que toca:** `scripts/filtro.py`, `tests/test_filtro.py`
  - **Criterio de aceptación:** Módulo desacoplado de la máquina de estados: no dispara con $< 5$ frames continuos, dispara exactamente al quinto frame consecutivo, bloquea nuevas emisiones durante el cooldown de 1.0 s y nunca dispara con clase `neutro`.
  - **Comando de verificación:** `pytest tests/test_filtro.py -v`

- [ ] **T-11: Implementación del Script de Demo en Tiempo Real (`demo.py`)**
  - **Fase:** F3
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-03, T-09, T-10
  - **Archivos que toca:** `scripts/demo.py`
  - **Criterio de aceptación:** Aplicación interactiva completa con OpenCV: carga `modelo.joblib`, procesa la cámara, muestra retroalimentación visual (nombre del gesto, barra de progreso del filtro de 5 frames, estado de cooldown) y ejecuta `playerctl` de forma asíncrona.
  - **Comando de verificación:** `python3 scripts/demo.py --dry-run --timeout 5`

- [ ] **T-12: Verificación de Integración en Vivo con Spotify [HUMANO]**
  - **Fase:** F3
  - **Owner:** `[HUMANO]`
  - **Dependencias:** T-11
  - **Archivos que toca:** Ninguno (prueba de usuario)
  - **Criterio de aceptación:** Prueba interactiva con Spotify abierto en Linux: verificar que cada uno de los 5 gestos activos acciona la música correctamente y que la mano en reposo (`neutro`) no interfiere con la canción.
  - **Comando de verificación:** `playerctl status && python3 scripts/demo.py`

---

## Fase F4: Documentación, Presentación y Ensayo

- [ ] **T-13: Compilación del Reporte Académico y Figuras**
  - **Fase:** F4
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-08, T-09
  - **Archivos que toca:** `reporte/reporte.md` (o LaTeX/PDF), `reporte/figuras/`
  - **Criterio de aceptación:** Reporte completo con introducción, metodología, resultados cuantitativos reales del notebook (tabla de métricas y matrices de confusión) y discusión técnica.
  - **Comando de verificación:** `test -f reporte/reporte.md && echo "REPORTE_PRESENTE"`

- [ ] **T-14: Actualización de Presentación Presenterm con Datos Reales**
  - **Fase:** F4
  - **Owner:** `[AGENTE]`
  - **Dependencias:** T-08, T-13
  - **Archivos que toca:** `docs/propuesta.md`
  - **Criterio de aceptación:** Diapositivas en Markdown sincronizadas con las métricas finales alcanzadas por el modelo y verificables con `presenterm`.
  - **Comando de verificación:** `test -f docs/propuesta.md && which presenterm`

- [ ] **T-15: Ensayo General de Presentación y Demo [HUMANO]**
  - **Fase:** F4
  - **Owner:** `[HUMANO]`
  - **Dependencias:** T-12, T-14
  - **Archivos que toca:** Ninguno (ensayo humano)
  - **Criterio de aceptación:** Corrida fluida de la presentación en `presenterm` y ejecución de la demo en vivo frente al público objetivo sin fallas de conexión ni bloqueos.
  - **Comando de verificación:** `presenterm docs/propuesta.md`
