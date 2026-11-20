# Control Multimedia por Gestos de Mano (ML Clásico)

Sistema de reconocimiento de gestos de mano en tiempo real para el control multimedia de Spotify en Linux, utilizando 21 landmarks 3D extraídos con **MediaPipe Hands** y clasificados mediante algoritmos clásicos de Machine Learning (**SVM**, **KNN**, **Random Forest** y **MLP**), con **suavizado temporal** y un **Random Forest entrenado con capturas reales** para robustez ante cambios de mano y cámara.

---

## 🎯 Objetivo
Controlar la reproducción de Spotify sin contacto físico con el equipo, manteniendo una inferencia fluida en CPU (ThinkPad T480), con alta robustez ante variaciones de posición/escala, respuesta inmediata y sin repeticiones gracias a un pestillo anti-repetición y un cooldown de 0.5 s.

---

## ✋ Gestos y Acciones

| Gesto | Acción Spotify | Comando MPRIS (`playerctl`) |
| :--- | :--- | :--- |
| **Palma abierta** | Play / Pausa | `playerctl play-pause` |
| **Señalar (Índice)** | Siguiente canción | `playerctl next` |
| **Paz (Índice + Medio)** | Canción anterior | `playerctl previous` |
| **Pulgar arriba** | Subir volumen (+10%) | `playerctl volume 0.1+` |
| **Puño cerrado** | Bajar volumen (-10%) | `playerctl volume 0.1-` |
| **Neutro / Mano relajada** | Ninguna acción | *(Ignorado por diseño)* |

---

## 🖐️ Demo en tiempo real

El demo decide la acción con un **Random Forest entrenado** (`models/modelo_rf_vivo.joblib`) sobre capturas reales de ambas manos. El modelo usa 64 características geométricas continuas y se calibró con 689 cuadros (570 base + 119 de corrección); no hay reglas `if/else` decidiendo la acción. La regla geométrica queda como baseline interpretable.

```bash
python scripts/demo.py                    # uso normal: el gesto se ejecuta al instante
python scripts/demo.py --dry-run          # no ejecuta playerctl
python scripts/demo.py --mostrar-ml       # muestra la predicción del modelo y la regla en el HUD
python scripts/demo.py --usar-regla       # baseline: decide con la regla geométrica
python scripts/demo.py --suavizado 0.3    # más suavizado (más retardo)
python scripts/entrenar_vivo.py           # reproduce el modelo de la demo
```

- El modelo se entrenó con landmarks suavizados con **EMA** (`alfa=0.4`), igual que en el demo.
- **Sin confirmación:** el gesto se ejecuta en el primer cuadro que el modelo predice; no hay Enter ni filtro de 5 cuadros en la demo.
- **Un disparo por presentación:** tras ejecutar una acción, el pestillo bloquea repeticiones mientras el gesto se mantiene. Se rearma al bajar la mano (3 cuadros) o al cambiar a otro gesto durante 5 cuadros.
- Se conserva un **cooldown de 0.5 s** entre acciones para evitar ráfagas.
- `neutro`, ausencia de mano y etiquetas desconocidas nunca disparan.
- Rendimiento auditado: **93.0%** mano izquierda y **96.1%** derecha en validación cruzada del modelo base; **100%** en una sesión independiente de 190 cuadros; **7.7 ms** de inferencia por muestra.

---

## 🛠️ Stack Tecnológico y Requisitos
- **Sistema Operativo:** Linux (Wayland / X11).
- **Procesador:** Intel Core i5 (ThinkPad T480, optimizado para ejecución en CPU sin GPU dedicada).
- **Control Multimedia:** `playerctl` sobre la interfaz MPRIS de Spotify.
- **Captura:** Teléfono Android vía `scrcpy`/ADB o cámara web integrada/USB.
- **Lenguaje:** Python 3.10+.
- **Librerías principales:** MediaPipe Hands, OpenCV, scikit-learn, pandas, numpy, joblib, pytest.

---

## 📂 Estructura del Proyecto

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── docs/                      # Presentación en Markdown y guía de captura
│   ├── propuesta.md
│   └── CAPTURA.md
├── scripts/                   # Captura, entrenamiento y demo
│   ├── capturar.py
│   ├── demo.py
│   ├── entrenar_geometrico.py
│   ├── entrenar_vivo.py
│   ├── filtro.py
│   ├── geometria.py
│   ├── modelado.py
│   ├── utils.py
│   └── vision.py
├── data/
│   ├── dataset.csv
│   └── vivo/
│       ├── base_ambas_manos.csv
│       ├── correccion_izquierda.csv
│       └── sesion_independiente.csv
├── notebooks/
│   └── analisis.ipynb
├── models/                    # Modelos entrenados y métricas
├── tests/                     # Pruebas automatizadas
└── reporte/                   # Reporte final, presentación y figuras
    ├── reporte.pdf
    ├── presentacion.pdf
    └── reporte.md
```

---

## 🚀 Metodología de trabajo
1. **Captura:** se graban 400 muestras por gesto con `scripts/capturar.py` y MediaPipe.
2. **Análisis y limpieza:** el notebook revisa nulos, duplicados, balance y normalización.
3. **Modelado:** se comparan SVM, KNN, Random Forest y MLP con una partición 70/15/15.
4. **Adaptación al dominio:** se capturan datos con ambas manos, se entrena el Random Forest de la demo y se valida con una sesión independiente.
5. **Verificación:** `pytest tests/ -q` cubre normalización, filtro, modelo y demo.
