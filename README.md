# Control Multimedia por Gestos de Mano (ML Clásico)

Sistema de reconocimiento de gestos de mano en tiempo real para el control multimedia de Spotify en Linux, utilizando 21 landmarks 3D extraídos con **MediaPipe Hands** y clasificados mediante algoritmos clásicos de Machine Learning (**SVM**, **KNN**, **Random Forest** y **MLP**).

---

## 🎯 Objetivo
Controlar la reproducción de Spotify sin contacto físico con el equipo, manteniendo una inferencia fluida en CPU (ThinkPad T480), con alta robustez ante variaciones de posición/escala y cero disparos accidentales gracias a un filtro temporal de confirmación.

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
├── AGENTS.md                  # Protocolo y directrices de trabajo agéntico
├── GEMINI.md                  # Puntero de configuración de agente
├── README.md                  # Documentación general del repositorio
├── docs/                      # Presentación en presenterm y documentos base
│   └── propuesta.md
├── specs/                     # Fuente de verdad metodológica
│   ├── requirements.md        # Requisitos funcionales y no funcionales
│   ├── design.md              # Arquitectura, esquema de datos y algoritmos
│   ├── plan.md                # Cronograma por fases y gestión de riesgos
│   ├── tasks.md               # Backlog de tareas con verificación estricta
│   └── decisions.md           # Bitácora de decisiones arquitectónicas
├── .agents/                   # Reglas de negocio y skills del agente
│   ├── rules/                 # Reglas sobre datos, código y seguridad
│   └── skills/                # Flujos guiados (nueva-tarea, verificar, captura, reporte)
├── scripts/                   # Scripts de captura, demo y utilidades
├── data/                      # Dataset propio en CSV
├── notebooks/                 # Exploración, entrenamiento y evaluación
├── models/                    # Modelos entrenados (.joblib)
└── reporte/                   # Reporte académico y figuras generadas
```

---

## 🚀 Metodología: Desarrollo Guiado por Specs
Este repositorio sigue una metodología donde las especificaciones (`specs/`) son la **única fuente de verdad**. Las tareas se implementan de forma atómica e incremental:
1. Las tareas se seleccionan desde [`specs/tasks.md`](specs/tasks.md).
2. Cada tarea cuenta con un comando verificable antes de darse por completada.
3. Las decisiones arquitectónicas se registran en [`specs/decisions.md`](specs/decisions.md).
