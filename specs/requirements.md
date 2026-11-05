# Especificación de Requisitos de Software

## 1. Objetivo del Proyecto
Desarrollar un sistema de visión por computadora y aprendizaje automático clásico que clasifique landmarks tridimensionales de la mano para controlar en tiempo real la reproducción multimedia de Spotify en Linux (ThinkPad T480), eliminando la necesidad de interactuar físicamente con el teclado o ratón.

---

## 2. Alcance y Fuera de Alcance

### 2.1 En Alcance
- Pipeline completo de desarrollo: captura de datos propia, preprocesamiento y normalización geométrica, entrenamiento comparativo de modelos, evaluación de métricas y prototipo interactivo en vivo.
- Detección de una mano dominante (derecha o izquierda con espejado) mediante MediaPipe Hands.
- Clasificación de 6 gestos estáticos bien diferenciados.
- Filtrado temporal anti-rebote (debounce) para estabilizar la inferencia y evitar ejecuciones espurias.
- Integración con el demonio MPRIS mediante el comando CLI `playerctl`.
- Ejecución eficiente y fluida 100% en CPU.

### 2.2 Fuera de Alcance
- Reconocimiento de gestos dinámicos en secuencia temporal (e.g., swipes continuos o círculos en el aire).
- Detección simultánea o bimanual (dos manos al mismo tiempo).
- Entrenamiento con redes neuronales convolucionales pesadas directamente sobre píxeles (CNNs de Deep Learning que requieran GPU dedicada).
- Control de reproductores ajenos al protocolo estándar MPRIS de Linux.

---

## 3. Requisitos Funcionales (RF)

| ID | Nombre | Descripción |
| :--- | :--- | :--- |
| **RF-01** | **Ingesta de Video** | El sistema debe capturar el flujo de video en tiempo real desde la cámara del smartphone Android (vía `scrcpy` o ADB) o desde una cámara web estándar a resolución mínima de 640×480 a ≥ 20 FPS. |
| **RF-02** | **Extracción de Landmarks** | El pipeline debe procesar cada cuadro con MediaPipe Hands y extraer 21 landmarks tridimensionales ($x_i, y_i, z_i$), totalizando 63 valores continuos por mano detectada. |
| **RF-03** | **Normalización Geométrica** | Cada vector de 63 características debe transformarse para ser invariante a la traslación (restando la muñeca), a la escala de la mano (dividiendo entre la distancia muñeca-nudillo medio), y a la lateralidad (invirtiendo el eje $x$ si es mano izquierda). |
| **RF-04** | **Captura y Generación de Dataset** | El sistema debe proveer una herramienta interactiva (`scripts/capturar.py`) para registrar muestras etiquetadas directamente en un archivo CSV estructurado. |
| **RF-05** | **Clasificación Multiclase ML** | El sistema debe comparar y evaluar clasificadores supervisados clásicos: Support Vector Machines (SVM con kernel RBF/lineal), K-Nearest Neighbors (KNN), Random Forest y Multi-Layer Perceptron (MLP). |
| **RF-06** | **Persistencia del Modelo** | El mejor modelo seleccionado debe serializarse en disco en formato `.joblib` junto con sus metadatos de configuración. |
| **RF-07** | **Filtro de Disparo Temporal** | Para ejecutar una acción, el clasificador debe predecir el mismo gesto durante al menos 5 cuadros consecutivos, seguido de un periodo de enfriamiento (cooldown) de 1.0 segundo. |
| **RF-08** | **Integración Multimedia MPRIS** | Tras confirmarse un gesto, el sistema debe despachar el comando correspondiente a Spotify mediante `playerctl` de forma asíncrona o no bloqueante. |
| **RF-09** | **Clase Neutro Inoperante** | La detección de la clase "neutro" o la ausencia de mano en el cuadro no debe emitir ningún comando ni alterar el estado del reproductor. |

---

## 4. Requisitos No Funcionales (RNF)

| ID | Nombre | Especificación |
| :--- | :--- | :--- |
| **RNF-01** | **Rendimiento en CPU** | El tiempo de cómputo por cuadro (extracción MediaPipe + normalización + inferencia del clasificador) no debe exceder los 100 ms en un procesador Intel Core i5 de 8va generación (ThinkPad T480). |
| **RNF-02** | **Fluidez de la Demo** | El bucle principal de inferencia en tiempo real debe sostener una tasa de refresco $\ge 15$ FPS continuos en pantalla. |
| **RNF-03** | **Reproducibilidad Científica** | Las particiones de datos (train/val/test) y los clasificadores estocásticos deben fijar una semilla pseudoaleatoria determinista (`random_state=42`). |
| **RNF-04** | **Aislamiento de Dependencias** | El software no debe requerir privilegios de superusuario (`sudo`) ni alterar paquetes globales del sistema, confinándose a un entorno virtual de Python (`.venv`). |
| **RNF-05** | **Robustez Wayland / Linux** | La emisión de eventos multimedia debe apoyarse exclusivamente en la especificación D-Bus MPRIS a través de `playerctl`. |

---

## 5. Mapeo de Gestos y Acciones

| Clase / Gesto | Acción en Spotify | Comando Ejecutado | Comportamiento |
| :--- | :--- | :--- | :--- |
| **palma** | Reproducir / Pausar | `playerctl play-pause` | Alterna entre reproducción y pausa. |
| **senalar** | Pista siguiente | `playerctl next` | Avanza a la siguiente canción en la lista. |
| **paz** | Pista anterior | `playerctl previous` | Regresa al inicio o pista anterior. |
| **pulgar** | Subir volumen | `playerctl volume 0.1+` | Incrementa el volumen en 10%. |
| **puno** | Bajar volumen | `playerctl volume 0.1-` | Reduce el volumen en 10%. |
| **neutro** | Sin acción | *(Ninguno)* | Ignorado intencionalmente (posición de reposo). |

---

## 6. Criterios de Aceptación Medibles

Para dar por concluida la implementación del proyecto, se deben satisfacer cuantitativamente los siguientes umbrales:

1. **Volumen del Dataset:** $\ge 300$ cuadros válidos y no nulos por cada una de las 6 clases (total $\ge 1,800$ instancias en `data/dataset.csv`).
2. **Balance de Clases:** Ninguna clase debe representar menos del 14% ni más del 19% del total del conjunto de datos limpio.
3. **Métrica de Generalización:** $F_1\text{-score macro} \ge 0.90$ en el conjunto de prueba independiente (`test set`).
4. **Partición de Datos:** División estratificada estricta 70% entrenamiento, 15% validación y 15% prueba con `random_state=42`.
5. **Latencia Máxima:** Latencia de inferencia por cuadro $\le 100\text{ ms}$ medida en CPU ThinkPad T480.
6. **Cadencia Visual:** Frecuencia de despliegue $\ge 15\text{ FPS}$ en el script `demo.py`.
7. **Control de Falsos Positivos:** Exactitud del 100% en no-emisión de comandos cuando se predice la clase `neutro` o ante cuadros vacíos (sin detección de mano).
8. **Eficacia del Cooldown:** Intervalo temporal comprobado $\ge 1.0\text{ s}$ entre despachos sucesivos de comandos a `playerctl`.
