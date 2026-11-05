---
name: reporte
description: Compila y estructura el reporte académico final a partir de los artefactos, métricas y figuras exportadas por los experimentos reales, garantizando trazabilidad total.
---

# Procedimiento de Compilación del Reporte Académico

Este skill compila el documento formal de entrega utilizando exclusivamente cifras cuantitativas obtenidas del entrenamiento real.

## Regla de Integridad de Datos
**Queda terminantemente prohibido inventar métricas, aproximaciones no fundamentadas o valores sintéticos.** Toda cifra de rendimiento (Accuracy, Precision, Recall, F1-Score macro, tiempo de inferencia) debe ser leída directamente desde `models/metricas.json` o de las tablas exportadas por `notebooks/analisis.ipynb`.

## Pasos de Compilación

1. **Lectura de Artefactos de Evaluación:**
   - Cargar `models/metricas.json` y verificar los valores para cada clasificador evaluado (SVM, KNN, Random Forest, MLP).
   - Verificar la existencia de las figuras exportadas en `reporte/figuras/` o `models/`:
     - Matriz de confusión (`matriz_confusion.png`).
     - Proyección PCA bidimensional de clusters (`pca_clusters.png`).

2. **Estructuración del Documento (`reporte/reporte.md`):**
   - **1. Resumen y Objetivos:** Contexto del problema, motivación de interacción sin contacto y metas del proyecto.
   - **2. Metodología:**
     - Adquisición del dataset (protocolo, distribución y balance).
     - Preprocesamiento matemático (detallar la normalización geométrica invariante con fórmulas).
     - Clasificadores implementados y estrategia de partición estratificada 70/15/15.
     - Lógica de la máquina de estados del filtro de disparo temporal.
   - **3. Resultados Experimentales:**
     - Tabla comparativa de modelos con métricas reales.
     - Análisis de la matriz de confusión e interpretación de errores (e.g. confusión entre *señalar* y *paz*).
     - Gráfica PCA y análisis de separabilidad de clases en el espacio euclidiano normalizado.
   - **4. Integración de la Demo:**
     - Rendimiento medido en CPU ThinkPad T480 (FPS continuos, latencia promedio).
     - Validación del control sobre Spotify mediante `playerctl`.
   - **5. Conclusiones y Trabajo Futuro:** Reflexión crítica y posibles extensiones (gestos dinámicos, soporte bimanual).

3. **Verificación de Consistencia:**
   - Asegurarse de que las referencias cruzadas y los enlaces a figuras apunten a rutas relativas válidas.
