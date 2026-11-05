# Bitácora de Decisiones Arquitectónicas (Decisions Log)

Este registro documenta de manera inmutable las decisiones estructurales del proyecto, sus motivaciones y sus consecuencias operativas.

---

### D-001: Desarrollo Guiado por Especificaciones (Spec-Driven Development)
- **Fecha:** 2026-11-05
- **Estado:** Aprobada
- **Contexto:** En proyectos asistidos por IA, los historiales de chat son efímeros y propensos a alucinaciones de contexto a medida que el proyecto crece.
- **Decisión:** Establecer el directorio `specs/` como la única fuente de verdad. El agente opera en sesiones limpias ejecutando una tarea por sesión, verificando contra criterios cuantitativos y actualizando este log.
- **Consecuencias:** Mayor predictibilidad, modularidad y facilidad para continuar el desarrollo con cualquier modelo o de forma manual.

---

### D-002: Invarianza Geométrica mediante Normalización de Landmarks 0 y 9
- **Fecha:** 2026-11-05
- **Estado:** Aprobada
- **Contexto:** Si se alimentan las coordenadas crudas de MediaPipe al clasificador, el modelo aprende la posición absoluta en la pantalla y la distancia al lente, fallando si la mano se desplaza.
- **Decisión:** Implementar la función `normalizar()` restando la muñeca (landmark 0), escalando según la norma euclidiana hacia el nudillo del dedo medio (landmark 9), e invirtiendo el eje $x$ cuando la mano detectada sea izquierda.
- **Consecuencias:** El modelo aprende la conformación topológica del gesto, alcanzando alta robustez ante usuarios con manos de distinto tamaño o a diferentes distancias.

---

### D-003: Filtro Temporal Anti-Rebote (5 frames + 1.0 s Cooldown)
- **Fecha:** 2026-11-05
- **Estado:** Aprobada
- **Contexto:** En video en tiempo real existen micro-variaciones e imprecisiones de cuadro a cuadro (jittering) que podrían disparar múltiples comandos indeseados en Spotify (e.g. pausar y reanudar 10 veces en un segundo).
- **Decisión:** Incorporar una máquina de estados que exija 5 detecciones consecutivas idénticas antes de despachar el comando, seguida de un bloqueo obligatorio de 1.0 segundo.
- **Consecuencias:** La experiencia de usuario es deliberada y estable. Se introduce una latencia intencional imperceptible de ~150–200 ms que previene por completo el spam de comandos.

---

### D-004: Inclusión de la Clase "Neutro" como Estado de Reposo Activo
- **Fecha:** 2026-11-05
- **Estado:** Aprobada
- **Contexto:** Los usuarios mantienen la mano en reposo o realizan movimientos transitorios mientras gesticulan o hablan.
- **Decisión:** Definir una sexta clase explícita (`neutro`) en el dataset y entrenar al clasificador para reconocerla, garantizando que su predicción descarte cualquier acción sobre MPRIS.
- **Consecuencias:** Se eliminan los falsos positivos por movimientos cotidianos sin requerir umbrales heurísticos de probabilidad arbitrarios.
