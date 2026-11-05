---
name: nueva-tarea
description: Selecciona, contextualiza, implementa y verifica la siguiente tarea disponible en specs/tasks.md respetando el flujo de una tarea por sesión.
---

# Procedimiento para Implementar una Nueva Tarea

Este skill define el flujo de trabajo estándar para avanzar en el backlog del proyecto de forma atómica y confiable.

## Pasos de Ejecución

1. **Revisión del Backlog:**
   - Abre [`specs/tasks.md`](../../specs/tasks.md).
   - Localiza la primera tarea pendiente (`- [ ]`) cuyo owner sea `[AGENTE]` y cuyas dependencias previas estén completamente satisfechas (`[x]`).

2. **Carga Quirúrgica de Contexto:**
   - No leas todo el repositorio. Consulta únicamente los archivos listados en la tarea y las secciones pertinentes de:
     - [`specs/requirements.md`](../../specs/requirements.md) (para criterios de aceptación).
     - [`specs/design.md`](../../specs/design.md) (para arquitectura y algoritmos requeridos).

3. **Implementación de Alcance Mínimo:**
   - Escribe o edita el código estrictamente necesario para cumplir los requisitos de la tarea seleccionada.
   - Aplica las reglas en [`.agents/rules/codigo.md`](../rules/codigo.md).

4. **Verificación Automatizada:**
   - Ejecuta el **Comando de verificación** exacto especificado en la ficha de la tarea en `tasks.md`.
   - Si la prueba falla, corrige el código y vuelve a ejecutar hasta que pase exitosamente. No marques la tarea sin evidencia de éxito.

5. **Cierre de Tarea y Trazabilidad:**
   - Cambia el estado en `specs/tasks.md` a `[x]`.
   - Si durante la implementación se tomó una decisión arquitectónica no trivial, agrégala en [`specs/decisions.md`](../../specs/decisions.md).
   - Genera una propuesta de mensaje de commit siguiendo la convención:
     ```text
     <tipo>(<alcance>): descripción concisa de la tarea T-XX
     ```

> **Regla de Oro:** Detén tu ejecución al terminar la tarea. Nunca avances a la siguiente sin la confirmación o revisión del usuario.
