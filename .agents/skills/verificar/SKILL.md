---
name: verificar
description: Ejecuta la suite de pruebas del proyecto y contrasta cuantitativamente los resultados frente a los criterios de aceptación medibles de specs/requirements.md.
---

# Procedimiento de Verificación y Auditoría de Criterios

Este skill realiza una inspección profunda del estado del proyecto para asegurar que el código cumple con los umbrales de aceptación formales.

## Pasos de Ejecución

1. **Ejecución de Tests Automatizados:**
   - Activa el entorno virtual y corre la suite completa de `pytest`:
     ```bash
     pytest tests/ -v
     ```
   - Verifica que no existan tests fallidos o advertencias críticas.

2. **Auditoría de Criterios de Aceptación Cuantitativos:**
   Revisa los siguientes parámetros contra el estado actual:
   - **Volumen de Datos:** Confirmar $\ge 300$ instancias por clase en `data/dataset.csv`.
   - **Balance de Clases:** Validar que la proporción de cada clase se encuentre en el rango 14% – 19%.
   - **Métricas ML:** Verificar en `models/metricas.json` que el $F_1\text{-score macro} \ge 0.90$.
   - **Tiempo de Inferencia:** Comprobar que la latencia por cuadro sea $\le 100\text{ ms}$ en CPU.
   - **Filtro Temporal:** Validar que el debounce requiere 5 frames idénticos y el cooldown sostiene 1.0 s.

3. **Generación del Reporte de Evidencias:**
   Produce un resumen estructurado para el usuario con el siguiente formato:

   | Criterio Evaluado | Umbral Requerido | Valor Obtenido | Estado |
   | :--- | :--- | :--- | :--- |
   | Tests unitarios | 100% pasando | X / X tests | ✅ Pasa / ❌ Falla |
   | Muestras por clase | $\ge 300$ | Min: N | ✅ Pasa / ❌ Falla |
   | F1-score macro test | $\ge 0.90$ | X.XX | ✅ Pasa / ❌ Falla |
   | Latencia de inferencia | $\le 100$ ms | XX ms | ✅ Pasa / ❌ Falla |
   | Comandos en clase neutro | 0 emitidos | 0 | ✅ Pasa / ❌ Falla |
