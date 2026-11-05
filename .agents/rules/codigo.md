# Reglas de Calidad y Estilo de Código

1. **Modularidad y Responsabilidad Única:**
   - Escribir scripts pequeños, desacoplados y enfocados en un único objetivo.
   - La lógica matemática, geométrica y de transformación pura reside centralizada en `scripts/utils.py`. No duplicar funciones entre scripts.

2. **Rutas Dinámicas y Portabilidad:**
   - Queda estrictamente prohibido codificar rutas absolutas (e.g. `/home/kaleb/...`).
   - Utilizar siempre `pathlib.Path(__file__).resolve().parent` para resolver rutas relativas a la raíz del proyecto.

3. **Tipado Estático y Claridad:**
   - Aplicar sugerencias de tipo (type hints PEP 484) en las firmas de funciones públicas:
     ```python
     def normalizar(landmarks: np.ndarray, handedness: str) -> np.ndarray:
     ```
   - Comentarios concisos y en español, explicando únicamente el "por qué" de decisiones no evidentes, nunca redundando en el "qué".

4. **Testing Automatizado Obligatorio:**
   - Toda función determinista (como `normalizar()` o la máquina de estados del filtro) debe contar con tests unitarios en `tests/` ejecutables con `pytest`.
   - Prohibido marcar una tarea de código como completada sin haber ejecutado su suite de pruebas correspondiente.

5. **Principio de Simplicidad (KISS / Ponytail):**
   - Evitar sobre-ingeniería, metaprogramación, o wrappers innecesarios.
   - Usar las herramientas estándar de scikit-learn, OpenCV y la biblioteca estándar de Python antes de introducir nuevas dependencias.
