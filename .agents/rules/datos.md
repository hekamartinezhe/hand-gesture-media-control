# Reglas de Gestión y Seguridad de Datos

1. **Inmutabilidad y Protección de `data/`:**
   - Queda terminantemente prohibido borrar, truncar, sobrescribir o alterar archivos dentro de `data/` sin autorización explícita previa del usuario.
   - Las operaciones de limpieza deben generar copias derivadas o guardar versiones preprocesadas sin destruir el archivo crudo original (`data/dataset.csv`).

2. **Esquema Estricto del Dataset:**
   - Todo archivo CSV de datos debe adherirse al esquema fijado en [`specs/design.md`](../../specs/design.md): 63 coordenadas numéricas (`x0`..`z20`), `label`, `handedness`, `session_id` y `timestamp`.
   - No se admiten nombres de columnas alternativos ni omisiones de metadatos.

3. **Determinismo y Reproducibilidad:**
   - Cualquier partición de datos (train / validation / test) o técnica de muestreo aleatorio (bootstrap, k-fold, etc.) debe utilizar obligatoriamente la semilla `random_state=42`.

4. **Autenticidad y Ética:**
   - El dataset debe ser generado íntegramente de forma propia por el equipo de trabajo.
   - Está prohibido descargar o inyectar datasets de landmarks de terceros (Kaggle, GitHub, etc.) a menos que se solicite como baseline comparativo.

5. **Privacidad por Diseño:**
   - El sistema almacena exclusivamente coordenadas geométricas normalizadas de la mano.
   - Jamás deben persistirse imágenes directas, cuadros de video crudo con rostros ni metadatos personales del usuario en el repositorio.
