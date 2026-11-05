# Reglas de Seguridad y Ejecución del Entorno

1. **Aislamiento Estricto del Entorno:**
   - La instalación de paquetes de Python (`pip install`) se realiza exclusivamente dentro del entorno virtual local (`.venv/`).
   - Jamás ejecutar comandos de instalación con flags globales como `--break-system-packages` ni alterar el Python del sistema operativo.

2. **Privilegios de Sistema:**
   - Queda prohibido ejecutar de manera autónoma comandos que requieran privilegios de administrador (`sudo`), manipulación de módulos del kernel (`modprobe`) o administradores de paquetes del sistema (`dnf`, `apt`, `pacman`).
   - Si un paquete del sistema es indispensable (e.g. `v4l2loopback` o `playerctl`), se debe informar y solicitar al usuario que lo instale manualmente.

3. **Invocación Segura de Subprocesos:**
   - Al ejecutar llamadas al sistema para `playerctl`, usar siempre listas de argumentos en `subprocess.run(["playerctl", ...])` o `subprocess.Popen`.
   - Queda prohibido el uso de `shell=True` con cadenas de texto concatenadas para evitar inyecciones de comandos.

4. **Operación Local y Offline:**
   - La captura, entrenamiento, inferencia y control de Spotify operan de forma 100% local y offline.
   - Prohibido realizar peticiones HTTP de red externas no autorizadas o subir datos del proyecto a servicios en la nube.
